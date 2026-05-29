#!/usr/bin/env python3
"""
Train teacher-forced latent rollout maps for MeshFlowNet.

The model learns a regional latent trajectory for leads [3, 6, 9, 12, 15].
Only initialization-time local/global fields are used by the online path.
Future regional states are used only by the EMA target encoder during training.

Run:
    torchrun --standalone --nproc_per_node=4 pretrain_latent_rollout_encoder.py --epochs 60 --batch_size 4
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from met_jepa import (
    DATA_CACHE,
    LATENT_ROLLOUT_CHECKPOINT_PATH,
    ROLLOUT_LEADS,
    LatentRolloutDataset,
    LatentRolloutFeatureModel,
    aggressive_context_augment,
    channel_variance_diagnostics,
    compute_norm_stats,
    load_climatology,
    load_shared_arrays,
    save_latent_rollout_checkpoint,
    split_indices,
    vicreg_map_loss,
)


def parse_leads(text):
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher-forced latent rollout maps")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.06)
    parser.add_argument("--map_channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--leads", type=str, default=",".join(str(x) for x in ROLLOUT_LEADS))
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--latent_weight", type=float, default=1.0)
    parser.add_argument("--huber_weight", type=float, default=0.25)
    parser.add_argument("--skill_weight", type=float, default=0.45)
    parser.add_argument("--corr_weight", type=float, default=0.25)
    parser.add_argument("--multiscale_weight", type=float, default=0.12)
    parser.add_argument("--spectral_weight", type=float, default=0.01)
    parser.add_argument("--vicreg_weight", type=float, default=1.0)
    parser.add_argument("--pred_var_weight", type=float, default=0.05)
    parser.add_argument("--smooth_weight", type=float, default=0.0005)
    parser.add_argument("--map_l2_weight", type=float, default=0.0001)
    parser.add_argument("--tail_weight", type=float, default=1.0)
    parser.add_argument("--tail_threshold", type=float, default=1.5)
    parser.add_argument("--tf_full_epochs", type=int, default=10)
    parser.add_argument("--tf_decay_epochs", type=int, default=30)
    parser.add_argument("--tf_final_prob", type=float, default=0.50)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--train_eval_samples", type=int, default=544)
    parser.add_argument("--gate_margin", type=float, default=0.02)
    parser.add_argument("--max_r2_gap", type=float, default=0.15)
    parser.add_argument("--early_stop_after", type=int, default=20)
    parser.add_argument("--early_stop_checks", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0, help="Smoke-test limit; 0 means full epoch")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Smoke-test validation limit; 0 means full validation")
    parser.add_argument("--checkpoint", type=str, default=LATENT_ROLLOUT_CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message):
    if is_main_process():
        print(message, flush=True)


def global_augment(global_fields, channel_drop=0.10, noise_std=0.02):
    keep = (
        torch.rand((global_fields.shape[0], global_fields.shape[1], 1, 1), device=global_fields.device)
        > channel_drop
    ).float()
    return global_fields * keep + noise_std * torch.randn_like(global_fields)


def teacher_forcing_probability(epoch_num, args):
    if epoch_num <= args.tf_full_epochs:
        return 1.0
    if epoch_num <= args.tf_decay_epochs:
        denom = max(args.tf_decay_epochs - args.tf_full_epochs, 1)
        frac = (epoch_num - args.tf_full_epochs) / float(denom)
        return 1.0 - (1.0 - args.tf_final_prob) * frac
    return float(args.tf_final_prob)


def _valid_vectors(pred, y, mask, sample_idx):
    valid = mask[sample_idx].expand_as(pred[sample_idx]) > 0.5
    return pred[sample_idx][valid].float(), y[sample_idx][valid].float()


def pattern_skill_loss(pred, y, mask, tail_weight=1.0, tail_threshold=1.5):
    losses = []
    for b in range(pred.shape[0]):
        pv, tv = _valid_vectors(pred, y, mask, b)
        if pv.numel() < 2:
            continue
        weights = 1.0 + tail_weight * torch.sigmoid((tv.abs() - tail_threshold) / 0.35)
        wsum = weights.sum().clamp_min(1e-6)
        tmean = (weights * tv).sum() / wsum
        var = (weights * (tv - tmean).pow(2)).sum() / wsum
        mse = (weights * (pv - tv).pow(2)).sum() / wsum
        losses.append(mse / (var.detach() + 1e-6))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def pattern_corr_loss(pred, y, mask):
    losses = []
    for b in range(pred.shape[0]):
        pv, tv = _valid_vectors(pred, y, mask, b)
        if pv.numel() < 2:
            continue
        pv = pv - pv.mean()
        tv = tv - tv.mean()
        denom = pv.std(unbiased=False) * tv.std(unbiased=False) + 1e-6
        corr = (pv * tv).mean() / denom
        losses.append(1.0 - corr.clamp(-1.0, 1.0))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def multiscale_pattern_loss(pred, y, mask, scales=(2, 4, 8)):
    losses = []
    for scale in scales:
        p = F.avg_pool2d(pred.float(), kernel_size=scale, stride=scale)
        t = F.avg_pool2d(y.float(), kernel_size=scale, stride=scale)
        m = (F.avg_pool2d(mask.float(), kernel_size=scale, stride=scale) > 0.5).float()
        losses.append(pattern_skill_loss(p, t, m, tail_weight=0.0))
        losses.append(pattern_corr_loss(p, t, m))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def spectral_amplitude_loss(pred, y, mask):
    p = pred.float() * mask.float()
    t = y.float() * mask.float()
    p = p - p.mean(dim=(-2, -1), keepdim=True)
    t = t - t.mean(dim=(-2, -1), keepdim=True)
    p_amp = torch.log1p(torch.abs(torch.fft.rfft2(p.squeeze(1), norm="ortho")))
    t_amp = torch.log1p(torch.abs(torch.fft.rfft2(t.squeeze(1), norm="ortho")))
    diff = (p_amp - t_amp).abs()
    with torch.no_grad():
        hard = diff.detach() / (diff.detach().mean(dim=(-2, -1), keepdim=True) + 1e-6)
        hard = hard.clamp(0.25, 4.0)
    return (hard * diff).mean()


def masked_prediction_variance_loss(pred, y, mask):
    losses = []
    for b in range(pred.shape[0]):
        valid = mask[b].expand_as(pred[b]) > 0.5
        pv = pred[b][valid].float()
        tv = y[b][valid].float()
        if pv.numel() < 2:
            continue
        losses.append(F.relu(0.85 * tv.std(unbiased=False).detach() - pv.std(unbiased=False)).pow(2))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def total_variation(z):
    return (z[:, :, :, 1:, :] - z[:, :, :, :-1, :]).abs().mean() + (z[:, :, :, :, 1:] - z[:, :, :, :, :-1]).abs().mean()


def lead_weights(leads, device):
    values = torch.tensor([0.5 + 1.5 * (float(lead) / max(leads)) for lead in leads], device=device)
    return values / values.mean()


def rollout_loss_terms(z_preds, z_targets, y_preds, y_seq, mask_seq, weights, args):
    latent = z_preds.sum() * 0.0
    huber = z_preds.sum() * 0.0
    skill = z_preds.sum() * 0.0
    corr = z_preds.sum() * 0.0
    multi = z_preds.sum() * 0.0
    spectral = z_preds.sum() * 0.0
    pred_var = z_preds.sum() * 0.0
    denom = weights.sum().clamp_min(1e-6)

    for i in range(z_preds.shape[1]):
        w = weights[i]
        latent = latent + w * F.mse_loss(F.normalize(z_preds[:, i].float(), dim=1), F.normalize(z_targets[:, i].float(), dim=1))
        pred_i = y_preds[:, i]
        y_i = y_seq[:, i]
        mask_i = mask_seq[:, i]
        valid = mask_i.expand_as(pred_i) > 0.5
        huber = huber + w * F.huber_loss(pred_i[valid], y_i[valid], delta=2.0)
        skill = skill + w * pattern_skill_loss(pred_i, y_i, mask_i, args.tail_weight, args.tail_threshold)
        corr = corr + w * pattern_corr_loss(pred_i, y_i, mask_i)
        multi = multi + w * multiscale_pattern_loss(pred_i, y_i, mask_i)
        spectral = spectral + w * spectral_amplitude_loss(pred_i, y_i, mask_i)
        pred_var = pred_var + w * masked_prediction_variance_loss(pred_i, y_i, mask_i)

    latent = latent / denom
    huber = huber / denom
    skill = skill / denom
    corr = corr / denom
    multi = multi / denom
    spectral = spectral / denom
    pred_var = pred_var / denom
    vic, spatial_var = vicreg_map_loss(
        z_preds.reshape(-1, z_preds.shape[2], z_preds.shape[3], z_preds.shape[4]),
        variance_threshold=args.variance_threshold,
        var_weight=1.0,
        cov_weight=0.02,
    )
    smooth = total_variation(z_preds)
    map_l2 = z_preds.float().pow(2).mean()
    total = (
        args.latent_weight * latent
        + args.huber_weight * huber
        + args.skill_weight * skill
        + args.corr_weight * corr
        + args.multiscale_weight * multi
        + args.spectral_weight * spectral
        + args.vicreg_weight * vic
        + args.pred_var_weight * pred_var
        + args.smooth_weight * smooth
        + args.map_l2_weight * map_l2
    )
    return total, {
        "latent": latent,
        "huber": huber,
        "skill": skill,
        "corr": corr,
        "multi": multi,
        "spectral": spectral,
        "vic": vic,
        "pred_var": pred_var,
        "spatial_var": spatial_var,
    }


@torch.inference_mode()
def evaluate(model, loader, device, leads, threshold, max_batches=0):
    model.eval()
    n_leads = len(leads)
    ss_res = np.zeros(n_leads, dtype=np.float64)
    truth_sum = np.zeros(n_leads, dtype=np.float64)
    truth_sq_sum = np.zeros(n_leads, dtype=np.float64)
    n_pix = np.zeros(n_leads, dtype=np.float64)
    huber_sums = np.zeros(n_leads, dtype=np.float64)
    batches = 0
    latent_losses = []
    diag_accum = []

    for context, target_contexts, global_fields, y_seq, mask_seq, _, _ in loader:
        if max_batches and batches >= max_batches:
            break
        context = context.to(device, non_blocking=True)
        target_contexts = target_contexts.to(device, non_blocking=True)
        global_fields = global_fields.to(device, non_blocking=True)
        y_seq = y_seq.to(device, non_blocking=True)
        mask_seq = mask_seq.to(device, non_blocking=True)

        z_preds, z_targets, y_preds = model(context, global_fields, target_contexts, teacher_forcing_prob=0.0)
        latent_losses.append(float(F.mse_loss(F.normalize(z_preds.float(), dim=2), F.normalize(z_targets.float(), dim=2)).item()))
        diag_accum.append(channel_variance_diagnostics(z_preds[:, -1], threshold))

        for i in range(n_leads):
            pred_i = y_preds[:, i]
            y_i = y_seq[:, i]
            mask_i = mask_seq[:, i]
            valid = mask_i.expand_as(pred_i) > 0.5
            diff = pred_i[valid].float() - y_i[valid].float()
            truth = y_i[valid].float()
            ss_res[i] += float((diff ** 2).sum().item())
            truth_sum[i] += float(truth.sum().item())
            truth_sq_sum[i] += float((truth ** 2).sum().item())
            n_pix[i] += float(truth.numel())
            huber_sums[i] += float(F.huber_loss(pred_i[valid], y_i[valid], delta=2.0).item())
        batches += 1

    metrics = {}
    for i, lead in enumerate(leads):
        mean = truth_sum[i] / max(n_pix[i], 1.0)
        ss_tot = truth_sq_sum[i] - n_pix[i] * mean * mean
        r2 = float(1.0 - ss_res[i] / (ss_tot + 1e-8))
        rmse = float(np.sqrt(ss_res[i] / max(n_pix[i], 1.0)))
        huber = float(huber_sums[i] / max(batches, 1))
        metrics[int(lead)] = {"r2": r2, "rmse": rmse, "huber": huber}

    per_channel = np.stack([d["per_channel_variance"] for d in diag_accum], axis=0).mean(axis=0)
    collapsed = per_channel < threshold
    diag = {
        "min_channel_variance": float(per_channel.min()),
        "mean_channel_variance": float(per_channel.mean()),
        "collapsed_channels": int(collapsed.sum()),
        "active_channels": int((~collapsed).sum()),
        "per_channel_variance": per_channel.astype(np.float32),
    }
    model.train()
    return metrics, float(np.mean(latent_losses)) if latent_losses else np.inf, diag


@torch.inference_mode()
def climatology_zero_r2(loader, device, final_lead_index=-1, max_batches=0):
    truth_sum = 0.0
    truth_sq_sum = 0.0
    n_pix = 0.0
    batches = 0
    for _, _, _, y_seq, mask_seq, _, _ in loader:
        if max_batches and batches >= max_batches:
            break
        y = y_seq[:, final_lead_index].to(device, non_blocking=True)
        mask = mask_seq[:, final_lead_index].to(device, non_blocking=True)
        valid = mask.expand_as(y) > 0.5
        truth = y[valid].float()
        truth_sum += float(truth.sum().item())
        truth_sq_sum += float((truth ** 2).sum().item())
        n_pix += float(truth.numel())
        batches += 1
    truth_mean = truth_sum / max(n_pix, 1.0)
    ss_tot = truth_sq_sum - n_pix * truth_mean * truth_mean
    return float(1.0 - truth_sq_sum / (ss_tot + 1e-8))


def main():
    args = parse_args()
    leads = parse_leads(args.leads)
    if leads[-1] != 15:
        raise ValueError(f"The final rollout lead must be 15 days; got leads={leads}")

    ddp, rank, world_size, local_rank = init_distributed()
    set_seed(args.seed + rank)
    if is_main_process():
        os.makedirs(DATA_CACHE, exist_ok=True)
    if ddp:
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    log(f"DDP: enabled={ddp}, rank={rank}, world_size={world_size}, local_rank={local_rank}")
    log("Loading cached data and climatology...")
    if ddp and not is_main_process():
        dist.barrier()
    shared = load_shared_arrays(mmap=True)
    if ddp and is_main_process():
        dist.barrier()
    clim = load_climatology()
    train_indices, val_indices, test_indices, _ = split_indices(np.array(shared["time_values"]))
    log(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    log(f"Rollout leads: {leads}")

    log("Computing/using normalization stats from train years only...")
    stats = compute_norm_stats(shared, clim, train_indices)
    train_ds = LatentRolloutDataset(shared, clim, stats, train_indices, leads=leads, return_targets=True)
    val_ds = LatentRolloutDataset(shared, clim, stats, val_indices, leads=leads, return_targets=True)
    train_eval_ds = LatentRolloutDataset(
        shared,
        clim,
        stats,
        train_indices[: min(args.train_eval_samples, len(train_indices))],
        leads=leads,
        return_targets=True,
    )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    sample_context, _, sample_global, _, _, _, _ = train_ds[0]
    raw_model = LatentRolloutFeatureModel(
        local_channels=sample_context.shape[0],
        global_channels=sample_global.shape[0],
        map_channels=args.map_channels,
        hidden=args.hidden,
        dropout=args.dropout,
        leads=leads,
    ).to(device)
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False) if ddp else raw_model
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    log(f"Local context channels: {sample_context.shape[0]}")
    log(f"Global teleconnection channels: {sample_global.shape[0]}")
    log(f"Latent rollout map channels: {args.map_channels}")
    log(f"Per-GPU batch size: {args.batch_size}; effective batch size: {args.batch_size * world_size}")
    log(
        "Teacher forcing: "
        f"epochs 1-{args.tf_full_epochs} use true previous latent, "
        f"then decay through epoch {args.tf_decay_epochs} to {args.tf_final_prob:.2f}."
    )
    log("Export path uses only context/global fields at t; target encoder is training-only.")

    clim_r2 = climatology_zero_r2(val_loader, device, max_batches=args.max_val_batches) if is_main_process() else None
    if is_main_process():
        print(f"Lead-15 climatology-only anomaly-space val R2: {clim_r2:.6f}", flush=True)

    best_val_r2 = -np.inf
    best_score = -np.inf
    bad_checks = 0
    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        stop_now = False
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        tf_prob = teacher_forcing_probability(epoch_num, args)
        weights = lead_weights(leads, device)
        losses, hubers, latents = [], [], []
        spatial_vars = []
        pbar = tqdm(
            train_loader,
            desc=f"Latent rollout epoch {epoch_num}/{args.epochs}",
            mininterval=10.0,
            disable=not is_main_process(),
        )
        for batch_idx, (context, target_contexts, global_fields, y_seq, mask_seq, _, _) in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            context = context.to(device, non_blocking=True)
            target_contexts = target_contexts.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            y_seq = y_seq.to(device, non_blocking=True)
            mask_seq = mask_seq.to(device, non_blocking=True)

            context_aug = aggressive_context_augment(context)
            global_aug = global_augment(global_fields)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                z_preds, z_targets, y_preds = model(context_aug, global_aug, target_contexts, teacher_forcing_prob=tf_prob)
                loss, terms = rollout_loss_terms(z_preds, z_targets, y_preds, y_seq, mask_seq, weights, args)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            raw_model.update_target_encoder(decay=args.ema_decay)

            losses.append(float(loss.detach().float().item()))
            hubers.append(float(terms["huber"].detach().float().item()))
            latents.append(float(terms["latent"].detach().float().item()))
            spatial_vars.append(channel_variance_diagnostics(z_preds[:, -1].detach(), args.variance_threshold)["per_channel_variance"])
            pbar.set_postfix({
                "loss": f"{losses[-1]:.4f}",
                "latent": f"{latents[-1]:.4f}",
                "huber": f"{hubers[-1]:.4f}",
                "tf": f"{tf_prob:.2f}",
            })

        if is_main_process():
            mean_var = np.stack(spatial_vars, axis=0).mean(axis=0) if spatial_vars else np.zeros(args.map_channels, dtype=np.float32)
            collapsed = mean_var < args.variance_threshold
            print(
                f"Epoch {epoch_num}: train_loss={np.mean(losses):.5f}, latent={np.mean(latents):.5f}, "
                f"huber={np.mean(hubers):.5f}, tf_prob={tf_prob:.3f}, "
                f"min_var={mean_var.min():.5f}, mean_var={mean_var.mean():.5f}, "
                f"active_channels={(~collapsed).sum()}/{args.map_channels}",
                flush=True,
            )

        if epoch_num % args.eval_freq == 0 or epoch == 0:
            if ddp:
                dist.barrier()
            if is_main_process():
                val_metrics, val_latent, diag = evaluate(raw_model, val_loader, device, leads, args.variance_threshold, max_batches=args.max_val_batches)
                train_metrics, _, _ = evaluate(raw_model, train_eval_loader, device, leads, args.variance_threshold, max_batches=args.max_val_batches)
                val15 = val_metrics[15]["r2"]
                train15 = train_metrics[15]["r2"]
                gap = train15 - val15
                gate = (val15 >= clim_r2 + args.gate_margin) and (gap < args.max_r2_gap) and diag["collapsed_channels"] == 0
                score = val15 - max(gap - args.max_r2_gap, 0.0)
                lead_text = " ".join([f"L{lead}:R2={val_metrics[lead]['r2']:+.4f}" for lead in leads])
                print(f"  Val latent loss={val_latent:.5f}; {lead_text}", flush=True)
                print(
                    f"  Lead15: val R2={val15:+.5f}, trainEval R2={train15:+.5f}, "
                    f"gap={gap:+.5f}, RMSE={val_metrics[15]['rmse']:.5f}, Huber={val_metrics[15]['huber']:.5f}",
                    flush=True,
                )
                print(
                    f"  Map variance: min={diag['min_channel_variance']:.5f}, "
                    f"mean={diag['mean_channel_variance']:.5f}, collapsed={diag['collapsed_channels']}, "
                    f"active={diag['active_channels']}/{args.map_channels}",
                    flush=True,
                )
                print(
                    f"  Gate: {'PASS' if gate else 'FAIL'} "
                    f"(need val15 R2 >= {clim_r2 + args.gate_margin:.5f}, gap < {args.max_r2_gap:.2f})",
                    flush=True,
                )

                improved = val15 > best_val_r2 + args.min_delta
                if improved and diag["collapsed_channels"] == 0:
                    best_val_r2 = val15
                    best_score = score
                    bad_checks = 0
                    save_latent_rollout_checkpoint(
                        args.checkpoint,
                        raw_model,
                        optimizer,
                        epoch_num,
                        stats,
                        train_indices,
                        val_indices,
                        vars(args),
                        best_score,
                        best_val_r2,
                    )
                    print(f"  Saved best latent rollout encoder to {args.checkpoint} (val15_r2={val15:+.5f})", flush=True)
                elif epoch_num >= args.early_stop_after:
                    bad_checks += 1
                    print(f"  No lead-15 validation R2 improvement: {bad_checks}/{args.early_stop_checks}", flush=True)
                    if bad_checks >= args.early_stop_checks:
                        stop_now = True
                        print("  Early stopping: lead-15 validation R2 did not improve.", flush=True)
            if ddp:
                stop_tensor = torch.tensor(int(stop_now), device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())
                dist.barrier()
            if stop_now:
                break

    if is_main_process():
        if not os.path.exists(args.checkpoint):
            save_latent_rollout_checkpoint(
                args.checkpoint,
                raw_model,
                optimizer,
                args.epochs,
                stats,
                train_indices,
                val_indices,
                vars(args),
                best_score,
                best_val_r2,
            )
            print(f"Saved final latent rollout checkpoint for inspection: {args.checkpoint}", flush=True)
        print(f"Best lead-15 validation R2 seen: {best_val_r2:+.5f}", flush=True)
        print(f"Best checkpoint: {args.checkpoint}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
