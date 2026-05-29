#!/usr/bin/env python3
"""
Train a supervised, heavily regularized spatial feature encoder.

The model uses only initialization-time inputs:
  - regional anomaly/physics/static context at t
  - global teleconnection fields at t

It predicts the t+15 climatology-normalized anomaly through a deliberately
small 1x1 decoder. The exported artifact is the 32-channel feature map before
that decoder, so MeshFlowNet can consume it as spatial input.

Run:
    python3 -u pretrain_supervised_spatial_encoder.py --epochs 60 --batch_size 8
"""

import argparse
import os
import random
from copy import deepcopy

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
    LEAD_TIME,
    SUPERVISED_CHECKPOINT_PATH,
    MetJEPADataset,
    SupervisedSpatialFeatureModel,
    channel_variance_diagnostics,
    compute_norm_stats,
    load_climatology,
    load_shared_arrays,
    random_blur_mix,
    random_spatial_jitter,
    save_supervised_checkpoint,
    split_indices,
    vicreg_map_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised spatial feature maps")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.08)
    parser.add_argument("--map_channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--feature_reg_weight", type=float, default=1.0)
    parser.add_argument("--huber_weight", type=float, default=0.25)
    parser.add_argument("--skill_weight", type=float, default=0.50)
    parser.add_argument("--corr_weight", type=float, default=0.25)
    parser.add_argument("--multiscale_weight", type=float, default=0.15)
    parser.add_argument("--spectral_weight", type=float, default=0.01)
    parser.add_argument("--tail_weight", type=float, default=1.0)
    parser.add_argument("--tail_threshold", type=float, default=1.5)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--consistency_weight", type=float, default=0.10)
    parser.add_argument("--map_consistency_weight", type=float, default=0.02)
    parser.add_argument("--consistency_ramp_epochs", type=int, default=10)
    parser.add_argument("--pred_var_weight", type=float, default=0.05)
    parser.add_argument("--smooth_weight", type=float, default=0.0005)
    parser.add_argument("--map_l2_weight", type=float, default=0.0001)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--train_eval_samples", type=int, default=544)
    parser.add_argument("--gate_margin", type=float, default=0.02)
    parser.add_argument("--max_r2_gap", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=SUPERVISED_CHECKPOINT_PATH)
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


def supervised_context_augment(context, noise_std=0.035):
    """Strong, but less destructive than JEPA masking."""
    b, c, h, w = context.shape
    device = context.device
    x = random_spatial_jitter(context, max_shift=2)

    probs = torch.full((c,), 0.12, device=device)
    probs[:3] = 0.35
    keep = (torch.rand((b, c, 1, 1), device=device) > probs.view(1, c, 1, 1)).float()
    x = x * keep

    groups = [(0, 3, 0.20), (3, 12, 0.10), (12, c, 0.05)]
    for start, end, p in groups:
        drop = (torch.rand((b, 1, 1, 1), device=device) < p).float()
        x[:, start:end] = x[:, start:end] * (1.0 - drop)

    for _ in range(2):
        bh = torch.randint(max(8, h // 10), max(9, h // 4), (b,), device=device)
        bw = torch.randint(max(8, w // 10), max(9, w // 4), (b,), device=device)
        y0 = torch.randint(0, max(1, h - int(bh.max().item())), (b,), device=device)
        x0 = torch.randint(0, max(1, w - int(bw.max().item())), (b,), device=device)
        for i in range(b):
            y1 = min(h, int(y0[i] + bh[i]))
            x1 = min(w, int(x0[i] + bw[i]))
            x[i, :, int(y0[i]):y1, int(x0[i]):x1] = 0.0

    x = random_blur_mix(x, p=0.10)
    return x + noise_std * torch.randn_like(x)


def global_augment(global_fields, channel_drop=0.12, noise_std=0.02):
    keep = (torch.rand((global_fields.shape[0], global_fields.shape[1], 1, 1), device=global_fields.device) > channel_drop).float()
    return global_fields * keep + noise_std * torch.randn_like(global_fields)


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    student = model.state_dict()
    teacher = ema_model.state_dict()
    for key, value in teacher.items():
        source = student[key]
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            value.copy_(source)


def _valid_vectors(pred, y, mask, sample_idx):
    valid = mask[sample_idx].expand_as(pred[sample_idx]) > 0.5
    pv = pred[sample_idx][valid].float()
    tv = y[sample_idx][valid].float()
    return pv, tv


def pattern_skill_loss(pred, y, mask, tail_weight=1.0, tail_threshold=1.5):
    """Per-sample normalized MSE, effectively optimizing skill rather than raw error scale."""
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
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def pattern_corr_loss(pred, y, mask):
    """One-minus spatial anomaly correlation, averaged per sample."""
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
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def multiscale_pattern_loss(pred, y, mask, scales=(2, 4, 8)):
    """Image/video-style coarse-structure loss to suppress pixel-noise gradients."""
    losses = []
    for scale in scales:
        p = F.avg_pool2d(pred.float(), kernel_size=scale, stride=scale)
        t = F.avg_pool2d(y.float(), kernel_size=scale, stride=scale)
        m = F.avg_pool2d(mask.float(), kernel_size=scale, stride=scale)
        m = (m > 0.5).float()
        losses.append(pattern_skill_loss(p, t, m, tail_weight=0.0))
        losses.append(pattern_corr_loss(p, t, m))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def spectral_amplitude_loss(pred, y, mask):
    """Compare 2D frequency amplitudes, inspired by focal/frequency reconstruction losses."""
    p = pred.float() * mask.float()
    t = y.float() * mask.float()
    p = p - p.mean(dim=(-2, -1), keepdim=True)
    t = t - t.mean(dim=(-2, -1), keepdim=True)
    p_fft = torch.fft.rfft2(p.squeeze(1), norm="ortho")
    t_fft = torch.fft.rfft2(t.squeeze(1), norm="ortho")
    p_amp = torch.log1p(torch.abs(p_fft))
    t_amp = torch.log1p(torch.abs(t_fft))
    diff = (p_amp - t_amp).abs()
    with torch.no_grad():
        hard = diff.detach()
        hard = hard / (hard.mean(dim=(-2, -1), keepdim=True) + 1e-6)
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
        pstd = pv.std(unbiased=False)
        tstd = tv.std(unbiased=False).detach()
        losses.append(F.relu(0.85 * tstd - pstd).pow(2))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def total_variation(z):
    dy = (z[:, :, 1:, :] - z[:, :, :-1, :]).abs().mean()
    dx = (z[:, :, :, 1:] - z[:, :, :, :-1]).abs().mean()
    return dx + dy


@torch.inference_mode()
def evaluate(model, loader, device, threshold, max_samples=0):
    model.eval()
    ss_res = 0.0
    truth_sum = 0.0
    truth_sq_sum = 0.0
    n_pix = 0.0
    huber_losses = []
    diag_accum = []
    count = 0
    for context, _, global_fields, y, mask, _ in loader:
        if max_samples and count >= max_samples:
            break
        context = context.to(device, non_blocking=True)
        global_fields = global_fields.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred, z = model(context, global_fields)
        valid = mask.expand_as(pred) > 0.5
        diff = pred[valid].float() - y[valid].float()
        truth = y[valid].float()
        ss_res += float((diff ** 2).sum().item())
        truth_sum += float(truth.sum().item())
        truth_sq_sum += float((truth ** 2).sum().item())
        n_pix += float(truth.numel())
        huber_losses.append(float(F.huber_loss(pred[valid], y[valid], delta=2.0).item()))
        diag_accum.append(channel_variance_diagnostics(z, threshold))
        count += context.shape[0]

    truth_mean = truth_sum / max(n_pix, 1.0)
    ss_tot = truth_sq_sum - n_pix * truth_mean * truth_mean
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    rmse = float(np.sqrt(ss_res / max(n_pix, 1.0)))
    huber = float(np.mean(huber_losses)) if huber_losses else np.inf
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
    return r2, rmse, huber, diag


@torch.inference_mode()
def climatology_zero_r2(loader, device):
    truth_sum = 0.0
    truth_sq_sum = 0.0
    n_pix = 0.0
    for _, _, _, y, mask, _ in loader:
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        valid = mask.expand_as(y) > 0.5
        truth = y[valid].float()
        truth_sum += float(truth.sum().item())
        truth_sq_sum += float((truth ** 2).sum().item())
        n_pix += float(truth.numel())
    truth_mean = truth_sum / max(n_pix, 1.0)
    ss_tot = truth_sq_sum - n_pix * truth_mean * truth_mean
    # Prediction is zero anomaly, so residual is truth.
    ss_res = truth_sq_sum
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def main():
    args = parse_args()
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
    log(f"Lead time: {LEAD_TIME} days")

    log("Computing/using normalization stats from train years only...")
    stats = compute_norm_stats(shared, clim, train_indices)

    train_ds = MetJEPADataset(shared, clim, stats, train_indices, return_target=True)
    val_ds = MetJEPADataset(shared, clim, stats, val_indices, return_target=True)
    train_eval_ds = MetJEPADataset(shared, clim, stats, train_indices[: min(args.train_eval_samples, len(train_indices))], return_target=True)
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

    sample_context, _, sample_global, _, _, _ = train_ds[0]
    raw_model = SupervisedSpatialFeatureModel(
        local_channels=sample_context.shape[0],
        global_channels=sample_global.shape[0],
        map_channels=args.map_channels,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    ema_model = deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False) if ddp else raw_model
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    log(f"Local context channels: {sample_context.shape[0]}")
    log(f"Global teleconnection channels: {sample_global.shape[0]}")
    log(f"Supervised map channels: {args.map_channels}")
    log(f"Per-GPU batch size: {args.batch_size}; effective batch size: {args.batch_size * world_size}")
    log("Regularization: Mean Teacher consistency, pattern skill/correlation, multiscale/spectral losses, VICReg maps, TV, weight decay.")

    clim_r2 = climatology_zero_r2(val_loader, device) if is_main_process() else None
    if is_main_process():
        print(f"Climatology-only anomaly-space val R2: {clim_r2:.6f}", flush=True)

    best_val_r2 = -np.inf
    best_score = -np.inf
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        losses, hubers, mses = [], [], []
        spatial_vars = []
        pbar = tqdm(
            train_loader,
            desc=f"Supervised feature epoch {epoch + 1}/{args.epochs}",
            mininterval=10.0,
            disable=not is_main_process(),
        )
        for context, _, global_fields, y, mask, _ in pbar:
            context = context.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            context_aug = supervised_context_augment(context)
            global_aug = global_augment(global_fields)
            consistency_ramp = min(1.0, float(epoch + 1) / max(float(args.consistency_ramp_epochs), 1.0))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    teacher_pred, teacher_z = ema_model(context, global_fields)
                pred, z = model(context_aug, global_aug)
                valid = mask.expand_as(pred) > 0.5
                huber = F.huber_loss(pred[valid], y[valid], delta=2.0)
                mse = F.mse_loss(pred[valid], y[valid])
                skill = pattern_skill_loss(
                    pred,
                    y,
                    mask,
                    tail_weight=args.tail_weight,
                    tail_threshold=args.tail_threshold,
                )
                corr_loss = pattern_corr_loss(pred, y, mask)
                ms_loss = multiscale_pattern_loss(pred, y, mask)
                spec_loss = spectral_amplitude_loss(pred, y, mask)
                vic_loss, spatial_var = vicreg_map_loss(
                    z,
                    variance_threshold=args.variance_threshold,
                    var_weight=1.0,
                    cov_weight=0.02,
                )
                pred_var_loss = masked_prediction_variance_loss(pred, y, mask)
                smooth = total_variation(z)
                map_l2 = z.float().pow(2).mean()
                consistency = F.mse_loss(pred.float(), teacher_pred.float())
                map_consistency = F.mse_loss(
                    F.normalize(z.float(), dim=1),
                    F.normalize(teacher_z.float(), dim=1),
                )
                loss = (
                    args.huber_weight * huber
                    + args.skill_weight * skill
                    + args.corr_weight * corr_loss
                    + args.multiscale_weight * ms_loss
                    + args.spectral_weight * spec_loss
                    + args.feature_reg_weight * vic_loss
                    + args.pred_var_weight * pred_var_loss
                    + args.smooth_weight * smooth
                    + args.map_l2_weight * map_l2
                    + consistency_ramp * args.consistency_weight * consistency
                    + consistency_ramp * args.map_consistency_weight * map_consistency
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema_model(ema_model, raw_model, args.ema_decay)

            losses.append(float(loss.detach().float().item()))
            hubers.append(float(huber.detach().float().item()))
            mses.append(float(mse.detach().float().item()))
            spatial_vars.append(spatial_var.detach().cpu().numpy())
            pbar.set_postfix({
                "loss": f"{losses[-1]:.4f}",
                "huber": f"{hubers[-1]:.4f}",
                "skill": f"{float(skill.detach().float().item()):.3f}",
                "corr_r": f"{1.0 - float(corr_loss.detach().float().item()):+.3f}",
            })

        if is_main_process():
            pred_vars = np.concatenate(spatial_vars, axis=0)
            mean_var = pred_vars.mean(axis=0)
            collapsed = mean_var < args.variance_threshold
            print(
                f"Epoch {epoch + 1}: train_loss={np.mean(losses):.5f}, "
                f"huber={np.mean(hubers):.5f}, mse={np.mean(mses):.5f}, "
                f"min_var={mean_var.min():.5f}, mean_var={mean_var.mean():.5f}, "
                f"active_channels={(~collapsed).sum()}/{args.map_channels}",
                flush=True,
            )

        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            if ddp:
                dist.barrier()
            if is_main_process():
                val_r2, val_rmse, val_huber, diag = evaluate(ema_model, val_loader, device, args.variance_threshold)
                train_r2, train_rmse, _, _ = evaluate(
                    ema_model, train_eval_loader, device, args.variance_threshold, max_samples=args.train_eval_samples
                )
                gap = train_r2 - val_r2
                gate = (val_r2 >= clim_r2 + args.gate_margin) and (gap < args.max_r2_gap) and diag["collapsed_channels"] == 0
                score = val_r2 - max(gap - args.max_r2_gap, 0.0)
                print(
                    f"  Val R2={val_r2:.5f}, RMSE={val_rmse:.5f}, Huber={val_huber:.5f}, "
                    f"TrainEval R2={train_r2:.5f}, gap={gap:+.5f}",
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
                    f"(need val R2 >= {clim_r2 + args.gate_margin:.5f}, gap < {args.max_r2_gap:.2f})",
                    flush=True,
                )
                if score > best_score:
                    best_score = score
                    best_val_r2 = val_r2
                    save_supervised_checkpoint(
                        args.checkpoint,
                        ema_model,
                        optimizer,
                        epoch + 1,
                        stats,
                        train_indices,
                        val_indices,
                        vars(args),
                        best_score,
                    )
                    print(f"  Saved best supervised encoder to {args.checkpoint} (val_r2={val_r2:.5f}, score={score:.5f})", flush=True)
            if ddp:
                dist.barrier()

    if is_main_process():
        print(f"Best validation R2 seen: {best_val_r2:.5f}")
        print(f"Best checkpoint: {args.checkpoint}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
