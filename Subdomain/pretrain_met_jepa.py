#!/usr/bin/env python3
"""
Pretrain Met-JEPA spatial feature maps for 15-day heatwave prediction.

Run on HiPerGator:
    python3 -u Subdomain/pretrain_met_jepa.py --epochs 100 --batch_size 8
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from met_jepa import (
    DATA_CACHE,
    JEPA_CHECKPOINT_PATH,
    LEAD_TIME,
    MetJEPADataset,
    MetJEPAModel,
    aggressive_context_augment,
    channel_variance_diagnostics,
    compute_norm_stats,
    load_climatology,
    load_shared_arrays,
    save_checkpoint,
    split_indices,
    vicreg_map_loss,
    weak_target_augment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Met-JEPA spatial maps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--map_channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--vicreg_weight", type=float, default=1.0)
    parser.add_argument("--probe_weight", type=float, default=0.0,
                        help="Optional auxiliary Huber loss from predicted JEPA maps to target anomaly.")
    parser.add_argument("--selection_metric", type=str, default="auto",
                        choices=["auto", "latent", "probe", "combined"],
                        help="Checkpoint selection metric. auto uses probe when --probe_weight > 0.")
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--collapse_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=JEPA_CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latent_loss(z_pred, z_target):
    z_pred_n = F.normalize(z_pred.float(), dim=1)
    z_tgt_n = F.normalize(z_target.float(), dim=1)
    return F.mse_loss(z_pred_n, z_tgt_n)


class AuxiliaryProbeHead(torch.nn.Module):
    """Tiny 1x1 decoder used only during JEPA pretraining to align maps with target anomaly."""

    def __init__(self, in_channels):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, z):
        return self.proj(z).clamp(-4.0, 4.0)


@torch.inference_mode()
def evaluate(model, loader, device, threshold, probe_head=None, probe_weight=0.0):
    model.eval()
    if probe_head is not None:
        probe_head.eval()
    losses = []
    probe_losses = []
    combined_losses = []
    diag_accum = []
    for context, target, global_fields, y, mask, _ in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        global_fields = global_fields.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        z_pred, z_target = model(context, target, global_fields)
        loss = latent_loss(z_pred, z_target)
        probe_loss = z_pred.sum() * 0.0
        if probe_head is not None:
            probe_pred = probe_head(z_pred)
            valid = mask.expand_as(probe_pred) > 0.5
            probe_loss = F.huber_loss(probe_pred[valid], y[valid], delta=2.0)
        losses.append(float(loss.item()))
        probe_losses.append(float(probe_loss.item()))
        combined_losses.append(float((loss + probe_weight * probe_loss).item()))
        diag_accum.append(channel_variance_diagnostics(z_pred, threshold))
    mean_loss = float(np.mean(losses)) if losses else np.inf
    mean_probe = float(np.mean(probe_losses)) if probe_losses else np.inf
    mean_combined = float(np.mean(combined_losses)) if combined_losses else np.inf
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
    if probe_head is not None:
        probe_head.train()
    return mean_loss, mean_probe, mean_combined, diag


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(DATA_CACHE, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading cached data and climatology...")
    shared = load_shared_arrays(mmap=True)
    clim = load_climatology()
    train_indices, val_indices, test_indices, _ = split_indices(np.array(shared["time_values"]))
    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    print(f"Lead time: {LEAD_TIME} days")

    print("Computing/using Met-JEPA normalization stats from train years only...")
    stats = compute_norm_stats(shared, clim, train_indices)

    train_ds = MetJEPADataset(shared, clim, stats, train_indices, return_target=True)
    val_ds = MetJEPADataset(shared, clim, stats, val_indices, return_target=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_context, _, sample_global, _, _, _ = train_ds[0]
    model = MetJEPAModel(
        local_channels=sample_context.shape[0],
        global_channels=sample_global.shape[0],
        map_channels=args.map_channels,
        hidden=args.hidden,
    ).to(device)
    probe_head = AuxiliaryProbeHead(args.map_channels).to(device) if args.probe_weight > 0 else None
    opt_params = list(model.parameters())
    if probe_head is not None:
        opt_params += list(probe_head.parameters())
    optimizer = AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print(f"Local context channels: {sample_context.shape[0]}")
    print(f"Global teleconnection channels: {sample_global.shape[0]}")
    print(f"JEPA map channels: {args.map_channels}")
    print("Aggressive context masking is enabled: block masks, channel dropout, jitter, and noise.")
    if probe_head is not None:
        print(f"Auxiliary target-anomaly probe loss enabled with weight {args.probe_weight}")
    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = "probe" if probe_head is not None else "latent"
    print(f"Checkpoint selection metric: {selection_metric}")

    best_val = np.inf
    collapse_bad_checks = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_pred = []
        pbar = tqdm(train_loader, desc=f"JEPA epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for context, target, global_fields, y, mask, _ in pbar:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            context_aug = aggressive_context_augment(context)
            target_aug = weak_target_augment(target)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                z_pred, z_target = model(context_aug, target_aug, global_fields)
                pred_loss = latent_loss(z_pred, z_target)
                vic_loss, spatial_var = vicreg_map_loss(
                    z_pred,
                    variance_threshold=args.variance_threshold,
                    var_weight=1.0,
                    cov_weight=0.02,
                )
                probe_loss = z_pred.sum() * 0.0
                if probe_head is not None:
                    probe_pred = probe_head(z_pred)
                    valid = mask.expand_as(probe_pred) > 0.5
                    probe_loss = F.huber_loss(probe_pred[valid], y[valid], delta=2.0)
                loss = pred_loss + args.vicreg_weight * vic_loss + args.probe_weight * probe_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target_encoder(decay=args.ema_decay)

            epoch_losses.append(float(loss.detach().float().item()))
            epoch_pred.append(spatial_var.detach().cpu().numpy())
            pbar.set_postfix({
                "loss": f"{epoch_losses[-1]:.4f}",
                "pred": f"{float(pred_loss.detach()):.4f}",
                "probe": f"{float(probe_loss.detach()):.4f}",
            })

        pred_vars = np.concatenate(epoch_pred, axis=0)
        mean_var = pred_vars.mean(axis=0)
        collapsed = mean_var < args.variance_threshold
        print(
            f"Epoch {epoch + 1}: train_loss={np.mean(epoch_losses):.5f}, "
            f"min_var={mean_var.min():.5f}, mean_var={mean_var.mean():.5f}, "
            f"active_channels={(~collapsed).sum()}/{args.map_channels}"
        )

        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            val_loss, val_probe_loss, val_combined_loss, diag = evaluate(
                model,
                val_loader,
                device,
                args.variance_threshold,
                probe_head=probe_head,
                probe_weight=args.probe_weight,
            )
            print(
                f"  Val latent loss={val_loss:.5f}, "
                f"probe_loss={val_probe_loss:.5f}, combined={val_combined_loss:.5f}, "
                f"min_channel_variance={diag['min_channel_variance']:.5f}, "
                f"mean_channel_variance={diag['mean_channel_variance']:.5f}, "
                f"collapsed={diag['collapsed_channels']}, active={diag['active_channels']}"
            )

            if diag["collapsed_channels"] > 0:
                collapse_bad_checks += 1
                print(
                    f"  WARNING: {diag['collapsed_channels']} JEPA channels below "
                    f"variance threshold {args.variance_threshold:.4f} "
                    f"({collapse_bad_checks}/{args.collapse_patience})"
                )
            else:
                collapse_bad_checks = 0

            if collapse_bad_checks >= args.collapse_patience:
                raise RuntimeError(
                    "JEPA channel collapse persisted across repeated checks. "
                    "Tune masking, augmentations, or VICReg weight before forecasting."
                )

            if selection_metric == "probe":
                selection_value = val_probe_loss
            elif selection_metric == "combined":
                selection_value = val_combined_loss
            else:
                selection_value = val_loss

            if selection_value < best_val and diag["collapsed_channels"] == 0:
                best_val = selection_value
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    epoch + 1,
                    stats,
                    train_indices,
                    val_indices,
                    vars(args),
                    best_val,
                )
                print(f"  Saved best Met-JEPA checkpoint to {args.checkpoint} ({selection_metric}={best_val:.5f})")

    if not os.path.exists(args.checkpoint):
        print("No non-collapsed best checkpoint was saved; saving final checkpoint for inspection.")
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            args.epochs,
            stats,
            train_indices,
            val_indices,
            vars(args),
            best_val,
        )

    print("Met-JEPA pretraining complete.")


if __name__ == "__main__":
    main()
