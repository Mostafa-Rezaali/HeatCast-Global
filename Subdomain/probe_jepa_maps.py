#!/usr/bin/env python3
"""
Probe frozen JEPA spatial maps with a cheap 1x1 Conv2d decoder.

Run:
    python3 -u Subdomain/probe_jepa_maps.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from met_jepa import (
    IMAGE_SIZE,
    JEPA_MAPS_PATH,
    JEPA_META_PATH,
    JEPA_PROBE_REPORT,
    LEAD_TIME,
    anomaly_field,
    compute_doy_array,
    load_climatology,
    load_shared_arrays,
    split_indices,
)


class JEPAMapProbeDataset(Dataset):
    def __init__(self, maps, shared, clim, indices):
        self.maps = maps
        self.shared = shared
        self.clim = clim
        self.indices = list(indices)
        self.doys = compute_doy_array(np.array(shared["time_values"]))
        self.doy_idx = np.clip(self.doys - 1, 0, 365)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        z = np.array(self.maps[t], dtype=np.float32)
        gt = t + LEAD_TIME
        y = anomaly_field(self.shared, self.clim, gt, int(self.doy_idx[gt]))
        raw = np.array(self.shared["heat_index"][:, :, t], dtype=np.float32)
        mask = (np.isfinite(raw) & (raw != 0.0)).astype(np.float32)
        return torch.from_numpy(z), torch.from_numpy(y).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), t


class ProbeHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.net(x).clamp(-4.0, 4.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe JEPA spatial maps")
    parser.add_argument("--maps_path", type=str, default=JEPA_MAPS_PATH)
    parser.add_argument("--meta_path", type=str, default=JEPA_META_PATH)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_eval_samples", type=int, default=544)
    parser.add_argument("--val_samples", type=int, default=0, help="0 means full validation set")
    parser.add_argument("--gate_margin", type=float, default=0.02)
    parser.add_argument("--max_r2_gap", type=float, default=0.15)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--report", type=str, default=JEPA_PROBE_REPORT)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--title", type=str, default="Met-JEPA Probe Gate")
    return parser.parse_args()


def compute_r2_from_flat(preds, truths):
    p = np.concatenate(preds)
    t = np.concatenate(truths)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


@torch.inference_mode()
def evaluate(model, loader, device, max_samples=0):
    model.eval()
    preds, truths = [], []
    mse_sum = 0.0
    n_pix = 0.0
    count = 0
    for z, y, mask, _ in loader:
        if max_samples and count >= max_samples:
            break
        z = z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(z)
        valid = mask.expand_as(pred) > 0.5
        mse_sum += float(((pred[valid] - y[valid]) ** 2).sum().item())
        n_pix += float(valid.sum().item())
        preds.append(pred[valid].detach().cpu().numpy().astype(np.float32))
        truths.append(y[valid].detach().cpu().numpy().astype(np.float32))
        count += z.shape[0]
    r2 = compute_r2_from_flat(preds, truths)
    rmse = float(np.sqrt(mse_sum / max(n_pix, 1.0)))
    model.train()
    return r2, rmse


@torch.inference_mode()
def climatology_zero_r2(loader, device, max_samples=0):
    preds, truths = [], []
    count = 0
    for _, y, mask, _ in loader:
        if max_samples and count >= max_samples:
            break
        y = y.to(device)
        mask = mask.to(device)
        valid = mask.expand_as(y) > 0.5
        preds.append(torch.zeros_like(y[valid]).cpu().numpy().astype(np.float32))
        truths.append(y[valid].cpu().numpy().astype(np.float32))
        count += y.shape[0]
    return compute_r2_from_flat(preds, truths)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.maps_path):
        raise FileNotFoundError(f"Missing JEPA maps: {args.maps_path}. Run export_jepa_maps.py first.")
    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"Missing JEPA metadata: {args.meta_path}. Run export_jepa_maps.py first.")
    meta = np.load(args.meta_path, allow_pickle=True)
    collapsed = int(meta["collapsed_channels"])
    min_var = float(meta["min_channel_variance"])
    print(f"JEPA map variance: min={min_var:.5f}, collapsed_channels={collapsed}")
    if collapsed > 0 or min_var < args.variance_threshold:
        raise RuntimeError(
            "JEPA maps show serious channel collapse. Re-train JEPA before probing or forecasting."
        )

    maps = np.load(args.maps_path, mmap_mode="r")
    shared = load_shared_arrays(mmap=True)
    clim = load_climatology()
    train_indices, val_indices, _, _ = split_indices(np.array(shared["time_values"]))
    if args.val_samples and args.val_samples < len(val_indices):
        positions = np.linspace(0, len(val_indices) - 1, args.val_samples, dtype=np.int64)
        val_indices = [int(val_indices[i]) for i in positions]

    train_ds = JEPAMapProbeDataset(maps, shared, clim, train_indices)
    val_ds = JEPAMapProbeDataset(maps, shared, clim, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    train_eval_indices = train_indices[: min(args.train_eval_samples, len(train_indices))]
    train_eval_ds = JEPAMapProbeDataset(maps, shared, clim, train_eval_indices)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    in_channels = maps.shape[1]
    h, w = IMAGE_SIZE
    if tuple(maps.shape[-2:]) != (h, w):
        raise ValueError(f"Unexpected JEPA map spatial shape: {maps.shape[-2:]}, expected {(h, w)}")
    model = ProbeHead(in_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Probe input channels: {in_channels}")
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Probe epoch {epoch + 1}/{args.epochs}", mininterval=10.0)
        for z, y, mask, _ in pbar:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(z)
            valid = mask.expand_as(pred) > 0.5
            loss = F.huber_loss(pred[valid], y[valid], delta=2.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pbar.set_postfix({"loss": f"{losses[-1]:.4f}"})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_r2, val_rmse = evaluate(model, val_loader, device)
            print(f"  Epoch {epoch + 1}: train_loss={np.mean(losses):.5f}, val_r2={val_r2:.4f}, val_rmse={val_rmse:.4f}")

    train_r2, train_rmse = evaluate(model, train_eval_loader, device, max_samples=args.train_eval_samples)
    val_r2, val_rmse = evaluate(model, val_loader, device)
    clim_r2 = climatology_zero_r2(val_loader, device)
    r2_gap = train_r2 - val_r2
    gate_pass = (val_r2 >= clim_r2 + args.gate_margin) and (r2_gap < args.max_r2_gap)

    lines = [
        args.title,
        "===================",
        f"Input maps: {args.maps_path}",
        f"Input channels: {in_channels}",
        f"Climatology-only val R2: {clim_r2:.6f}",
        f"Probe train R2:          {train_r2:.6f}",
        f"Probe val R2:            {val_r2:.6f}",
        f"Probe train RMSE:        {train_rmse:.6f}",
        f"Probe val RMSE:          {val_rmse:.6f}",
        f"Train-val R2 gap:        {r2_gap:.6f}",
        f"Gate margin required:    {args.gate_margin:.6f}",
        f"Max R2 gap allowed:      {args.max_r2_gap:.6f}",
        f"Gate:                    {'PASS' if gate_pass else 'FAIL'}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Saved probe report to {args.report}")

    if not gate_pass:
        raise RuntimeError("JEPA probe gate failed. Do not proceed to MeshFlowNet training yet.")


if __name__ == "__main__":
    main()
