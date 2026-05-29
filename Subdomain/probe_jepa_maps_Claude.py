#!/usr/bin/env python3
"""
================================================================================
Probe Met-JEPA Spatial Maps
================================================================================

Fast validation gate before committing to full MeshFlowNet training.
Fits a cheap 1x1 Conv2d decoder from frozen JEPA maps to target anomaly.

Pass criteria:
  1. Probe val R² >= climatology-only R² + 0.02
  2. Train-val R² gap < 0.15
  3. No severe channel collapse (>8 channels dead)

Usage:
    python3 -u Subdomain/probe_jepa_maps.py
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pretrain_met_jepa import (
    JEPAConfig, apply_subdomain_config, load_all_data,
    detect_continuous_runs, build_valid_indices,
    time_to_date, compute_doy_array,
)

CFG = JEPAConfig()
BASE_DATE = datetime(1981, 5, 1)


class ProbeDataset(Dataset):
    """Pairs JEPA maps with target anomalies for probe training."""

    def __init__(self, jepa_maps, valid_indices, indices, data, cfg):
        self.cfg = cfg
        self.data = data
        self.indices = indices

        # Build lookup: timestep -> position in jepa_maps array
        self.t_to_map_idx = {int(t): i for i, t in enumerate(valid_indices)}

        self.jepa_maps = jepa_maps
        self.clim_mean = data["clim_mean"]
        self.clim_std = data["clim_std"]
        self.valid_doys = data["valid_doys"].astype(bool)

        time_values = np.array(data["time_values"])
        doy_values = compute_doy_array(time_values)
        self.doy_indices = np.clip(doy_values.astype(np.int32) - 1, 0, 365)

    def __len__(self):
        return len(self.indices)

    def _to_anomaly(self, time_idx):
        doy_idx = int(self.doy_indices[time_idx])
        raw = torch.from_numpy(
            np.array(self.data["heat_index"][:, :, time_idx], dtype=np.float32)
        )
        clim = torch.from_numpy(self.clim_mean[doy_idx].copy().astype(np.float32))
        cstd = torch.from_numpy(self.clim_std[doy_idx].copy().astype(np.float32))
        valid = torch.isfinite(raw) & (raw != 0.0) & torch.isfinite(clim) & torch.isfinite(cstd)
        anom = torch.zeros_like(raw)
        anom[valid] = (raw[valid] - clim[valid]) / (cstd[valid] + 1e-6)
        return anom.unsqueeze(0)

    def __getitem__(self, idx):
        t = self.indices[idx]
        map_idx = self.t_to_map_idx[t]

        jepa_map = torch.from_numpy(
            np.array(self.jepa_maps[map_idx], dtype=np.float32)
        )
        target = self._to_anomaly(t + self.cfg.LEAD_TIME)

        raw_t = torch.from_numpy(
            np.array(self.data["heat_index"][:, :, t], dtype=np.float32)
        )
        land_mask = (torch.isfinite(raw_t) & (raw_t != 0.0)).float().unsqueeze(0)

        return jepa_map, target, land_mask


class LinearProbe(nn.Module):
    """1x1 Conv decoder: (32, H, W) -> (1, H, W)."""

    def __init__(self, in_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class ShallowProbe(nn.Module):
    """Two-layer 1x1 Conv decoder for slightly more capacity."""

    def __init__(self, in_channels=32, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_r2(preds, truths, mask):
    """Compute R² over land pixels."""
    valid = mask > 0.5
    p = preds[valid].numpy()
    t = truths[valid].numpy()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot


def main():
    print(f"\n{'='*70}")
    print(f"Probe Met-JEPA Spatial Maps")
    print(f"{'='*70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_subdomain_config(CFG)

    # Load JEPA maps
    sub_tag = ""
    if CFG.SUBDOMAIN_ENABLED:
        la0, la1 = CFG.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = CFG.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"

    maps_path = os.path.join(CFG.WORK_DIR, "data_cache", f"jepa_spatial_maps{sub_tag}.npz")
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"JEPA maps not found: {maps_path}. Run export_jepa_maps.py first.")

    maps_data = np.load(maps_path)
    jepa_maps = maps_data["jepa_maps"]
    valid_indices = maps_data["valid_indices"]
    train_indices_arr = maps_data["train_indices"]
    val_indices_arr = maps_data["val_indices"]

    print(f"Loaded JEPA maps: shape={jepa_maps.shape}")
    print(f"  Train indices: {len(train_indices_arr)}")
    print(f"  Val indices:   {len(val_indices_arr)}")

    # Channel collapse check
    channel_vars = maps_data["channel_vars"]
    n_collapsed = (channel_vars < 0.01).sum()
    print(f"  Channel vars: min={channel_vars.min():.6f}, mean={channel_vars.mean():.4f}")
    print(f"  Collapsed channels: {n_collapsed}/32")

    if n_collapsed > 8:
        print(f"\n  FAIL: {n_collapsed} collapsed channels. Retrain JEPA.")
        return

    # Load data for targets
    data = load_all_data(CFG)

    # Datasets
    train_dataset = ProbeDataset(
        jepa_maps, valid_indices, train_indices_arr.tolist(), data, CFG
    )
    val_dataset = ProbeDataset(
        jepa_maps, valid_indices, val_indices_arr.tolist(), data, CFG
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Train both probes
    for probe_name, ProbeClass in [("linear", LinearProbe), ("shallow", ShallowProbe)]:
        print(f"\n  --- {probe_name.upper()} PROBE ---")

        probe = ProbeClass(in_channels=jepa_maps.shape[1]).to(device)
        optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)

        n_params = sum(p.numel() for p in probe.parameters())
        print(f"  Parameters: {n_params}")

        best_val_r2 = -999.0

        for epoch in range(50):
            probe.train()
            for jmap, target, mask in train_loader:
                jmap = jmap.to(device)
                target = target.to(device)
                mask = mask.to(device)

                pred = probe(jmap)
                valid = mask.expand_as(pred) > 0.5
                if valid.any():
                    loss = F.huber_loss(pred[valid], target[valid], delta=2.0)
                else:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate every 10 epochs
            if (epoch + 1) % 10 == 0:
                probe.eval()
                all_preds, all_truths, all_masks = [], [], []
                train_preds, train_truths, train_masks = [], [], []

                with torch.no_grad():
                    for jmap, target, mask in val_loader:
                        pred = probe(jmap.to(device)).cpu()
                        all_preds.append(pred)
                        all_truths.append(target)
                        all_masks.append(mask)

                    for i, (jmap, target, mask) in enumerate(train_loader):
                        if i >= len(val_loader):
                            break
                        pred = probe(jmap.to(device)).cpu()
                        train_preds.append(pred)
                        train_truths.append(target)
                        train_masks.append(mask)

                val_preds = torch.cat(all_preds)
                val_truths = torch.cat(all_truths)
                val_masks = torch.cat(all_masks)
                val_r2 = compute_r2(val_preds, val_truths, val_masks)

                tr_preds = torch.cat(train_preds)
                tr_truths = torch.cat(train_truths)
                tr_masks = torch.cat(train_masks)
                train_r2 = compute_r2(tr_preds, tr_truths, tr_masks)

                r2_gap = train_r2 - val_r2

                print(
                    f"    Epoch {epoch+1:3d}: "
                    f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Gap={r2_gap:.4f}"
                )
                best_val_r2 = max(best_val_r2, val_r2)

        # Climatology baseline (predicting zero anomaly)
        clim_preds = torch.zeros_like(val_truths)
        clim_r2 = compute_r2(clim_preds, val_truths, val_masks)

        print(f"\n  {probe_name.upper()} PROBE RESULTS:")
        print(f"    Best val R²:     {best_val_r2:.4f}")
        print(f"    Climatology R²:  {clim_r2:.4f}")
        print(f"    Margin:          {best_val_r2 - clim_r2:.4f} (need >= 0.02)")
        print(f"    Train-Val gap:   {r2_gap:.4f} (need < 0.15)")

        # Gate check
        passes_r2 = best_val_r2 >= clim_r2 + 0.02
        passes_gap = r2_gap < 0.15
        passes_collapse = n_collapsed <= 8

        gate_pass = passes_r2 and passes_gap and passes_collapse

        print(f"\n  GATE CHECK:")
        print(f"    R² margin >= 0.02:       {'PASS' if passes_r2 else 'FAIL'}")
        print(f"    Train-val gap < 0.15:    {'PASS' if passes_gap else 'FAIL'}")
        print(f"    Channels not collapsed:  {'PASS' if passes_collapse else 'FAIL'}")
        print(f"    Overall:                 {'PASS' if gate_pass else 'FAIL'}")

        if gate_pass:
            print(f"\n  Proceed to MeshFlowNet training with --input_mode jepa_spatial_global")
        else:
            print(f"\n  Do not proceed. Adjust JEPA pretraining first.")

    print(f"\n{'='*70}")
    print(f"Probe Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
