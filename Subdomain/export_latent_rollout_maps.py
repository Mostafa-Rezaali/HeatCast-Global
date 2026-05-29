#!/usr/bin/env python3
"""
Export frozen teacher-forced latent rollout maps.

Only context/global fields at initialization time t are used. The EMA target
encoder and future target fields are not used during export.

Run:
    python3 -u export_latent_rollout_maps.py --device cuda --batch_size 8
"""

import argparse
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from met_jepa import (
    DATA_CACHE,
    IMAGE_SIZE,
    LATENT_ROLLOUT_CHECKPOINT_PATH,
    LATENT_ROLLOUT_MAPS_PATH,
    LATENT_ROLLOUT_META_PATH,
    LatentRolloutDataset,
    ROLLOUT_LEADS,
    channel_variance_diagnostics,
    load_climatology,
    load_latent_rollout_checkpoint,
    load_shared_arrays,
    split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export teacher-forced latent rollout maps")
    parser.add_argument("--checkpoint", type=str, default=LATENT_ROLLOUT_CHECKPOINT_PATH)
    parser.add_argument("--maps_path", type=str, default=LATENT_ROLLOUT_MAPS_PATH)
    parser.add_argument("--meta_path", type=str, default=LATENT_ROLLOUT_META_PATH)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    os.makedirs(DATA_CACHE, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading latent rollout checkpoint: {args.checkpoint}")
    model, ckpt = load_latent_rollout_checkpoint(args.checkpoint, device)
    stats = ckpt["stats"]
    rollout_leads = tuple(int(x) for x in ckpt.get("rollout_leads", ROLLOUT_LEADS))
    print(f"Rollout leads in checkpoint: {rollout_leads}")

    shared = load_shared_arrays(mmap=True)
    clim = load_climatology()
    time_values = np.array(shared["time_values"])
    train_indices, val_indices, test_indices, all_valid = split_indices(time_values)
    n_time = len(time_values)
    h, w = IMAGE_SIZE
    map_channels = int(ckpt.get("map_channels", 32))
    valid_map_mask = np.zeros(n_time, dtype=bool)
    valid_map_mask[np.array(all_valid, dtype=np.int64)] = True

    ds = LatentRolloutDataset(shared, clim, stats, all_valid, leads=rollout_leads, return_targets=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    out_dtype = np.float16 if args.dtype == "float16" else np.float32
    print(f"Writing latent rollout maps to {args.maps_path}")
    print(f"Map shape: ({n_time}, {map_channels}, {h}, {w}), dtype={args.dtype}")
    maps = open_memmap(args.maps_path, mode="w+", dtype=out_dtype, shape=(n_time, map_channels, h, w))
    maps[:] = 0
    maps.flush()

    per_batch_vars = []
    exported = 0
    for context, global_fields, t in tqdm(loader, desc="Exporting latent rollout maps", mininterval=10.0):
        context = context.to(device, non_blocking=True)
        global_fields = global_fields.to(device, non_blocking=True)
        z = model.predict_lead15_maps(context, global_fields).float()
        if not torch.isfinite(z).all():
            raise ValueError("Non-finite values detected in exported latent rollout maps.")
        diag = channel_variance_diagnostics(z, args.variance_threshold)
        per_batch_vars.append(diag["per_channel_variance"])
        maps[t.numpy().astype(np.int64)] = z.cpu().numpy().astype(out_dtype, copy=False)
        exported += len(t)

    maps.flush()
    per_channel_var = np.stack(per_batch_vars, axis=0).mean(axis=0).astype(np.float32)
    collapsed = per_channel_var < args.variance_threshold
    print(
        f"Exported {exported} valid maps. "
        f"min_var={per_channel_var.min():.5f}, mean_var={per_channel_var.mean():.5f}, "
        f"collapsed={collapsed.sum()}, active={(~collapsed).sum()}/{map_channels}"
    )

    np.savez(
        args.meta_path,
        jepa_maps_path=args.maps_path,
        latent_rollout_maps_path=args.maps_path,
        jepa_maps_shape=np.array([n_time, map_channels, h, w], dtype=np.int64),
        jepa_maps_dtype=np.array(args.dtype),
        valid_map_mask=valid_map_mask,
        train_indices=np.array(train_indices, dtype=np.int64),
        val_indices=np.array(val_indices, dtype=np.int64),
        test_indices=np.array(test_indices, dtype=np.int64),
        lead_time=np.array(15, dtype=np.int32),
        rollout_leads=np.array(rollout_leads, dtype=np.int32),
        map_channels=np.array(map_channels, dtype=np.int32),
        variance_threshold=np.array(args.variance_threshold, dtype=np.float32),
        per_channel_variance=per_channel_var,
        min_channel_variance=np.array(float(per_channel_var.min()), dtype=np.float32),
        mean_channel_variance=np.array(float(per_channel_var.mean()), dtype=np.float32),
        collapsed_channels=np.array(int(collapsed.sum()), dtype=np.int32),
        active_channels=np.array(int((~collapsed).sum()), dtype=np.int32),
        target_encoder_used_for_export=np.array(False),
        future_targets_used_for_export=np.array(False),
        context_only_export=np.array(True),
        artifact_type=np.array("teacher_forced_latent_rollout_maps"),
        checkpoint_path=args.checkpoint,
    )
    print(f"Saved latent rollout metadata to {args.meta_path}")


if __name__ == "__main__":
    main()
