#!/usr/bin/env python3
"""
================================================================================
Export Met-JEPA Spatial Maps
================================================================================

Loads the frozen context encoder + predictor from pretrain_met_jepa.py
and generates (T, 32, H, W) spatial maps for all valid timesteps.

CRITICAL: Only uses the context encoder and predictor.
          The target encoder is NOT used during export.
          Maps contain only information available at initialization time.

Usage:
    python3 -u Subdomain/export_jepa_maps.py
================================================================================
"""

import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Import from pretrain script
from pretrain_met_jepa import (
    JEPAConfig, JEPAEncoder, JEPAPredictor,
    apply_subdomain_config, load_all_data,
    detect_continuous_runs, build_valid_indices,
    time_to_date, compute_doy_array, compute_toa_insolation,
    JEPADataset, check_collapse,
)

CFG = JEPAConfig()


def main():
    print(f"\n{'='*70}")
    print(f"Export Met-JEPA Spatial Maps")
    print(f"{'='*70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_subdomain_config(CFG)

    # Load best checkpoint
    best_path = os.path.join(CFG.OUTPUT_DIR, "best_jepa.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No trained JEPA found at {best_path}. Run pretrain_met_jepa.py first.")

    ckpt = torch.load(best_path, map_location=device)
    model_cfg = ckpt["config"]
    print(f"Loaded JEPA checkpoint from epoch {ckpt['epoch']}")
    print(f"  Val loss: {ckpt['val_loss']:.4f}")
    print(f"  Collapse info: {ckpt['collapse_info']}")

    # Fail fast on collapsed model
    if ckpt["collapse_info"]["collapsed_channels"] > 8:
        raise ValueError(
            f"JEPA model has {ckpt['collapse_info']['collapsed_channels']}/32 collapsed channels. "
            f"Retrain with adjusted VICReg weights before exporting."
        )

    # Build context encoder + predictor (NOT target encoder)
    context_encoder = JEPAEncoder(
        in_channels=model_cfg["CONTEXT_CHANNELS"],
        out_channels=model_cfg["JEPA_MAP_CHANNELS"],
        base_dim=model_cfg["ENCODER_BASE_DIM"],
        num_blocks=model_cfg["ENCODER_NUM_BLOCKS"],
    ).to(device)
    context_encoder.load_state_dict(ckpt["context_encoder"])
    context_encoder.eval()

    predictor = JEPAPredictor(
        embed_dim=model_cfg["JEPA_MAP_CHANNELS"],
        hidden_dim=model_cfg.get("PREDICTOR_DIM", 64),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    print(f"\n  Loaded context encoder + predictor (target encoder NOT loaded)")

    # Load data
    data = load_all_data(CFG)

    # Load norm stats
    stats_path = os.path.join(CFG.OUTPUT_DIR, "jepa_norm_stats.npz")
    s = np.load(stats_path)
    norm_stats = {}
    for k in s.files:
        v = s[k]
        if v.ndim == 0:
            norm_stats[k] = float(v)
        else:
            norm_stats[k] = torch.from_numpy(v)
    print(f"  Loaded norm stats from {stats_path}")

    # Build indices for ALL valid timesteps (train + val + test)
    time_values = np.array(data["time_values"])
    runs = detect_continuous_runs(time_values)
    all_valid = build_valid_indices(runs, lead_time=CFG.LEAD_TIME, min_history=2)

    time_years = np.array([time_to_date(tv).year for tv in time_values])
    train_indices = [i for i in all_valid if time_years[i] in set(CFG.TRAIN_YEARS)]
    val_indices = [i for i in all_valid if time_years[i] in set(CFG.VAL_YEARS)]
    test_indices = [i for i in all_valid if time_years[i] in set(CFG.TEST_YEARS)]

    print(f"\n  Generating maps for:")
    print(f"    Train: {len(train_indices)} timesteps")
    print(f"    Val:   {len(val_indices)} timesteps")
    print(f"    Test:  {len(test_indices)} timesteps")
    print(f"    Total: {len(all_valid)} timesteps")

    # Create dataset for all indices (no masking during export)
    full_dataset = JEPADataset(CFG, data, all_valid, norm_stats=norm_stats)

    # Allocate output array
    H, W = CFG.IMAGE_SIZE
    C = model_cfg["JEPA_MAP_CHANNELS"]
    n_total = len(all_valid)
    jepa_maps = np.zeros((n_total, C, H, W), dtype=np.float32)

    # Generate maps
    print(f"\n  Generating {n_total} JEPA maps...")
    batch_size = 16

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_contexts = []
            for i in range(start, end):
                ctx, _, _ = full_dataset[i]
                batch_contexts.append(ctx)

            batch = torch.stack(batch_contexts).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ctx_emb = context_encoder(batch)
                pred_emb = predictor(ctx_emb)

            jepa_maps[start:end] = pred_emb.float().cpu().numpy()

            if (start // batch_size) % 50 == 0 or end == n_total:
                print(f"    {end}/{n_total} maps generated")

    # Validate maps
    print(f"\n  Validating exported maps...")
    finite_check = np.all(np.isfinite(jepa_maps))
    print(f"    All finite: {finite_check}")
    if not finite_check:
        n_nan = np.isnan(jepa_maps).sum()
        n_inf = np.isinf(jepa_maps).sum()
        raise ValueError(f"Maps contain {n_nan} NaN and {n_inf} Inf values!")

    # Per-channel variance check
    channel_vars = np.var(jepa_maps, axis=(0, 2, 3))
    n_collapsed = (channel_vars < 0.01).sum()
    print(f"    Channel variances: min={channel_vars.min():.6f}, max={channel_vars.max():.4f}")
    print(f"    Collapsed channels: {n_collapsed}/32")

    if n_collapsed > 8:
        print(f"    WARNING: {n_collapsed} channels collapsed in exported maps!")

    # Build index mapping: dataset_index -> valid_timestep_index
    index_to_timestep = {i: all_valid[i] for i in range(n_total)}

    # Save
    sub_tag = ""
    if CFG.SUBDOMAIN_ENABLED:
        la0, la1 = CFG.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = CFG.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"

    output_path = os.path.join(
        CFG.WORK_DIR, "data_cache",
        f"jepa_spatial_maps{sub_tag}.npz",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        jepa_maps=jepa_maps,
        valid_indices=np.array(all_valid, dtype=np.int64),
        train_indices=np.array(train_indices, dtype=np.int64),
        val_indices=np.array(val_indices, dtype=np.int64),
        test_indices=np.array(test_indices, dtype=np.int64),
        channel_vars=channel_vars,
        jepa_checkpoint_epoch=ckpt["epoch"],
        jepa_val_loss=ckpt["val_loss"],
        target_encoder_used=False,
        image_size=np.array(CFG.IMAGE_SIZE),
        jepa_map_channels=CFG.JEPA_MAP_CHANNELS,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved JEPA maps to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Shape: {jepa_maps.shape}")
    print(f"\n  Next step: python3 -u probe_jepa_maps.py")


if __name__ == "__main__":
    main()
