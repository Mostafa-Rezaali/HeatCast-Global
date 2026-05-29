#!/usr/bin/env python3
"""Re-run hindcast inference and export TAC sufficient statistics by target year."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from publication_analysis_utils import STAT_KEYS, ensure_dir, land_mean, model_persistence_corr_maps


def target_datetime(time_values, time_index):
    return datetime(1981, 5, 1) + timedelta(days=float(time_values[int(time_index)]))


def checkpoint_for_fold(pattern: str, fold: int, run_name: str) -> str:
    return pattern.format(fold=fold, run_name=run_name)


def empty_stats_like(cfm, h, w):
    return cfm._empty_tac_stats(h, w)


def accumulate_weekly7_by_year(records, time_values, tac_climo, mask_np, cfm, half_window=3):
    h, w = mask_np.shape
    by_year = {}
    by_time = {int(rec["target_time_idx"]): rec for rec in records}
    if not by_time:
        return by_year

    for center_time_idx in sorted(by_time):
        window_time_indices = list(
            range(center_time_idx - int(half_window), center_time_idx + int(half_window) + 1)
        )
        window_records = [by_time.get(t) for t in window_time_indices]
        if any(rec is None for rec in window_records):
            continue
        window_values = np.asarray(time_values[window_time_indices], dtype=np.float64)
        if len(window_values) != 2 * int(half_window) + 1:
            continue
        if not np.allclose(np.diff(window_values), 1.0, atol=1e-6):
            continue

        pred_mean = np.mean([rec["pred"] for rec in window_records], axis=0, dtype=np.float32)
        truth_mean = np.mean([rec["truth"] for rec in window_records], axis=0, dtype=np.float32)
        persist_mean = np.mean([rec["persist"] for rec in window_records], axis=0, dtype=np.float32)
        window_doys = [
            int((datetime(1981, 5, 1) + timedelta(days=float(v))).timetuple().tm_yday)
            for v in window_values
        ]
        climo_mean = np.mean(
            [np.asarray(tac_climo[doy], dtype=np.float32) for doy in window_doys],
            axis=0,
            dtype=np.float32,
        )

        center_year = target_datetime(time_values, center_time_idx).year
        if center_year not in by_year:
            by_year[center_year] = empty_stats_like(cfm, h, w)
        cfm._accumulate_tac_stats_with_climo(
            by_year[center_year], pred_mean, truth_mean, persist_mean, climo_mean, mask_np
        )
    return by_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--run_name_pattern",
        default="cvfold{fold}",
        help="Run name pattern with {fold}, for example cvfold{fold}_pub_weekly7.",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        default="trained_cfm_direct15_{run_name}_best_monitor.pth",
        help="Checkpoint pattern with {fold} and/or {run_name}.",
    )
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--output", default="hindcast_stats/per_year_tac_stats.npz")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multi_lead_tube", action="store_true")
    parser.add_argument("--prediction_leads", default=None)
    args = parser.parse_args()

    import torch
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x

    import cfm_mesh_train as cfm
    if args.multi_lead_tube:
        cfm.Config.MULTI_LEAD_TUBE = True
    if args.prediction_leads is not None:
        cfm.Config.PREDICTION_LEADS = cfm.parse_int_tuple(args.prediction_leads)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    per_year = {}
    weekly7_per_year = {}
    mask_np = None

    for fold in args.folds:
        cfm.Config.CV_TEST_OFFSETS = (int(fold),)
        cfm.Config.CV_VAL_OFFSETS = ((int(fold) + 1) % int(cfm.Config.CV_STRIDE),)
        run_name = args.run_name_pattern.format(fold=fold)
        cfm.apply_run_name(run_name)
        cfm.apply_extended_global_fields()

        print(f"\n=== Fold {fold} ({cfm.cv_split_tag(cfm.Config)}) ===", flush=True)
        shared = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
        time_values = np.asarray(shared["time_values"])
        runs = cfm.detect_continuous_runs(time_values)
        all_valid = cfm.build_valid_indices(
            runs,
            lead_time=cfm.max_prediction_lead(cfm.Config),
            min_history=cfm.required_input_history(cfm.Config),
        )
        train_idx, val_idx, test_idx, *_ = cfm.build_crossval_split(all_valid, time_values)
        split_idx = test_idx if args.split == "test" else val_idx

        stats_path = cfm.get_norm_stats_path(cfm.Config)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing normalization stats for fold {fold}: {stats_path}")
        norm_stats = cfm.load_norm_stats_npz(stats_path)
        tac_climo = cfm.load_or_build_train_climatology(
            shared, train_idx, norm_stats, cfm.Config, ddp=False
        )

        mask = cfm.load_conus_mask(cfm.Config)
        mask_np = mask.numpy().astype(np.uint8)
        mask_dev = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        mesh = cfm.build_mesh_once(cfm.Config, mask.to(device), device, ddp=False)
        ckpt = checkpoint_for_fold(args.checkpoint_pattern, fold, run_name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt}")
        model = cfm._load_meshflownet_checkpoint(ckpt, mesh, device)

        dataset = cfm.ClimateDataset(
            cfm.Config,
            mode=args.split,
            val_indices=split_idx if args.split == "val" else None,
            test_indices=split_idx if args.split == "test" else None,
            normalization_stats=norm_stats,
            shared_data=shared,
            target_climatology=tac_climo,
        )
        total = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
        iterator = tqdm(range(total), desc=f"Fold {fold} {args.split}")
        h, w = cfm.Config.IMAGE_SIZE
        tube_mode = bool(cfm.Config.MULTI_LEAD_TUBE)
        tube_leads = cfm.prediction_leads(cfm.Config)
        tube_center_idx = cfm.center_lead_index(cfm.Config)
        weekly_records = [] if not tube_mode else None

        for dataset_idx in iterator:
            y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = dataset[dataset_idx]
            pred = cfm.predict_direct(
                model,
                x_t.unsqueeze(0).to(device),
                x_tm1.unsqueeze(0).to(device),
                x_tm2.unsqueeze(0).to(device),
                spatial_c.unsqueeze(0).to(device),
                vec_c.unsqueeze(0).to(device),
                global_fields.unsqueeze(0).to(device),
                batch_mask.unsqueeze(0).to(device),
                device,
            )
            target_time_idx = int(t_idx) + int(cfm.Config.LEAD_TIME)
            target_doy = int(dataset.doy_values[target_time_idx])
            year = target_datetime(dataset.time_values, target_time_idx).year

            if tube_mode:
                pred_tube_np = pred[0, :, :h, :w].detach().cpu().numpy()
                truth_tube_np = y[:, :h, :w].detach().cpu().numpy()
                pred_np = pred_tube_np[tube_center_idx]
                truth_np = truth_tube_np[tube_center_idx]
            else:
                pred_tube_np = None
                truth_tube_np = None
                pred_np = pred[0, 0, :h, :w].detach().cpu().numpy()
                truth_np = y[0, :h, :w].detach().cpu().numpy()
            if cfm.Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
                climo_np = np.asarray(tac_climo[target_doy], dtype=np.float32)
                if tube_mode:
                    for lead_pos, lead in enumerate(tube_leads):
                        lead_doy = int(dataset.doy_values[int(t_idx) + int(lead)])
                        lead_climo = np.asarray(tac_climo[lead_doy], dtype=np.float32)
                        pred_tube_np[lead_pos] = pred_tube_np[lead_pos] + lead_climo
                        truth_tube_np[lead_pos] = truth_tube_np[lead_pos] + lead_climo
                    pred_np = pred_tube_np[tube_center_idx]
                    truth_np = truth_tube_np[tube_center_idx]
                else:
                    pred_np = pred_np + climo_np
                    truth_np = truth_np + climo_np
            persist_np = x_t[0, :h, :w].detach().cpu().numpy()

            if year not in per_year:
                per_year[year] = cfm._empty_tac_stats(h, w)
            cfm._accumulate_tac_stats(
                per_year[year], pred_np, truth_np, persist_np, target_doy, tac_climo, mask_np
            )
            if tube_mode:
                if year not in weekly7_per_year:
                    weekly7_per_year[year] = cfm._empty_tac_stats(h, w)
                lead_doys = [
                    int(dataset.doy_values[int(t_idx) + int(lead)])
                    for lead in tube_leads
                ]
                climo_mean = np.mean(
                    [np.asarray(tac_climo[doy], dtype=np.float32) for doy in lead_doys],
                    axis=0,
                    dtype=np.float32,
                )
                cfm._accumulate_tac_stats_with_climo(
                    weekly7_per_year[year],
                    pred_tube_np.mean(axis=0, dtype=np.float32),
                    truth_tube_np.mean(axis=0, dtype=np.float32),
                    persist_np,
                    climo_mean,
                    mask_np,
                )
            else:
                weekly_records.append({
                    "target_time_idx": int(target_time_idx),
                    "pred": pred_np.astype(np.float32, copy=True),
                    "truth": truth_np.astype(np.float32, copy=True),
                    "persist": persist_np.astype(np.float32, copy=True),
                })

        if not tube_mode:
            fold_weekly = accumulate_weekly7_by_year(
                weekly_records, np.asarray(dataset.time_values), tac_climo, mask_np, cfm
            )
            for year, stats in fold_weekly.items():
                if year not in weekly7_per_year:
                    weekly7_per_year[year] = stats
                else:
                    for key in STAT_KEYS:
                        weekly7_per_year[year][key] += stats[key]

    if mask_np is None:
        raise RuntimeError("No samples were exported.")

    years = sorted(per_year)
    payload = {
        "years": np.array(years, dtype=np.int16),
        "mask": mask_np.astype(np.uint8),
        "multi_lead_tube": np.array(int(cfm.Config.MULTI_LEAD_TUBE), dtype=np.int8),
        "prediction_leads": np.array(cfm.prediction_leads(cfm.Config), dtype=np.int16),
    }
    for key in STAT_KEYS:
        payload[f"{key}_by_year"] = np.stack([per_year[y][key] for y in years]).astype(np.float64)
    empty_weekly_stats = cfm._empty_tac_stats(mask_np.shape[0], mask_np.shape[1])
    for key in STAT_KEYS:
        payload[f"weekly7_{key}_by_year"] = np.stack([
            (weekly7_per_year[y] if y in weekly7_per_year else empty_weekly_stats)[key]
            for y in years
        ]).astype(np.float64)

    out = Path(args.output)
    ensure_dir(out.parent)
    np.savez_compressed(out, **payload)

    print("\nPer-year TAC summary")
    print("====================")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)})")
    aggregate = {key: np.sum(payload[f"{key}_by_year"], axis=0) for key in STAT_KEYS}
    model_map, persist_map = model_persistence_corr_maps(aggregate, mask_np)
    print(f"Model TAC:       {land_mean(model_map, mask_np):.4f}")
    print(f"Persistence TAC: {land_mean(persist_map, mask_np):.4f}")
    weekly_aggregate = {
        key: np.sum(payload[f"weekly7_{key}_by_year"], axis=0) for key in STAT_KEYS
    }
    weekly_model_map, weekly_persist_map = model_persistence_corr_maps(weekly_aggregate, mask_np)
    print(f"Weekly7 model TAC:       {land_mean(weekly_model_map, mask_np):.4f}")
    print(f"Weekly7 persistence TAC: {land_mean(weekly_persist_map, mask_np):.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
