#!/usr/bin/env python3
"""Compute publication baselines from stitched hindcast statistics.

The default mode is immediate and CPU-light: climatology, persistence, and
MeshFlowNet summaries from existing hindcast stat/sample-summary files. The
ridge baseline is available behind --run_ridge because it loads the full CONUS
training arrays and can take substantial CPU/RAM.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from publication_analysis_utils import (
    STAT_KEYS,
    climatology_mse_from_stats,
    default_fold_stat_paths,
    ensure_dir,
    land_mean,
    model_persistence_corr_maps,
    mse_from_sums,
    stitch_stats,
    write_latex_table,
)


S2S_REFERENCES = {
    "SubX ensemble mean": {
        "tac": "~0.05",
        "source": "Pegion et al. 2019, BAMS; comparable week-3 CONUS skill varies by model/season.",
    },
    "ECMWF S2S": {
        "tac": "~0.08",
        "source": "Vitart 2017, QJRMS; not an exact same-domain MJJAS T2max TAC.",
    },
    "UFS prototype": {
        "tac": "near zero",
        "source": "NOAA/EMC operational verification; use as qualitative reference only.",
    },
}


def weighted_mean_from_summaries(summary_glob: str, key: str) -> float:
    total = 0.0
    count = 0
    for path in sorted(glob.glob(summary_glob)):
        with np.load(path, allow_pickle=True) as data:
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            total += float(values.sum())
            count += int(values.size)
    return total / count if count else float("nan")


def summary_paths_for_stat_paths(stat_paths: List[str], paper_dir: str) -> List[str]:
    paths = []
    for stat_path in stat_paths:
        basename = os.path.basename(stat_path)
        summary_name = basename.replace("hindcast_tac_stats_", "hindcast_sample_summary_")
        candidate = os.path.join(paper_dir, summary_name)
        if os.path.exists(candidate):
            paths.append(candidate)
    return paths


def weighted_mean_from_summary_paths(summary_paths: List[str], key: str) -> float:
    total = 0.0
    count = 0
    for path in summary_paths:
        with np.load(path, allow_pickle=True) as data:
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            total += float(values.sum())
            count += int(values.size)
    return total / count if count else float("nan")


def stitch_prefixed_stats(stat_paths: List[str], prefixes: Tuple[str, ...]):
    aggregate = None
    mask = None
    prefix_used = None
    for path in stat_paths:
        with np.load(path, allow_pickle=False) as data:
            prefix = next(
                (p for p in prefixes if all(f"{p}{key}" in data for key in STAT_KEYS)),
                None,
            )
            if prefix is None:
                continue
            if prefix_used is None:
                prefix_used = prefix
            elif prefix != prefix_used:
                raise ValueError(
                    f"Mixed weekly statistic types are not supported: {prefix_used!r} and {prefix!r}"
                )
            if aggregate is None:
                aggregate = {
                    key: np.asarray(data[f"{prefix}{key}"], dtype=np.float64).copy()
                    for key in STAT_KEYS
                }
                mask = np.asarray(data["mask"], dtype=np.uint8)
            else:
                for key in STAT_KEYS:
                    aggregate[key] += np.asarray(data[f"{prefix}{key}"], dtype=np.float64)
    return aggregate, mask


def quick_baselines(stat_paths: List[str], paper_dir: str) -> Dict[str, Dict[str, float]]:
    stats, mask, _, _ = stitch_stats(stat_paths)
    model_corr, persist_corr = model_persistence_corr_maps(stats, mask)
    model_tac = land_mean(model_corr, mask)
    persist_tac = land_mean(persist_corr, mask)

    model_mse = mse_from_sums(
        stats["pred_sq_sum"], stats["truth_sq_sum"], stats["pred_truth_sum"], stats["count"], mask
    )
    persist_mse = mse_from_sums(
        stats["persist_sq_sum"], stats["truth_sq_sum"], stats["persist_truth_sum"], stats["count"], mask
    )
    climo_mse = climatology_mse_from_stats(stats, mask)
    weekly_stats, weekly_mask = stitch_prefixed_stats(stat_paths, ("weekly7_", "weekly_truth7_"))
    weekly_model_tac = float("nan")
    weekly_persist_tac = float("nan")
    if weekly_stats is not None and weekly_mask is not None:
        weekly_model_corr, weekly_persist_corr = model_persistence_corr_maps(weekly_stats, weekly_mask)
        weekly_model_tac = land_mean(weekly_model_corr, weekly_mask)
        weekly_persist_tac = land_mean(weekly_persist_corr, weekly_mask)

    summary_paths = summary_paths_for_stat_paths(stat_paths, paper_dir)
    model_crps = weighted_mean_from_summary_paths(summary_paths, "mae")
    persist_crps = weighted_mean_from_summary_paths(summary_paths, "persist_mae")

    return {
        "Climatology": {
            "tac": 0.0,
            "weekly_tac": 0.0,
            "mse": climo_mse,
            "crps": float("nan"),
            "note": "TAC is zero/undefined by construction for anomaly climatology; CRPS needs per-sample abs anomalies.",
        },
        "Persistence": {
            "tac": persist_tac,
            "weekly_tac": weekly_persist_tac,
            "mse": persist_mse,
            "crps": persist_crps,
            "note": "",
        },
        "MeshFlowNet": {
            "tac": model_tac,
            "weekly_tac": weekly_model_tac,
            "mse": model_mse,
            "crps": model_crps,
            "note": "",
        },
    }


def _global_lon_mask(lon_1d: np.ndarray, lon0: float, lon1: float) -> np.ndarray:
    if abs(float(lon1) - float(lon0)) >= 359.0:
        return np.ones_like(lon_1d, dtype=bool)
    lon0 %= 360.0
    lon1 %= 360.0
    if lon0 <= lon1:
        return (lon_1d >= lon0) & (lon_1d <= lon1)
    return (lon_1d >= lon0) | (lon_1d <= lon1)


def global_region_masks(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    h, w = shape
    lat = np.linspace(90.0, -90.0, h)[:, None]
    lon = np.linspace(0.0, 360.0, w, endpoint=False)[None, :]
    boxes = {
        "nino34": (-5.0, 5.0, -170.0, -120.0),
        "iod_west": (-10.0, 10.0, 50.0, 70.0),
        "iod_east": (-10.0, 0.0, 90.0, 110.0),
        "north_pacific": (30.0, 60.0, 150.0, -150.0),
        "north_atlantic": (30.0, 60.0, -80.0, 0.0),
        "tropical_atlantic": (-15.0, 15.0, -60.0, 0.0),
        "southern_ocean": (-60.0, -30.0, 0.0, 360.0),
        "arctic": (60.0, 90.0, 0.0, 360.0),
        "full_tropics": (-30.0, 30.0, 0.0, 360.0),
    }
    masks = {}
    lon_1d = lon.ravel()
    for name, (lat0, lat1, lon0, lon1) in boxes.items():
        lat_mask = (lat >= lat0) & (lat <= lat1)
        lon_mask = _global_lon_mask(lon_1d, lon0, lon1)[None, :]
        masks[name] = lat_mask & lon_mask
    return masks


def run_ridge_baseline(args, output_dir: Path):
    """Optional heavy ridge baseline using persistence PCs and global region means."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("The ridge baseline requires scikit-learn.") from exc

    import cfm_mesh_train as cfm

    ridge_fold_tac = []
    ridge_stats_all = None
    mask_np = None

    for fold in args.folds:
        cfm.Config.CV_TEST_OFFSETS = (int(fold),)
        cfm.Config.CV_VAL_OFFSETS = ((int(fold) + 1) % int(cfm.Config.CV_STRIDE),)
        cfm.apply_run_name(f"cvfold{fold}")
        cfm.apply_extended_global_fields()

        shared = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
        time_values = np.asarray(shared["time_values"])
        runs = cfm.detect_continuous_runs(time_values)
        all_valid = cfm.build_valid_indices(
            runs,
            lead_time=cfm.Config.LEAD_TIME,
            min_history=cfm.required_input_history(cfm.Config),
        )
        train_idx, _, test_idx, *_ = cfm.build_crossval_split(all_valid, time_values)
        stats_path = cfm.get_norm_stats_path(cfm.Config)
        norm_stats = cfm.load_norm_stats_npz(stats_path)
        climo = cfm.load_or_build_train_climatology(
            shared, train_idx, norm_stats, cfm.Config, ddp=False
        )
        mask_t = cfm.load_conus_mask(cfm.Config)
        mask_np = mask_t.numpy().astype(bool)
        land_flat = mask_np.ravel()

        train_idx = np.asarray(train_idx[: args.max_ridge_train or None], dtype=np.int64)
        test_idx = np.asarray(test_idx[: args.max_ridge_test or None], dtype=np.int64)

        hi = np.asarray(shared["heat_index"], dtype=np.float32)
        hi_mean = float(norm_stats["hi_mean"])
        hi_std = float(norm_stats["hi_std"])

        def z_at(t):
            field = hi[:, :, int(t)]
            z = (np.where(np.isfinite(field) & (field != 0.0), field, hi_mean) - hi_mean) / (hi_std + 1e-8)
            return z.astype(np.float32)

        def target_anom_matrix(indices):
            out = np.empty((len(indices), int(land_flat.sum())), dtype=np.float32)
            for i, t in enumerate(indices):
                tt = int(t) + int(cfm.Config.LEAD_TIME)
                doy = int(cfm.compute_doy_array(time_values[[tt]])[0])
                out[i] = (z_at(tt) - np.asarray(climo[doy], dtype=np.float32)).ravel()[land_flat]
            return out

        def persistence_matrix(indices, target_anom=False):
            out = np.empty((len(indices), int(land_flat.sum())), dtype=np.float32)
            for i, t in enumerate(indices):
                z = z_at(int(t))
                if target_anom:
                    tt = int(t) + int(cfm.Config.LEAD_TIME)
                    doy = int(cfm.compute_doy_array(time_values[[tt]])[0])
                    z = z - np.asarray(climo[doy], dtype=np.float32)
                out[i] = z.ravel()[land_flat]
            return out

        print(f"Fold {fold}: building ridge feature matrices...")
        x_persist_train = persistence_matrix(train_idx, target_anom=False)
        x_persist_test = persistence_matrix(test_idx, target_anom=False)
        pca_persist = PCA(
            n_components=min(args.persistence_pcs, x_persist_train.shape[0] - 1),
            svd_solver="randomized",
            random_state=args.random_state,
        )
        persist_train_pc = pca_persist.fit_transform(x_persist_train)
        persist_test_pc = pca_persist.transform(x_persist_test)
        del x_persist_train, x_persist_test

        region_masks = None
        global_items = list(shared.get("global_data", {}).items())
        if global_items:
            sample_field = np.asarray(global_items[0][1][:, :, 0])
            region_masks = global_region_masks(sample_field.shape)

        doys = cfm.compute_doy_array(time_values)

        def global_features(indices):
            feats = []
            for t in indices:
                row = []
                if region_masks:
                    for _, arr in global_items:
                        field = np.asarray(arr[:, :, int(t)], dtype=np.float32)
                        for rm in region_masks.values():
                            row.append(float(np.nanmean(field[rm])))
                doy = float(doys[int(t)])
                row.extend([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
                feats.append(row)
            return np.asarray(feats, dtype=np.float32)

        x_train = np.column_stack([persist_train_pc, global_features(train_idx)])
        x_test = np.column_stack([persist_test_pc, global_features(test_idx)])

        y_train = target_anom_matrix(train_idx)
        pca_target = PCA(
            n_components=min(args.target_pcs, y_train.shape[0] - 1),
            svd_solver="randomized",
            random_state=args.random_state,
        )
        y_train_pc = pca_target.fit_transform(y_train)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        ridge.fit(x_train, y_train_pc)

        pred_anom = pca_target.inverse_transform(ridge.predict(x_test)).astype(np.float32)
        truth_anom = target_anom_matrix(test_idx)
        persist_anom = persistence_matrix(test_idx, target_anom=True)

        h, w = mask_np.shape
        fold_stats = {key: np.zeros((h, w), dtype=np.float64) for key in STAT_KEYS}
        for i in range(len(test_idx)):
            valid_f = land_flat.astype(np.float64).reshape(h, w)
            pred_full = np.zeros(h * w, dtype=np.float32)
            truth_full = np.zeros(h * w, dtype=np.float32)
            persist_full = np.zeros(h * w, dtype=np.float32)
            pred_full[land_flat] = pred_anom[i]
            truth_full[land_flat] = truth_anom[i]
            persist_full[land_flat] = persist_anom[i]
            pred_full = pred_full.reshape(h, w).astype(np.float64)
            truth_full = truth_full.reshape(h, w).astype(np.float64)
            persist_full = persist_full.reshape(h, w).astype(np.float64)
            fold_stats["pred_sum"] += pred_full
            fold_stats["truth_sum"] += truth_full
            fold_stats["pred_sq_sum"] += pred_full * pred_full
            fold_stats["truth_sq_sum"] += truth_full * truth_full
            fold_stats["pred_truth_sum"] += pred_full * truth_full
            fold_stats["persist_sum"] += persist_full
            fold_stats["persist_sq_sum"] += persist_full * persist_full
            fold_stats["persist_truth_sum"] += persist_full * truth_full
            fold_stats["count"] += valid_f

        ridge_map, _ = model_persistence_corr_maps(fold_stats, mask_np.astype(np.uint8))
        ridge_fold_tac.append(land_mean(ridge_map, mask_np))
        if ridge_stats_all is None:
            ridge_stats_all = fold_stats
        else:
            for key in STAT_KEYS:
                ridge_stats_all[key] += fold_stats[key]

    assert ridge_stats_all is not None and mask_np is not None
    ridge_map, _ = model_persistence_corr_maps(ridge_stats_all, mask_np.astype(np.uint8))
    ridge_tac = land_mean(ridge_map, mask_np)
    ridge_mse = mse_from_sums(
        ridge_stats_all["pred_sq_sum"],
        ridge_stats_all["truth_sq_sum"],
        ridge_stats_all["pred_truth_sum"],
        ridge_stats_all["count"],
        mask_np,
    )
    np.savez_compressed(
        output_dir / "ridge_baseline_results.npz",
        ridge_tac_stitched=np.array(ridge_tac, dtype=np.float32),
        ridge_tac_per_fold=np.array(ridge_fold_tac, dtype=np.float32),
        ridge_mse_stitched=np.array(ridge_mse, dtype=np.float32),
        ridge_corr_map=ridge_map.astype(np.float32),
        mask=mask_np.astype(np.uint8),
        **ridge_stats_all,
    )
    return ridge_tac, ridge_mse, ridge_map, np.asarray(ridge_fold_tac, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Fold hindcast stat files or globs.")
    parser.add_argument("--paper_dir", default="hindcast_paper_data")
    parser.add_argument("--output_dir", default="paper_figures")
    parser.add_argument("--run_ridge", action="store_true", help="Run the heavy target-PCA ridge baseline.")
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--persistence_pcs", type=int, default=50)
    parser.add_argument("--target_pcs", type=int, default=100)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--max_ridge_train", type=int, default=None)
    parser.add_argument("--max_ridge_test", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    paths = args.files or default_fold_stat_paths(".")
    out = ensure_dir(args.output_dir)
    results = quick_baselines(paths, args.paper_dir)

    ridge_tac = ridge_mse = float("nan")
    ridge_fold = np.full(5, np.nan, dtype=np.float32)
    ridge_map = None
    if args.run_ridge:
        ridge_tac, ridge_mse, ridge_map, ridge_fold = run_ridge_baseline(args, out)
        results["Ridge Regression"] = {
            "tac": float(ridge_tac),
            "mse": float(ridge_mse),
            "crps": float("nan"),
            "note": "Target-PCA multivariate ridge; CRPS requires absolute-error export.",
        }
    else:
        results["Ridge Regression"] = {
            "tac": float("nan"),
            "mse": float("nan"),
            "crps": float("nan"),
            "note": "Skipped. Re-run with --run_ridge.",
        }

    stats, mask, years, n_samples = stitch_stats(paths)
    model_corr, persist_corr = model_persistence_corr_maps(stats, mask)

    np.savez_compressed(
        out / "baselines_results.npz",
        climo_tac_stitched=np.array(results["Climatology"]["tac"], dtype=np.float32),
        climo_weekly7_tac_stitched=np.array(results["Climatology"]["weekly_tac"], dtype=np.float32),
        climo_mse_stitched=np.array(results["Climatology"]["mse"], dtype=np.float32),
        persistence_tac_stitched=np.array(results["Persistence"]["tac"], dtype=np.float32),
        persistence_weekly7_tac_stitched=np.array(results["Persistence"]["weekly_tac"], dtype=np.float32),
        persistence_mse_stitched=np.array(results["Persistence"]["mse"], dtype=np.float32),
        persistence_crps_stitched=np.array(results["Persistence"]["crps"], dtype=np.float32),
        meshflownet_tac_stitched=np.array(results["MeshFlowNet"]["tac"], dtype=np.float32),
        meshflownet_weekly7_tac_stitched=np.array(results["MeshFlowNet"]["weekly_tac"], dtype=np.float32),
        meshflownet_mse_stitched=np.array(results["MeshFlowNet"]["mse"], dtype=np.float32),
        meshflownet_crps_stitched=np.array(results["MeshFlowNet"]["crps"], dtype=np.float32),
        ridge_tac_stitched=np.array(ridge_tac, dtype=np.float32),
        ridge_tac_per_fold=ridge_fold,
        ridge_mse_stitched=np.array(ridge_mse, dtype=np.float32),
        meshflownet_tac_map=model_corr.astype(np.float32),
        persistence_tac_map=persist_corr.astype(np.float32),
        ridge_tac_map=(ridge_map.astype(np.float32) if ridge_map is not None else np.full(mask.shape, np.nan, dtype=np.float32)),
        mask=mask.astype(np.uint8),
        years=np.array(years, dtype=np.int16),
        n_samples=np.array(n_samples, dtype=np.int32),
    )

    table_rows = []
    for name in ["Climatology", "Persistence", "Ridge Regression", "MeshFlowNet"]:
        row = results[name]
        table_rows.append([
            name,
            f"{row['tac']:.3f}" if np.isfinite(row["tac"]) else "-",
            f"{row.get('weekly_tac', float('nan')):.3f}" if np.isfinite(row.get("weekly_tac", float("nan"))) else "-",
            f"{row['mse']:.3f}" if np.isfinite(row["mse"]) else "-",
            f"{row['crps']:.3f}" if np.isfinite(row["crps"]) else "-",
        ])
    for name, ref in S2S_REFERENCES.items():
        table_rows.append([name, ref["tac"], ref["tac"], "-", "-"])
    write_latex_table(out / "baselines_table.tex", ["Model", "Daily TAC", "Weekly7 TAC", "MSE", "CRPS"], table_rows)

    with open(out / "baselines_notes.txt", "w", encoding="utf-8") as f:
        f.write("Baseline notes\n")
        f.write("==============\n")
        for name, row in results.items():
            if row.get("note"):
                f.write(f"{name}: {row['note']}\n")
        f.write("\nOperational S2S references are approximate literature context, not exact same-domain verification.\n")
        for name, ref in S2S_REFERENCES.items():
            f.write(f"{name}: {ref['tac']} ({ref['source']})\n")

    print("Baseline summary")
    print("================")
    for row in table_rows:
        print(
            f"{row[0]:22s} DailyTAC={row[1]:>7s}  Weekly7TAC={row[2]:>7s}  "
            f"MSE={row[3]:>7s}  CRPS={row[4]:>7s}"
        )
    print(f"Saved baseline outputs to: {out}")


if __name__ == "__main__":
    main()
