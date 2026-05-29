#!/usr/bin/env python3
"""Lightweight fold-2 audit for MeshFlowNet hindcast outputs.

This script is intentionally login-node friendly: it does not import torch,
open NetCDF files, build datasets, or run inference. It only reads compact
NPZ/CSV diagnostics that already exist after export.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


STAT_KEYS: Tuple[str, ...] = (
    "pred_sum",
    "truth_sum",
    "pred_sq_sum",
    "truth_sq_sum",
    "pred_truth_sum",
    "persist_sum",
    "persist_sq_sum",
    "persist_truth_sum",
    "count",
)

EXPECTED_FOLD2_TEST_YEARS = [1983, 1988, 1993, 1998, 2003, 2008, 2013, 2018, 2023]


def expand(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def fold_id(path: str) -> int:
    match = re.search(r"cvfold(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def scalar(data: Mapping[str, np.ndarray], key: str, default: float = float("nan")) -> float:
    if key not in data:
        return default
    try:
        return float(np.asarray(data[key]).item())
    except Exception:
        return default


def text_scalar(data: Mapping[str, np.ndarray], key: str) -> str:
    if key not in data:
        return ""
    arr = np.asarray(data[key])
    try:
        return str(arr.item())
    except Exception:
        return str(arr)


def first_scalar(data: Mapping[str, np.ndarray], keys: Sequence[str], default: float = float("nan")) -> float:
    for key in keys:
        if key in data:
            return scalar(data, key, default)
    return default


def land_mean(field: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask > 0.5) & np.isfinite(field)
    return float(np.nanmean(field[valid])) if np.any(valid) else float("nan")


def corr_from_sums(
    x_sum: np.ndarray,
    y_sum: np.ndarray,
    x_sq_sum: np.ndarray,
    y_sq_sum: np.ndarray,
    xy_sum: np.ndarray,
    count: np.ndarray,
    mask: np.ndarray,
    min_count: int = 2,
) -> np.ndarray:
    safe_count = np.maximum(count.astype(np.float64), 1.0)
    cov = xy_sum - (x_sum * y_sum) / safe_count
    x_var = x_sq_sum - (x_sum * x_sum) / safe_count
    y_var = y_sq_sum - (y_sum * y_sum) / safe_count
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid = (mask > 0.5) & (count >= min_count) & (denom > 1e-12)
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr


def mse_map_from_sums(
    x_sq_sum: np.ndarray,
    y_sq_sum: np.ndarray,
    xy_sum: np.ndarray,
    count: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    sqerr = x_sq_sum + y_sq_sum - 2.0 * xy_sum
    out = np.full(count.shape, np.nan, dtype=np.float64)
    valid = (mask > 0.5) & (count > 0)
    out[valid] = sqerr[valid] / np.maximum(count[valid], 1.0)
    return out


def temporal_variance(sum_: np.ndarray, sq_sum: np.ndarray, count: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(count.shape, np.nan, dtype=np.float64)
    valid = (mask > 0.5) & (count > 1)
    mean = sum_[valid] / count[valid]
    out[valid] = np.maximum(sq_sum[valid] / count[valid] - mean * mean, 0.0)
    return out


def require_stats(data: Mapping[str, np.ndarray], path: str) -> None:
    missing = [key for key in STAT_KEYS + ("mask",) if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")


def stats_summary(path: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        require_stats(data, path)
        stats = {key: np.asarray(data[key], dtype=np.float64) for key in STAT_KEYS}
        mask = np.asarray(data["mask"], dtype=np.uint8)
        years = np.atleast_1d(data["years"]).astype(int).tolist() if "years" in data else []
        n_samples = int(scalar(data, "n_samples", np.nan))
        run_name = text_scalar(data, "run_name")
        cv_split = text_scalar(data, "cv_split")
        stored_model_tac = scalar(data, "model_tac")
        stored_persistence_tac = scalar(data, "persistence_tac")
        weekly_model = first_scalar(data, ("weekly7_model_tac", "weekly_truth7_model_tac"))
        weekly_persistence = first_scalar(
            data, ("weekly7_persistence_tac", "weekly_truth7_persistence_tac")
        )
        weekly_n = int(first_scalar(data, ("weekly7_n_samples", "weekly_truth7_n_samples"), 0.0))
        weekly_label = "True weekly7" if "weekly7_model_tac" in data else "Legacy truth7"

    model_corr = corr_from_sums(
        stats["pred_sum"],
        stats["truth_sum"],
        stats["pred_sq_sum"],
        stats["truth_sq_sum"],
        stats["pred_truth_sum"],
        stats["count"],
        mask,
    )
    persistence_corr = corr_from_sums(
        stats["persist_sum"],
        stats["truth_sum"],
        stats["persist_sq_sum"],
        stats["truth_sq_sum"],
        stats["persist_truth_sum"],
        stats["count"],
        mask,
    )
    skill_map = model_corr - persistence_corr
    model_mse_map = mse_map_from_sums(
        stats["pred_sq_sum"], stats["truth_sq_sum"], stats["pred_truth_sum"], stats["count"], mask
    )
    persistence_mse_map = mse_map_from_sums(
        stats["persist_sq_sum"], stats["truth_sq_sum"], stats["persist_truth_sum"], stats["count"], mask
    )

    pred_var = temporal_variance(stats["pred_sum"], stats["pred_sq_sum"], stats["count"], mask)
    truth_var = temporal_variance(stats["truth_sum"], stats["truth_sq_sum"], stats["count"], mask)
    persist_var = temporal_variance(stats["persist_sum"], stats["persist_sq_sum"], stats["count"], mask)
    valid = (mask > 0.5) & np.isfinite(skill_map)

    count_land = stats["count"][mask > 0.5]
    count_min = float(np.nanmin(count_land)) if count_land.size else float("nan")
    count_max = float(np.nanmax(count_land)) if count_land.size else float("nan")
    count_mean = float(np.nanmean(count_land)) if count_land.size else float("nan")

    return {
        "path": path,
        "fold": fold_id(path),
        "run_name": run_name,
        "cv_split": cv_split,
        "years": years,
        "n_samples": n_samples,
        "shape": tuple(mask.shape),
        "land_pixels": int(np.sum(mask > 0.5)),
        "model_tac": land_mean(model_corr, mask),
        "persistence_tac": land_mean(persistence_corr, mask),
        "stored_model_tac": stored_model_tac,
        "stored_persistence_tac": stored_persistence_tac,
        "skill": land_mean(skill_map, mask),
        "model_mse": land_mean(model_mse_map, mask),
        "persistence_mse": land_mean(persistence_mse_map, mask),
        "mse_skill": land_mean(persistence_mse_map - model_mse_map, mask),
        "frac_model_beats_persistence": float(np.nanmean(skill_map[valid] > 0.0)) if np.any(valid) else float("nan"),
        "skill_p05": float(np.nanpercentile(skill_map[valid], 5)) if np.any(valid) else float("nan"),
        "skill_p50": float(np.nanpercentile(skill_map[valid], 50)) if np.any(valid) else float("nan"),
        "skill_p95": float(np.nanpercentile(skill_map[valid], 95)) if np.any(valid) else float("nan"),
        "count_min": count_min,
        "count_mean": count_mean,
        "count_max": count_max,
        "truth_var": land_mean(truth_var, mask),
        "pred_var": land_mean(pred_var, mask),
        "persist_var": land_mean(persist_var, mask),
        "pred_truth_var_ratio": land_mean(pred_var / np.maximum(truth_var, 1e-12), mask),
        "persist_truth_var_ratio": land_mean(persist_var / np.maximum(truth_var, 1e-12), mask),
        "weekly_model_tac": weekly_model,
        "weekly_persistence_tac": weekly_persistence,
        "weekly_n": weekly_n,
        "weekly_label": weekly_label,
    }


def print_stat_summary(summary: Mapping[str, object]) -> List[str]:
    flags: List[str] = []
    print("Fold-2 hindcast stats")
    print("====================")
    print(f"File:        {summary['path']}")
    print(f"Run:         {summary['run_name'] or '-'}")
    print(f"CV split:    {summary['cv_split'] or '-'}")
    print(f"Years:       {summary['years']}")
    print(f"Samples:     {summary['n_samples']}")
    print(f"Shape/land:  {summary['shape']} / {summary['land_pixels']}")
    print()
    print(f"Model TAC:       {summary['model_tac']:.4f}")
    print(f"Persistence TAC: {summary['persistence_tac']:.4f}")
    print(f"TAC skill:       {summary['skill']:+.4f}")
    print(f"Model MSE:       {summary['model_mse']:.4f}")
    print(f"Persistence MSE: {summary['persistence_mse']:.4f}")
    print(f"MSE improvement: {summary['mse_skill']:+.4f}")
    print(f"Pixels model > persistence TAC: {100.0 * summary['frac_model_beats_persistence']:.1f}%")
    print(
        "Skill map percentiles: "
        f"p05={summary['skill_p05']:+.3f}, "
        f"p50={summary['skill_p50']:+.3f}, "
        f"p95={summary['skill_p95']:+.3f}"
    )
    print(
        "Temporal variance ratio: "
        f"pred/truth={summary['pred_truth_var_ratio']:.3f}, "
        f"persist/truth={summary['persist_truth_var_ratio']:.3f}"
    )
    if int(summary["weekly_n"]) > 0:
        print(
            f"{summary['weekly_label']} diagnostic: "
            f"n={summary['weekly_n']}, model={summary['weekly_model_tac']:.4f}, "
            f"persist={summary['weekly_persistence_tac']:.4f}, "
            f"skill={summary['weekly_model_tac'] - summary['weekly_persistence_tac']:+.4f}"
        )
    print(f"Land-pixel sample counts: min={summary['count_min']:.0f}, mean={summary['count_mean']:.1f}, max={summary['count_max']:.0f}")
    print()

    years = [int(y) for y in summary["years"]]
    if years and years != EXPECTED_FOLD2_TEST_YEARS:
        flags.append(f"Unexpected fold-2 test years: {years}; expected {EXPECTED_FOLD2_TEST_YEARS}.")
    cv_split = str(summary["cv_split"])
    if cv_split and ("val3" not in cv_split or "test2" not in cv_split):
        flags.append(f"Fold-2 file has unusual cv_split='{cv_split}' (expected cv5_val3_test2).")
    if abs(float(summary["stored_model_tac"]) - float(summary["model_tac"])) > 5e-4:
        flags.append("Stored model_tac does not match recomputed TAC from sums.")
    if abs(float(summary["stored_persistence_tac"]) - float(summary["persistence_tac"])) > 5e-4:
        flags.append("Stored persistence_tac does not match recomputed TAC from sums.")
    if np.isfinite(summary["count_min"]) and np.isfinite(summary["n_samples"]):
        if float(summary["count_min"]) < 0.95 * float(summary["n_samples"]):
            flags.append("Some land pixels have much lower sample counts than n_samples; check NaNs/mask/counting.")
    if float(summary["mse_skill"]) > 0.0 and float(summary["skill"]) < 0.0:
        flags.append("Model beats persistence on MSE but loses on TAC; this points to phase/correlation, not a basic export failure.")
    return flags


def print_fold_comparison(paths: Sequence[str], target_path: str) -> List[str]:
    flags: List[str] = []
    rows = [stats_summary(path) for path in paths]
    if len(rows) <= 1:
        return flags

    print("Cross-fold comparison")
    print("=====================")
    print(f"{'Fold':>4s} {'Run':<22s} {'Years':<11s} {'N':>5s} {'TAC':>7s} {'Pers':>7s} {'Skill':>8s} {'MSE':>7s} {'PersMSE':>8s}")
    print("-" * 92)
    for row in sorted(rows, key=lambda r: (int(r["fold"]), str(r["run_name"]))):
        years = row["years"]
        year_text = f"{min(years)}-{max(years)}" if years else "-"
        print(
            f"{row['fold']:4d} {str(row['run_name'])[:22]:<22s} {year_text:<11s} {row['n_samples']:5d} "
            f"{row['model_tac']:7.4f} {row['persistence_tac']:7.4f} {row['skill']:+8.4f} "
            f"{row['model_mse']:7.3f} {row['persistence_mse']:8.3f}"
        )
    print()

    target_abs = os.path.abspath(target_path)
    target = next((row for row in rows if os.path.abspath(str(row["path"])) == target_abs), None)
    if target is None:
        return flags
    skills = np.array([float(row["skill"]) for row in rows], dtype=np.float64)
    persist = np.array([float(row["persistence_tac"]) for row in rows], dtype=np.float64)
    if len(rows) >= 3 and np.nanstd(skills) > 1e-8:
        z_skill = (float(target["skill"]) - float(np.nanmean(skills))) / float(np.nanstd(skills))
        print(f"Target fold skill z-score vs these files: {z_skill:+.2f}")
        if z_skill < -1.25:
            flags.append("Fold 2 is a low-skill outlier relative to the supplied fold files.")
    if len(rows) >= 3 and np.nanstd(persist) > 1e-8:
        z_persist = (float(target["persistence_tac"]) - float(np.nanmean(persist))) / float(np.nanstd(persist))
        print(f"Target fold persistence-TAC z-score:      {z_persist:+.2f}")
        if z_persist > 1.25:
            flags.append("Fold 2 has unusually high persistence TAC; beating persistence is harder for this held-out year set.")
    print()
    return flags


def monthly_summaries(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, object]] = []
    with np.load(path, allow_pickle=False) as data:
        if "months" not in data or "mask" not in data:
            return []
        months = np.asarray(data["months"], dtype=np.int16)
        mask = np.asarray(data["mask"], dtype=np.uint8)
        for i, month in enumerate(months):
            stats = {}
            missing = False
            for key in STAT_KEYS:
                full_key = f"{key}_by_month"
                if full_key not in data:
                    missing = True
                    break
                stats[key] = np.asarray(data[full_key][i], dtype=np.float64)
            if missing:
                continue
            temp_path = f"{path}::month{int(month)}"
            row = stats_summary_from_arrays(temp_path, stats, mask)
            row["month"] = int(month)
            rows.append(row)
    return rows


def stats_summary_from_arrays(path: str, stats: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, object]:
    model_corr = corr_from_sums(
        stats["pred_sum"], stats["truth_sum"], stats["pred_sq_sum"], stats["truth_sq_sum"],
        stats["pred_truth_sum"], stats["count"], mask,
    )
    persistence_corr = corr_from_sums(
        stats["persist_sum"], stats["truth_sum"], stats["persist_sq_sum"], stats["truth_sq_sum"],
        stats["persist_truth_sum"], stats["count"], mask,
    )
    model_mse_map = mse_map_from_sums(
        stats["pred_sq_sum"], stats["truth_sq_sum"], stats["pred_truth_sum"], stats["count"], mask
    )
    persistence_mse_map = mse_map_from_sums(
        stats["persist_sq_sum"], stats["truth_sq_sum"], stats["persist_truth_sum"], stats["count"], mask
    )
    count_land = stats["count"][mask > 0.5]
    return {
        "path": path,
        "n_mean": float(np.nanmean(count_land)) if count_land.size else float("nan"),
        "model_tac": land_mean(model_corr, mask),
        "persistence_tac": land_mean(persistence_corr, mask),
        "skill": land_mean(model_corr - persistence_corr, mask),
        "model_mse": land_mean(model_mse_map, mask),
        "persistence_mse": land_mean(persistence_mse_map, mask),
    }


def print_monthly(path: str) -> List[str]:
    rows = monthly_summaries(path)
    flags: List[str] = []
    if not rows:
        print("Monthly fold-2 audit")
        print("====================")
        print(f"No monthly file found/readable: {path or '-'}")
        print()
        return flags
    print("Monthly fold-2 audit")
    print("====================")
    names = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    print(f"{'Month':>5s} {'NpixMean':>9s} {'TAC':>7s} {'Pers':>7s} {'Skill':>8s} {'MSE':>7s} {'PersMSE':>8s}")
    print("-" * 68)
    for row in rows:
        print(
            f"{names.get(row['month'], str(row['month'])):>5s} {row['n_mean']:9.1f} "
            f"{row['model_tac']:7.4f} {row['persistence_tac']:7.4f} {row['skill']:+8.4f} "
            f"{row['model_mse']:7.3f} {row['persistence_mse']:8.3f}"
        )
    worst = min(rows, key=lambda row: float(row["skill"]))
    best = max(rows, key=lambda row: float(row["skill"]))
    print()
    print(f"Worst month: {names.get(worst['month'], worst['month'])} ({worst['skill']:+.4f})")
    print(f"Best month:  {names.get(best['month'], best['month'])} ({best['skill']:+.4f})")
    if float(worst["skill"]) < -0.03:
        flags.append(f"Fold 2 has a strongly negative monthly TAC skill in {names.get(worst['month'], worst['month'])}.")
    print()
    return flags


def arr_to_list(data: Mapping[str, np.ndarray], key: str) -> List[object]:
    if key not in data:
        return []
    return np.asarray(data[key]).tolist()


def print_sample_summary(path: str) -> List[str]:
    flags: List[str] = []
    print("Sample-summary audit")
    print("====================")
    if not path or not os.path.exists(path):
        print(f"No sample summary found: {path or '-'}")
        print()
        return flags
    with np.load(path, allow_pickle=True) as data:
        years = [int(x) for x in arr_to_list(data, "year")]
        months = [int(x) for x in arr_to_list(data, "month")]
        keys = set(data.files)
        numeric = {key: np.asarray(data[key], dtype=np.float64) for key in keys if key not in {"init_date", "target_date"} and np.asarray(data[key]).ndim == 1}

    print(f"Rows: {len(years) if years else 'unknown'}")
    if years:
        print(f"Year counts:  {dict(sorted(Counter(years).items()))}")
    if months:
        name = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
        month_counts = {name.get(k, str(k)): v for k, v in sorted(Counter(months).items())}
        print(f"Month counts: {month_counts}")

    def mean_key(key: str) -> float:
        vals = numeric.get(key)
        if vals is None:
            return float("nan")
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else float("nan")

    if {"mse", "persist_mse"} <= keys:
        print(f"Mean sample MSE: model={mean_key('mse'):.3f}, persistence={mean_key('persist_mse'):.3f}")
    if {"mae", "persist_mae"} <= keys:
        print(f"Mean sample MAE: model={mean_key('mae'):.3f}, persistence={mean_key('persist_mae'):.3f}")
    if {"pred_mean", "truth_mean", "persist_mean"} <= keys:
        pred_bias = mean_key("pred_mean") - mean_key("truth_mean")
        persist_bias = mean_key("persist_mean") - mean_key("truth_mean")
        print(f"Mean CONUS bias: model={pred_bias:+.3f}, persistence={persist_bias:+.3f}")
    if "heat_area_frac" in keys:
        heat = numeric["heat_area_frac"]
        strong = int(np.sum(heat > 0.10))
        print(f"Samples with heat_area_frac > 0.10: {strong}/{len(heat)}")
        if strong == 0:
            flags.append("Sample summary contains no strong heat-area cases; paper-map selection may miss extremes.")
    print()
    return flags


def safe_float(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def print_training_metrics(paths: Sequence[str], test_persistence_tac: float) -> List[str]:
    flags: List[str] = []
    print("Training-curve audit")
    print("====================")
    if not paths:
        print("No training metric CSVs found.")
        print()
        return flags

    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        val_tac = np.array([safe_float(r, "val_tac") for r in rows], dtype=np.float64)
        val_r2 = np.array([safe_float(r, "val_r2") for r in rows], dtype=np.float64)
        spatial = np.array([safe_float(r, "spatial_anom_r2") for r in rows], dtype=np.float64)
        val_persistence = np.array([safe_float(r, "persistence_tac") for r in rows], dtype=np.float64)
        epochs = np.array([safe_float(r, "epoch") for r in rows], dtype=np.float64)
        best_tac_i = int(np.nanargmax(val_tac)) if np.any(np.isfinite(val_tac)) else -1
        best_r2_i = int(np.nanargmax(val_r2)) if np.any(np.isfinite(val_r2)) else -1
        best_spatial_i = int(np.nanargmax(spatial)) if np.any(np.isfinite(spatial)) else -1
        print(f"File: {path}")
        if best_tac_i >= 0:
            print(
                f"  Best val TAC: epoch {epochs[best_tac_i]:.0f}, "
                f"TAC={val_tac[best_tac_i]:.4f}, val persistence={val_persistence[best_tac_i]:.4f}"
            )
        if best_r2_i >= 0:
            print(f"  Best val R2:  epoch {epochs[best_r2_i]:.0f}, R2={val_r2[best_r2_i]:.4f}, TAC={val_tac[best_r2_i]:.4f}")
        if best_spatial_i >= 0:
            print(f"  Best spatial anomaly R2: epoch {epochs[best_spatial_i]:.0f}, spatial={spatial[best_spatial_i]:.4f}, TAC={val_tac[best_spatial_i]:.4f}")
        print(f"  Final row: epoch {epochs[-1]:.0f}, TAC={val_tac[-1]:.4f}, R2={val_r2[-1]:.4f}")
        if np.any(np.isfinite(val_persistence)) and np.isfinite(test_persistence_tac):
            val_pers_mean = float(np.nanmean(val_persistence))
            gap = test_persistence_tac - val_pers_mean
            print(f"  Test persistence TAC minus mean val persistence TAC: {gap:+.4f}")
            if gap > 0.04:
                flags.append(
                    "Fold-2 test years are much more persistence-friendly than validation years; "
                    "TAC-based checkpoint selection may not transfer cleanly."
                )
        if best_tac_i >= 0 and len(rows) > 4 and best_tac_i == len(rows) - 1:
            flags.append("Training ended while val TAC was still peaking; fold 2 may need more epochs/patience.")
        print()
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stat",
        default="hindcast_stats/hindcast_tac_stats_cvfold2_test.npz",
        help="Fold-2 hindcast stats NPZ. Use the run-specific file for diagnostics, e.g. cvfold2_acloss010.",
    )
    parser.add_argument(
        "--all-stats",
        nargs="*",
        default=["hindcast_stats/hindcast_tac_stats_cvfold[0-4]_test.npz"],
        help="Files/globs used to compare fold 2 against other folds.",
    )
    parser.add_argument(
        "--monthly",
        default="hindcast_paper_data/hindcast_monthly_stats_cvfold2_test.npz",
        help="Fold-2 monthly stats NPZ.",
    )
    parser.add_argument(
        "--summary",
        default="hindcast_paper_data/hindcast_sample_summary_cvfold2_test.npz",
        help="Fold-2 sample-summary NPZ.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["training_metrics/fold_epoch_metrics_cvfold2*.csv"],
        help="Fold-2 training metric CSVs/globs.",
    )
    args = parser.parse_args()

    stat_paths = expand([args.stat])
    if not stat_paths:
        raise FileNotFoundError(f"Could not find --stat file: {args.stat}")
    stat_path = stat_paths[0]
    flags: List[str] = []

    summary = stats_summary(stat_path)
    flags.extend(print_stat_summary(summary))

    all_stat_paths = expand(args.all_stats)
    if stat_path not in all_stat_paths:
        all_stat_paths.append(stat_path)
    flags.extend(print_fold_comparison(all_stat_paths, stat_path))
    flags.extend(print_monthly(args.monthly))
    flags.extend(print_sample_summary(args.summary))
    flags.extend(print_training_metrics(expand(args.metrics), float(summary["persistence_tac"])))

    print("Fold-2 checks")
    print("=============")
    if flags:
        for item in dict.fromkeys(flags):
            print(f"[CHECK] {item}")
    else:
        print("No obvious file/stat consistency issue found.")

    print()
    print("Useful interpretation")
    print("=====================")
    print(
        "If the consistency checks pass but fold 2 has positive MSE improvement and negative TAC skill, "
        "the likely issue is not an upside-down map, stale split, or corrupt NPZ. It is probably that "
        "the fold-2 test years are unusually persistence-friendly, while the model still smooths or "
        "mis-times the temporal anomaly signal."
    )


if __name__ == "__main__":
    main()
