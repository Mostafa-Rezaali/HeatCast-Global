#!/usr/bin/env python3
"""Fold-safe daily heat-exceedance evaluation for existing HeatCast checkpoints.

Stage 1 only: no architecture, training-loop, or loss changes. This script
derives daily exceedance probabilities from the existing point forecast mean
using month-specific train-year q95 thresholds and climatological anomaly
spread, then evaluates probabilistic exceedance skill.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import cfm_mesh_train as cfm
from publication_analysis_utils import NOAA_REGION_BOXES, conus_lat_lon, ensure_dir, region_masks


MJJAS_MONTHS = (5, 6, 7, 8, 9)
BASE_DATE = datetime(1981, 5, 1)


def time_datetimes(time_values: Sequence[float]) -> List[datetime]:
    return [BASE_DATE + timedelta(days=float(v)) for v in time_values]


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def configure_from_args(args: argparse.Namespace) -> None:
    cfm.Config.DETERMINISTIC = True
    cfm.Config.RUN_NAME = args.run_name
    cfm.Config.CV_STRIDE = int(args.cv_stride)
    fold = int(args.cv_fold) % int(args.cv_stride)
    cfm.Config.CV_FOLD = fold
    cfm.Config.CV_TEST_OFFSETS = (fold,)
    cfm.Config.CV_VAL_OFFSETS = ((fold + 1) % int(args.cv_stride),)

    if args.multi_lead_tube or "tube" in args.run_name.lower():
        cfm.Config.MULTI_LEAD_TUBE = True
        cfm.Config.PREDICTION_LEADS = parse_int_list(args.prediction_leads)
    else:
        cfm.Config.MULTI_LEAD_TUBE = False

    cfm.Config.GRADIENT_LOSS_WEIGHT = 0.0
    cfm.apply_run_name(args.run_name)
    cfm.apply_extended_global_fields()
    cfm.set_random_seed(int(args.seed))


def checkpoint_path_for(run_name: str, checkpoint: str) -> str:
    if checkpoint not in {"best_monitor", "best_tac", "best_r2", "auto"}:
        return checkpoint

    candidates = []
    if checkpoint in {"best_monitor", "auto"}:
        candidates.append(cfm.Config.MONITOR_MODEL_SAVE_PATH)
    if checkpoint in {"best_tac", "auto"}:
        candidates.append(cfm.Config.TAC_MODEL_SAVE_PATH)
    if checkpoint in {"best_r2", "auto"}:
        candidates.append(cfm.Config.MODEL_SAVE_PATH)

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not resolve checkpoint={checkpoint!r} for run_name={run_name}. Tried: {candidates}"
    )


def split_indices_for_config(shared_data: Mapping[str, np.ndarray]):
    time_values = np.asarray(shared_data["time_values"])
    runs = cfm.detect_continuous_runs(time_values)
    all_valid = cfm.build_valid_indices(
        runs,
        lead_time=cfm.max_prediction_lead(cfm.Config),
        min_history=cfm.required_input_history(cfm.Config),
    )
    return cfm.build_crossval_split(all_valid, time_values)


def load_norm_stats() -> Dict[str, torch.Tensor]:
    stats_path = cfm.get_norm_stats_path(cfm.Config)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing norm stats for this fold/config: {stats_path}. "
            "Run training setup first; Stage 1 evaluation does not create training stats."
        )
    return cfm.load_norm_stats_npz(stats_path)


def train_year_mask_from_years(time_values: Sequence[float], train_years: Iterable[int]) -> np.ndarray:
    train_years = set(int(y) for y in train_years)
    dts = time_datetimes(time_values)
    return np.array([dt.year in train_years for dt in dts], dtype=bool)


def month_doy_year_arrays(time_values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dts = time_datetimes(time_values)
    months = np.array([dt.month for dt in dts], dtype=np.int16)
    years = np.array([dt.year for dt in dts], dtype=np.int16)
    doys = cfm.compute_doy_array(np.asarray(time_values)).astype(np.int16)
    return months, years, doys


def _source_mtime(paths: Sequence[str]) -> float:
    existing = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return max(existing) if existing else 0.0


def exceedance_cache_path() -> str:
    suffix = (
        "_tube" + "-".join(str(x) for x in cfm.prediction_leads(cfm.Config))
        if cfm.Config.MULTI_LEAD_TUBE else ""
    )
    fold = int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
    return os.path.join(
        cfm.Config.OUTPUT_DIR,
        "data_cache",
        f"month_q95_sigma_direct15{suffix}_{cfm.cv_split_tag(cfm.Config)}_fold{fold}.npz",
    )


def _normalize_field(field: np.ndarray, norm_stats: Mapping[str, torch.Tensor]) -> np.ndarray:
    hi_mean = float(norm_stats["hi_mean"])
    hi_std = float(norm_stats["hi_std"])
    out = np.asarray(field, dtype=np.float32).copy()
    valid = np.isfinite(out) & (out != 0.0)
    out[valid] = (out[valid] - hi_mean) / (hi_std + 1e-8)
    out[~valid] = np.nan
    return out


def build_month_q95(shared_data, train_year_mask, norm_stats) -> np.ndarray:
    """Month-specific pixelwise train-year q95 in normalized z units."""
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, _ = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    q95_z = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        if indices.size == 0:
            raise RuntimeError(f"No train-year days found for month={month}")
        fields = np.asarray(heat[:, :, indices], dtype=np.float32)
        valid = np.isfinite(fields) & (fields != 0.0)
        fields = (fields - float(norm_stats["hi_mean"])) / (float(norm_stats["hi_std"]) + 1e-8)
        fields[~valid] = np.nan
        with np.errstate(all="ignore"):
            q95_z[month] = np.nanpercentile(fields, 95.0, axis=2).astype(np.float32)
        q95_z[month][~land_mask] = np.nan
        del fields, valid
    return q95_z


def build_month_sigma_clim(shared_data, climo_by_doy, train_year_mask, norm_stats) -> np.ndarray:
    """Month-specific std of train-year daily anomalies about train-year daily climatology."""
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, doys = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    sigma = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        anomalies = []
        for ti in indices:
            field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
            anomalies.append(field_z - np.asarray(climo_by_doy[int(doys[ti])], dtype=np.float32))
        stack = np.stack(anomalies, axis=2)
        with np.errstate(all="ignore"):
            sigma_m = np.nanstd(stack, axis=2).astype(np.float32)
        sigma_m = np.where(np.isfinite(sigma_m), sigma_m, np.nan)
        sigma_m[land_mask] = np.maximum(sigma_m[land_mask], 0.1)
        sigma_m[~land_mask] = np.nan
        sigma[month] = sigma_m
        del stack, anomalies
    return sigma


def build_month_base_rate(shared_data, train_year_mask, norm_stats, q95_z) -> np.ndarray:
    heat = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months, _, _ = month_doy_year_arrays(time_values)
    h, w = cfm.Config.IMAGE_SIZE
    base_rate = np.full((12, h, w), np.nan, dtype=np.float32)
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)

    for month in MJJAS_MONTHS:
        indices = np.where(train_year_mask & (months == month))[0]
        count = np.zeros((h, w), dtype=np.float32)
        exceed = np.zeros((h, w), dtype=np.float32)
        threshold = q95_z[month]
        for ti in indices:
            field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
            valid = land_mask & np.isfinite(field_z) & np.isfinite(threshold)
            count[valid] += 1.0
            exceed[valid] += (field_z[valid] > threshold[valid]).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = exceed / np.maximum(count, 1.0)
        rate[~land_mask] = np.nan
        base_rate[month] = rate.astype(np.float32)
    return base_rate


def load_or_build_exceedance_stats(shared_data, train_years, norm_stats, climo_by_doy):
    path = exceedance_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    time_values = np.asarray(shared_data["time_values"])
    train_mask = train_year_mask_from_years(time_values, train_years)
    norm_path = cfm.get_norm_stats_path(cfm.Config)
    climo_path = cfm.get_climatology_cache_path(cfm.Config)
    src_mtime = _source_mtime([norm_path, climo_path, cfm.Config.TRAINING_DATA_PATH])

    def _cache_ok() -> bool:
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) < src_mtime:
            return False
        try:
            with np.load(path, allow_pickle=True) as data:
                cached_years = set(np.atleast_1d(data["train_years"]).astype(int).tolist())
                cached_fold = int(data["cv_fold"]) if "cv_fold" in data.files else None
                return (
                    cached_years == set(int(y) for y in train_years)
                    and cached_fold == int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0]))
                    and str(data["cv_split"].item()) == cfm.cv_split_tag(cfm.Config)
                    and abs(float(data["hi_mean"]) - float(norm_stats["hi_mean"])) < 1e-6
                    and abs(float(data["hi_std"]) - float(norm_stats["hi_std"])) < 1e-6
                )
        except Exception:
            return False

    if _cache_ok():
        print(f"Loading fold-safe exceedance stats from {path}")
        with np.load(path, allow_pickle=False) as data:
            return (
                np.asarray(data["q95_z"], dtype=np.float32),
                np.asarray(data["sigma_clim"], dtype=np.float32),
                np.asarray(data["base_rate"], dtype=np.float32),
            )

    print("Building fold-safe month-specific q95/base-rate/sigma stats from train years only...")
    q95_z = build_month_q95(shared_data, train_mask, norm_stats)
    sigma_clim = build_month_sigma_clim(shared_data, climo_by_doy, train_mask, norm_stats)
    base_rate = build_month_base_rate(shared_data, train_mask, norm_stats, q95_z)

    for month in MJJAS_MONTHS:
        rate = np.nanmean(base_rate[month])
        print(f"  Train build check month={month}: mean exceedance rate={rate:.4f}")
        if not (0.025 <= rate <= 0.075):
            raise RuntimeError(f"Train exceedance rate for month {month} is not near 5%: {rate:.4f}")

    np.savez_compressed(
        path,
        q95_z=q95_z.astype(np.float32),
        sigma_clim=sigma_clim.astype(np.float32),
        base_rate=base_rate.astype(np.float32),
        train_years=np.array(sorted(int(y) for y in train_years), dtype=np.int16),
        cv_fold=np.array(int(getattr(cfm.Config, "CV_FOLD", cfm.Config.CV_TEST_OFFSETS[0])), dtype=np.int16),
        cv_split=np.array(cfm.cv_split_tag(cfm.Config), dtype=object),
        hi_mean=np.array(float(norm_stats["hi_mean"]), dtype=np.float32),
        hi_std=np.array(float(norm_stats["hi_std"]), dtype=np.float32),
        months=np.array(MJJAS_MONTHS, dtype=np.int8),
        norm_stats_mtime=np.array(os.path.getmtime(norm_path) if os.path.exists(norm_path) else 0.0),
        climo_mtime=np.array(os.path.getmtime(climo_path) if os.path.exists(climo_path) else 0.0),
    )
    print(f"  Saved exceedance stats to {path}")
    return q95_z, sigma_clim, base_rate


def normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(x).astype(np.float32)
    except Exception:
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            return (0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))).numpy().astype(np.float32)


@dataclass
class ReliabilityStats:
    bins: int = 10

    def __post_init__(self):
        self.count = np.zeros(self.bins, dtype=np.float64)
        self.pred_sum = np.zeros(self.bins, dtype=np.float64)
        self.obs_sum = np.zeros(self.bins, dtype=np.float64)

    def update(self, prob: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        valid = mask & np.isfinite(prob) & np.isfinite(truth)
        if not np.any(valid):
            return
        p = np.clip(prob[valid].astype(np.float64), 0.0, 1.0)
        y = truth[valid].astype(np.float64)
        idx = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.count += np.bincount(idx, minlength=self.bins)
        self.pred_sum += np.bincount(idx, weights=p, minlength=self.bins)
        self.obs_sum += np.bincount(idx, weights=y, minlength=self.bins)

    def table(self) -> List[Dict[str, float]]:
        rows = []
        for i in range(self.bins):
            n = self.count[i]
            rows.append({
                "bin": i,
                "lo": i / self.bins,
                "hi": (i + 1) / self.bins,
                "count": n,
                "mean_pred": self.pred_sum[i] / n if n > 0 else float("nan"),
                "obs_freq": self.obs_sum[i] / n if n > 0 else float("nan"),
            })
        return rows

    def slope_ece(self) -> Tuple[float, float]:
        rows = self.table()
        x = np.array([r["mean_pred"] for r in rows], dtype=np.float64)
        y = np.array([r["obs_freq"] for r in rows], dtype=np.float64)
        w = np.array([r["count"] for r in rows], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & (w > 0)
        if valid.sum() < 2:
            slope = float("nan")
        else:
            x0 = np.average(x[valid], weights=w[valid])
            y0 = np.average(y[valid], weights=w[valid])
            denom = np.sum(w[valid] * (x[valid] - x0) ** 2)
            slope = float(np.sum(w[valid] * (x[valid] - x0) * (y[valid] - y0)) / denom) if denom > 0 else float("nan")
        total = np.sum(w[valid])
        ece = float(np.sum(w[valid] * np.abs(x[valid] - y[valid])) / total) if total > 0 else float("nan")
        return slope, ece


@dataclass
class MetricAccumulator:
    name: str
    hist_bins: int = 101

    def __post_init__(self):
        self.brier_sum = 0.0
        self.count = 0.0
        self.truth_pos = 0.0
        self.rel = ReliabilityStats()
        self.hist_pos = np.zeros(self.hist_bins, dtype=np.float64)
        self.hist_neg = np.zeros(self.hist_bins, dtype=np.float64)

    def update(self, prob: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        valid = mask & np.isfinite(prob) & np.isfinite(truth)
        if not np.any(valid):
            return
        p = np.clip(prob[valid].astype(np.float64), 0.0, 1.0)
        y = truth[valid].astype(np.float64)
        self.brier_sum += float(np.sum((p - y) ** 2))
        self.count += float(p.size)
        self.truth_pos += float(np.sum(y))
        self.rel.update(prob, truth, mask)
        idx = np.minimum((p * (self.hist_bins - 1)).round().astype(np.int64), self.hist_bins - 1)
        self.hist_pos += np.bincount(idx, weights=y, minlength=self.hist_bins)
        self.hist_neg += np.bincount(idx, weights=1.0 - y, minlength=self.hist_bins)

    def brier(self) -> float:
        return self.brier_sum / self.count if self.count > 0 else float("nan")

    def aucs(self) -> Tuple[float, float]:
        pos_total = float(self.hist_pos.sum())
        neg_total = float(self.hist_neg.sum())
        if pos_total <= 0 or neg_total <= 0:
            return float("nan"), float("nan")
        tp = np.cumsum(self.hist_pos[::-1])
        fp = np.cumsum(self.hist_neg[::-1])
        tpr = np.r_[0.0, tp / pos_total, 1.0]
        fpr = np.r_[0.0, fp / neg_total, 1.0]
        roc_auc = float(np.trapz(tpr, fpr))
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / pos_total
        pr_auc = float(np.trapz(np.r_[precision[0], precision], np.r_[0.0, recall]))
        return roc_auc, pr_auc

    def threshold_scores(self, thresholds: Sequence[float]) -> Dict[str, float]:
        out = {}
        pos_total = float(self.hist_pos.sum())
        for th in thresholds:
            idx = int(np.clip(round(th * (self.hist_bins - 1)), 0, self.hist_bins - 1))
            hits = float(self.hist_pos[idx:].sum())
            false_alarms = float(self.hist_neg[idx:].sum())
            misses = max(pos_total - hits, 0.0)
            out[f"hit_rate_{th:g}"] = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
            out[f"false_alarm_ratio_{th:g}"] = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
        return out


class EvaluationAccumulator:
    def __init__(self, model_names: Sequence[str], region_masks_map: Mapping[str, np.ndarray]):
        self.metrics = {name: MetricAccumulator(name) for name in model_names}
        self.monthly = defaultdict(lambda: {name: MetricAccumulator(name) for name in model_names})
        self.region_abs = {
            name: {region: 0.0 for region in region_masks_map}
            for name in model_names
        }
        self.region_count = {
            name: {region: 0 for region in region_masks_map}
            for name in model_names
        }
        self.region_masks = region_masks_map

    def update(self, name: str, prob: np.ndarray, truth: np.ndarray, mask: np.ndarray, month: int) -> None:
        self.metrics[name].update(prob, truth, mask)
        self.monthly[int(month)][name].update(prob, truth, mask)
        for region, rmask in self.region_masks.items():
            valid = mask & rmask & np.isfinite(prob) & np.isfinite(truth)
            if not np.any(valid):
                continue
            pred_count = float(np.sum(prob[valid]))
            obs_count = float(np.sum(truth[valid]))
            self.region_abs[name][region] += abs(pred_count - obs_count)
            self.region_count[name][region] += 1

    def summary_rows(self, reference_name: str) -> List[Dict[str, float]]:
        ref = self.metrics[reference_name].brier()
        rows = []
        for name, acc in self.metrics.items():
            roc_auc, pr_auc = acc.aucs()
            slope, ece = acc.rel.slope_ece()
            brier = acc.brier()
            row = {
                "model": name,
                "brier": brier,
                "bss_vs_monthly_climo": 1.0 - brier / ref if np.isfinite(ref) and ref > 0 else float("nan"),
                "base_rate": acc.truth_pos / acc.count if acc.count > 0 else float("nan"),
                "reliability_slope": slope,
                "ece": ece,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            row.update(acc.threshold_scores((0.1, 0.2, 0.3, 0.5)))
            rows.append(row)
        return rows

    def monthly_rows(self, reference_name: str) -> List[Dict[str, float]]:
        rows = []
        for month in sorted(self.monthly):
            ref = self.monthly[month][reference_name].brier()
            for name, acc in self.monthly[month].items():
                roc_auc, pr_auc = acc.aucs()
                slope, ece = acc.rel.slope_ece()
                brier = acc.brier()
                rows.append({
                    "month": month,
                    "model": name,
                    "brier": brier,
                    "bss_vs_monthly_climo": 1.0 - brier / ref if np.isfinite(ref) and ref > 0 else float("nan"),
                    "reliability_slope": slope,
                    "ece": ece,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                })
        return rows

    def region_rows(self) -> List[Dict[str, float]]:
        rows = []
        for name, by_region in self.region_abs.items():
            for region, value in by_region.items():
                denom = max(self.region_count[name].get(region, 0), 1)
                rows.append({
                    "model": name,
                    "region": region,
                    "expected_count_mae": value / denom,
                    "n_samples": self.region_count[name].get(region, 0),
                })
        return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_reliability(path: Path, rel_tables: Mapping[str, List[Mapping[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, rows in rel_tables.items():
        x = np.array([r["mean_pred"] for r in rows], dtype=np.float64)
        y = np.array([r["obs_freq"] for r in rows], dtype=np.float64)
        n = np.array([r["count"] for r in rows], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & (n > 0)
        if np.any(valid):
            ax.plot(x[valid], y[valid], marker="o", label=name)
    ax.set_xlabel("Predicted exceedance probability")
    ax.set_ylabel("Observed exceedance frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_run_start(runs: Sequence[Tuple[int, int]], t: int) -> int:
    for start, end in runs:
        if start <= t <= end:
            return start
    raise RuntimeError(f"Could not locate continuous season for t={t}")


def trailing_exceedance_probability(
    heat: np.ndarray,
    t: int,
    target_t: int,
    window: int,
    run_start: int,
    q95_z: np.ndarray,
    months: np.ndarray,
    norm_stats: Mapping[str, torch.Tensor],
    land_mask: np.ndarray,
) -> np.ndarray:
    lo = max(run_start, t - int(window) + 1)
    hist = np.arange(lo, t + 1, dtype=np.int64)
    if hist.size == 0:
        return np.zeros_like(land_mask, dtype=np.float32)
    if np.max(hist) >= target_t:
        raise RuntimeError("Persistence baseline attempted to use target/future information.")

    count = np.zeros(land_mask.shape, dtype=np.float32)
    exceed = np.zeros(land_mask.shape, dtype=np.float32)
    for ti in hist:
        month = int(months[ti])
        field_z = _normalize_field(np.asarray(heat[:, :, ti]), norm_stats)
        threshold = q95_z[month]
        valid = land_mask & np.isfinite(field_z) & np.isfinite(threshold)
        count[valid] += 1.0
        exceed[valid] += (field_z[valid] > threshold[valid]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = exceed / np.maximum(count, 1.0)
    out[~land_mask] = np.nan
    return out.astype(np.float32)


@dataclass
class PooledLogistic:
    mean: float
    std: float
    coef: float
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.mean) / (self.std + 1e-8)
        logits = np.clip(self.intercept + self.coef * z, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass
class ModelOutputLogisticCalibrator:
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    calibration_split: str
    n_samples: int
    event_rate: float

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.mean) / (self.std + 1e-8)
        logits = np.clip(self.intercept + z @ self.coef, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def predict_grid(
        self,
        mu_z: np.ndarray,
        q95: np.ndarray,
        sigma: np.ndarray,
        month: int,
        region_features: Mapping[str, np.ndarray],
        mask: np.ndarray,
    ) -> np.ndarray:
        valid = mask & np.isfinite(mu_z) & np.isfinite(q95) & np.isfinite(sigma)
        out = np.full(mask.shape, np.nan, dtype=np.float32)
        flat_idx = np.flatnonzero(valid.ravel())
        if flat_idx.size == 0:
            return out
        x = model_output_calibration_features(mu_z, q95, sigma, month, region_features, flat_idx)
        out.ravel()[flat_idx] = self.predict_features(x)
        return out

    def coefficient_rows(self) -> List[Dict[str, object]]:
        rows = [{
            "feature": "intercept",
            "coef": self.intercept,
            "feature_mean": "",
            "feature_std": "",
            "calibration_split": self.calibration_split,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
        }]
        for name, coef, mean, std in zip(self.feature_names, self.coef, self.mean, self.std):
            rows.append({
                "feature": name,
                "coef": float(coef),
                "feature_mean": float(mean),
                "feature_std": float(std),
                "calibration_split": self.calibration_split,
                "n_samples": self.n_samples,
                "event_rate": self.event_rate,
            })
        return rows


def calibration_feature_names(region_names: Sequence[str]) -> Tuple[str, ...]:
    return (
        "stage1_logit",
        "margin_z",
        "mu_z",
        "sigma_clim",
        *[f"month_{m}" for m in MJJAS_MONTHS],
        *[f"region_{name}" for name in region_names],
    )


def model_output_calibration_features(
    mu_z: np.ndarray,
    q95: np.ndarray,
    sigma: np.ndarray,
    month: int,
    region_features: Mapping[str, np.ndarray],
    flat_idx: np.ndarray,
) -> np.ndarray:
    mu = mu_z.ravel()[flat_idx].astype(np.float32)
    q = q95.ravel()[flat_idx].astype(np.float32)
    sig = np.maximum(sigma.ravel()[flat_idx].astype(np.float32), 0.1)
    margin = mu - q
    stage1_logit = np.clip(margin / sig, -10.0, 10.0)
    month_flags = np.zeros((flat_idx.size, len(MJJAS_MONTHS)), dtype=np.float32)
    if int(month) in MJJAS_MONTHS:
        month_flags[:, MJJAS_MONTHS.index(int(month))] = 1.0
    region_flags = np.column_stack([
        rmask.ravel()[flat_idx].astype(np.float32)
        for rmask in region_features.values()
    ]) if region_features else np.empty((flat_idx.size, 0), dtype=np.float32)
    return np.column_stack([
        stage1_logit,
        margin,
        mu,
        sig,
        month_flags,
        region_flags,
    ]).astype(np.float32)


def fit_model_output_logistic_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    calibration_split: str,
    steps: int,
    lr: float,
    l2: float,
) -> ModelOutputLogisticCalibrator:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    valid = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.shape[0] < 100 or np.unique(y).size < 2:
        raise RuntimeError("Not enough valid positive/negative samples to fit calibrated logistic model.")

    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    z = (x - mean) / std
    coef = np.zeros(z.shape[1], dtype=np.float64)
    base = np.clip(y.mean(), 1e-5, 1.0 - 1e-5)
    intercept = float(math.log(base / (1.0 - base)))

    for _ in range(int(steps)):
        logits = np.clip(intercept + z @ coef, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        intercept -= float(lr) * float(err.mean())
        coef -= float(lr) * ((z.T @ err) / z.shape[0] + float(l2) * coef)

    return ModelOutputLogisticCalibrator(
        feature_names=tuple(feature_names),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        coef=coef.astype(np.float32),
        intercept=float(intercept),
        calibration_split=calibration_split,
        n_samples=int(y.size),
        event_rate=float(y.mean()),
    )


def resolve_calibration_split(requested: str, eval_split: str) -> str:
    if requested != "auto":
        return requested
    return "val" if eval_split == "test" else "train"


def train_model_output_calibrator(
    model,
    calibration_dataset,
    calibration_split: str,
    q95_z: np.ndarray,
    sigma_clim: np.ndarray,
    months: np.ndarray,
    region_features: Mapping[str, np.ndarray],
    mask_np: np.ndarray,
    device,
    center_idx: int,
    max_cases: int,
    max_samples: int,
    seed: int,
    steps: int,
    lr: float,
    l2: float,
) -> ModelOutputLogisticCalibrator:
    rng = np.random.default_rng(seed)
    n_cases = min(len(calibration_dataset), max(1, int(max_cases)))
    subset_idx = np.unique(np.linspace(0, len(calibration_dataset) - 1, n_cases, dtype=np.int64))
    loader = DataLoader(Subset(calibration_dataset, subset_idx.tolist()), batch_size=1, shuffle=False, num_workers=0)
    per_case = max(1, int(math.ceil(max_samples / max(len(subset_idx), 1))))
    feature_names = calibration_feature_names(tuple(region_features.keys()))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    print(
        "Training calibrated model-output logistic layer: "
        f"split={calibration_split}, cases={len(subset_idx)}, max_pixels={int(max_samples)}"
    )
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            mu_z, truth_z, t = predict_center_field(model, batch, device, center_idx, cfm.Config.IMAGE_SIZE)
            target_t = t + int(cfm.Config.LEAD_TIME)
            month = int(months[target_t])
            if month not in MJJAS_MONTHS:
                continue
            q = q95_z[month]
            sigma = sigma_clim[month]
            valid = mask_np & np.isfinite(mu_z) & np.isfinite(truth_z) & np.isfinite(q) & np.isfinite(sigma)
            candidates = np.flatnonzero(valid.ravel())
            if candidates.size == 0:
                continue
            chosen = rng.choice(candidates, size=min(per_case, candidates.size), replace=False)
            xs.append(model_output_calibration_features(mu_z, q, sigma, month, region_features, chosen))
            truth = (truth_z.ravel()[chosen] > q.ravel()[chosen]).astype(np.float32)
            ys.append(truth)
            if sum(x.shape[0] for x in xs) >= max_samples:
                break
            if (batch_idx + 1) % 50 == 0:
                print(f"  calibration cases processed {batch_idx + 1}/{len(subset_idx)}")

    if not xs:
        raise RuntimeError("No samples collected for calibrated model-output logistic layer.")
    features = np.concatenate(xs, axis=0)[:max_samples]
    labels = np.concatenate(ys, axis=0)[:max_samples]
    calibrator = fit_model_output_logistic_calibrator(
        features,
        labels,
        feature_names,
        calibration_split=calibration_split,
        steps=int(steps),
        lr=float(lr),
        l2=float(l2),
    )
    print(
        "Calibrated model-output logistic fitted: "
        f"n={calibrator.n_samples}, event_rate={calibrator.event_rate:.4f}, "
        f"intercept={calibrator.intercept:.4f}"
    )
    return calibrator


def train_pooled_logistic_baseline(
    shared_data,
    train_indices: Sequence[int],
    q95_z: np.ndarray,
    norm_stats: Mapping[str, torch.Tensor],
    months: np.ndarray,
    max_samples: int,
    seed: int,
) -> PooledLogistic:
    rng = np.random.default_rng(seed)
    heat = shared_data["heat_index"]
    land_mask = np.isfinite(np.asarray(heat[:, :, 0])) & (np.asarray(heat[:, :, 0]) != 0.0)
    land_flat = np.flatnonzero(land_mask.ravel())
    per_time = max(1, int(math.ceil(max_samples / max(len(train_indices), 1))))

    xs = []
    ys = []
    for t in train_indices:
        target_t = int(t) + int(cfm.Config.LEAD_TIME)
        month = int(months[target_t])
        if month not in MJJAS_MONTHS:
            continue
        chosen = rng.choice(land_flat, size=min(per_time, land_flat.size), replace=False)
        x_t_z = _normalize_field(np.asarray(heat[:, :, int(t)]), norm_stats).ravel()
        y_z = _normalize_field(np.asarray(heat[:, :, target_t]), norm_stats).ravel()
        q = q95_z[month].ravel()
        valid = np.isfinite(x_t_z[chosen]) & np.isfinite(y_z[chosen]) & np.isfinite(q[chosen])
        if np.any(valid):
            idx = chosen[valid]
            xs.append((x_t_z[idx] - q[idx]).astype(np.float32))
            ys.append((y_z[idx] > q[idx]).astype(np.float32))
        if sum(len(a) for a in xs) >= max_samples:
            break

    if not xs:
        raise RuntimeError("Could not sample training data for pooled logistic baseline.")
    x = np.concatenate(xs)[:max_samples].astype(np.float64)
    y = np.concatenate(ys)[:max_samples].astype(np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-8)
    z = (x - mean) / std

    coef = 0.0
    intercept = math.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4))
    lr = 0.2
    for _ in range(300):
        logits = np.clip(intercept + coef * z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        intercept -= lr * float(err.mean())
        coef -= lr * float(np.mean(err * z))
    print(
        "Pooled logistic baseline trained on train years only: "
        f"n={len(y)}, event_rate={y.mean():.4f}, coef={coef:.4f}, intercept={intercept:.4f}"
    )
    return PooledLogistic(mean=mean, std=std, coef=float(coef), intercept=float(intercept))


def predict_center_field(model, batch, device, center_idx: int, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = batch
    h, w = image_size
    pred = cfm.predict_direct(
        model,
        x_t.to(device),
        x_tm1.to(device),
        x_tm2.to(device),
        spatial_c.to(device),
        vec_c.to(device),
        global_fields.to(device),
        batch_mask.to(device),
        device,
    )
    if cfm.Config.MULTI_LEAD_TUBE:
        mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
        truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
    else:
        mu_z = pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
        truth_z = y[0, 0, :h, :w].numpy().astype(np.float32)
    return mu_z, truth_z, int(t_idx.item())


def select_dataset(split: str, config, shared_data, norm_stats, climo, train_indices, val_indices, test_indices):
    if split == "train":
        return cfm.ClimateDataset(config, mode="train", train_indices=train_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    if split == "val":
        return cfm.ClimateDataset(config, mode="val", val_indices=val_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    if split == "test":
        return cfm.ClimateDataset(config, mode="test", test_indices=test_indices, normalization_stats=norm_stats, shared_data=shared_data, target_climatology=climo)
    raise ValueError(f"Unsupported split: {split}")


def evaluate(args: argparse.Namespace) -> None:
    configure_from_args(args)
    out_dir = ensure_dir(Path(args.output_dir) / args.run_name / args.split)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(int(args.local_rank))

    shared_data = cfm.prepare_shared_data(cfm.Config, rank=0, world_size=1, ddp=False)
    time_values = np.asarray(shared_data["time_values"])
    months, years, doys = month_doy_year_arrays(time_values)
    runs = cfm.detect_continuous_runs(time_values)
    train_indices, val_indices, test_indices, train_years, val_years, test_years = split_indices_for_config(shared_data)
    print(f"CV split: {cfm.cv_split_tag(cfm.Config)}")
    print(f"Train years ({len(train_years)}): {sorted(train_years)}")
    print(f"Val years: {sorted(val_years)}")
    print(f"Test years: {sorted(test_years)}")

    split_years = {"train": train_years, "val": val_years, "test": test_years}[args.split]
    if set(split_years) & set(train_years) and args.split != "train":
        raise RuntimeError("Leakage check failed: evaluation split overlaps training years.")

    norm_stats = load_norm_stats()
    climo = cfm.load_or_build_train_climatology(shared_data, train_indices, norm_stats, cfm.Config, ddp=False)
    q95_z, sigma_clim, base_rate = load_or_build_exceedance_stats(
        shared_data, train_years, norm_stats, climo
    )
    for month in MJJAS_MONTHS:
        rate = float(np.nanmean(base_rate[month]))
        print(f"Build check month={month}: train-year mean exceedance rate={rate:.4f}")
        if not (0.025 <= rate <= 0.075):
            raise RuntimeError(f"Train exceedance rate for month {month} is not near 5%: {rate:.4f}")

    dataset = select_dataset(
        args.split, cfm.Config, shared_data, norm_stats, climo,
        train_indices, val_indices, test_indices,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    conus_mask = cfm.load_conus_mask(cfm.Config)
    mask_np = conus_mask.numpy() > 0.5
    mesh = cfm.build_mesh_once(cfm.Config, conus_mask, device, ddp=False)
    ckpt_path = checkpoint_path_for(args.run_name, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    model = cfm._load_meshflownet_checkpoint(ckpt_path, mesh, device)

    regions = {
        name: (rmask & mask_np)
        for name, rmask in region_masks(cfm.Config.IMAGE_SIZE).items()
    }
    calibration_split = resolve_calibration_split(args.calibration_split, args.split)
    split_year_map = {"train": train_years, "val": val_years, "test": test_years}
    if calibration_split == args.split and args.split != "train" and not args.allow_eval_split_calibration:
        raise RuntimeError(
            f"Calibration split '{calibration_split}' matches evaluation split '{args.split}'. "
            "Use a held-out calibration split, or pass --allow_eval_split_calibration only for diagnostics."
        )
    if args.split != "train" and set(split_year_map[calibration_split]) & set(split_years):
        raise RuntimeError("Calibration leakage check failed: calibration years overlap evaluation years.")
    calibration_dataset = select_dataset(
        calibration_split, cfm.Config, shared_data, norm_stats, climo,
        train_indices, val_indices, test_indices,
    )
    model_calibrator = train_model_output_calibrator(
        model,
        calibration_dataset,
        calibration_split,
        q95_z,
        sigma_clim,
        months,
        regions,
        mask_np,
        device,
        center_idx=cfm.center_lead_index(cfm.Config) if cfm.Config.MULTI_LEAD_TUBE else 0,
        max_cases=int(args.max_calibration_cases),
        max_samples=int(args.max_calibration_samples),
        seed=int(args.seed) + 137,
        steps=int(args.calibration_steps),
        lr=float(args.calibration_lr),
        l2=float(args.calibration_l2),
    )

    logistic = train_pooled_logistic_baseline(
        shared_data, train_indices, q95_z, norm_stats, months,
        max_samples=int(args.max_logistic_samples), seed=int(args.seed)
    )

    persistence_windows = parse_int_list(args.persistence_windows)
    model_names = [
        "monthly_climatology",
        "persistence_init",
        *[f"persistence_trailing{w}" for w in persistence_windows],
        "thresholded_point_model",
        "pooled_logistic_init_margin",
        "stage1_mu_sigma_clim",
        "calibrated_model_logistic",
    ]
    acc = EvaluationAccumulator(model_names, regions)

    h, w = cfm.Config.IMAGE_SIZE
    center_idx = cfm.center_lead_index(cfm.Config) if cfm.Config.MULTI_LEAD_TUBE else 0
    heat = shared_data["heat_index"]
    print(f"Evaluating split={args.split}, samples={len(dataset)}")
    leakage_ok = True
    causal_ok = True

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask = batch
            t = int(t_idx.item())
            target_t = t + int(cfm.Config.LEAD_TIME)
            target_month = int(months[target_t])
            if target_month not in MJJAS_MONTHS:
                continue
            if int(years[target_t]) in train_years and args.split != "train":
                leakage_ok = False

            x_t_dev = x_t.to(device)
            x_tm1_dev = x_tm1.to(device)
            x_tm2_dev = x_tm2.to(device)
            spatial_c_dev = spatial_c.to(device)
            vec_c_dev = vec_c.to(device)
            global_fields_dev = global_fields.to(device)
            mask_dev = batch_mask.to(device)

            pred = cfm.predict_direct(
                model, x_t_dev, x_tm1_dev, x_tm2_dev,
                spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device,
            )
            if cfm.Config.MULTI_LEAD_TUBE:
                mu_z = pred[0, center_idx, :h, :w].detach().cpu().numpy().astype(np.float32)
                truth_z = y[0, center_idx, :h, :w].numpy().astype(np.float32)
            else:
                mu_z = pred[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                truth_z = y[0, 0, :h, :w].numpy().astype(np.float32)
            if batch_idx == 0:
                mu_land = mu_z[mask_np & np.isfinite(mu_z)]
                if mu_land.size:
                    p = np.nanpercentile(mu_land, [1, 50, 99])
                    print(
                        "First-sample mu_z land range check: "
                        f"min={np.nanmin(mu_land):+.3f}, p01={p[0]:+.3f}, "
                        f"p50={p[1]:+.3f}, p99={p[2]:+.3f}, max={np.nanmax(mu_land):+.3f}"
                    )

            q = q95_z[target_month]
            sigma = sigma_clim[target_month]
            truth = (truth_z > q).astype(np.float32)
            truth[~mask_np] = np.nan
            stage1 = normal_cdf((mu_z - q) / np.maximum(sigma, 0.1))
            stage1[~mask_np] = np.nan
            calibrated = model_calibrator.predict_grid(mu_z, q, sigma, target_month, regions, mask_np)
            monthly_climo = base_rate[target_month]
            thresholded = (mu_z > q).astype(np.float32)
            thresholded[~mask_np] = np.nan

            init_margin = x_t[0, 0].numpy().astype(np.float32) - q
            logistic_prob = logistic.predict(init_margin)
            logistic_prob[~mask_np] = np.nan

            init_prob = (x_t[0, 0].numpy().astype(np.float32) > q).astype(np.float32)
            init_prob[~mask_np] = np.nan
            run_start = find_run_start(runs, t)
            if t >= target_t:
                causal_ok = False

            forecasts = {
                "monthly_climatology": monthly_climo,
                "persistence_init": init_prob,
                "thresholded_point_model": thresholded,
                "pooled_logistic_init_margin": logistic_prob,
                "stage1_mu_sigma_clim": stage1,
                "calibrated_model_logistic": calibrated,
            }
            for window in persistence_windows:
                forecasts[f"persistence_trailing{window}"] = trailing_exceedance_probability(
                    heat, t, target_t, window, run_start, q95_z, months, norm_stats, mask_np
                )

            for name, prob in forecasts.items():
                acc.update(name, prob, truth, mask_np, target_month)

            if (batch_idx + 1) % max(1, int(args.progress_every)) == 0:
                print(f"  processed {batch_idx + 1}/{len(dataset)} samples")

    if not leakage_ok:
        raise RuntimeError("Leakage assert failed: evaluation target year was in train years.")
    if not causal_ok:
        raise RuntimeError("Causality assert failed for persistence baseline.")

    summary_rows = acc.summary_rows("monthly_climatology")
    monthly_rows = acc.monthly_rows("monthly_climatology")
    region_rows = acc.region_rows()
    rel_tables = {name: metric.rel.table() for name, metric in acc.metrics.items()}

    write_csv(out_dir / "exceedance_results.csv", summary_rows)
    write_csv(out_dir / "monthly_exceedance_results.csv", monthly_rows)
    write_csv(out_dir / "regional_exceedance_count_mae.csv", region_rows)
    write_csv(out_dir / "calibrated_model_logistic_coefficients.csv", model_calibrator.coefficient_rows())
    for name, rows in rel_tables.items():
        write_csv(out_dir / f"reliability_{name}.csv", rows)
    plot_reliability(out_dir / "reliability_diagram.png", rel_tables)

    stage1 = next(row for row in summary_rows if row["model"] == "stage1_mu_sigma_clim")
    calibrated_row = next(row for row in summary_rows if row["model"] == "calibrated_model_logistic")
    thresholded = next(row for row in summary_rows if row["model"] == "thresholded_point_model")
    climo = next(row for row in summary_rows if row["model"] == "monthly_climatology")
    stage1_rel = acc.metrics["stage1_mu_sigma_clim"].rel.table()
    calibrated_rel = acc.metrics["calibrated_model_logistic"].rel.table()
    bin_02 = next((r for r in stage1_rel if r["lo"] <= 0.2 < r["hi"]), None)
    calibrated_bin_02 = next((r for r in calibrated_rel if r["lo"] <= 0.2 < r["hi"]), None)

    print("\nExceedance evaluation summary")
    print("=============================")
    for row in summary_rows:
        print(
            f"{row['model']:28s} Brier={row['brier']:.5f} "
            f"BSS={row['bss_vs_monthly_climo']:+.4f} "
            f"slope={row['reliability_slope']:.3f} ECE={row['ece']:.4f} "
            f"ROC-AUC={row['roc_auc']:.3f} PR-AUC={row['pr_auc']:.3f}"
        )
    print("\nAcceptance checks")
    print("=================")
    print("Leakage asserts: PASS (train-only thresholds/stats; eval years held out)")
    print("Causal persistence asserts: PASS (history windows end at init day)")
    print(f"Monthly climatology Brier reference: {climo['brier']:.5f}")
    print(f"Stage 1 BSS vs monthly climatology: {stage1['bss_vs_monthly_climo']:+.4f}")
    print(
        "Calibrated model-output logistic BSS vs monthly climatology: "
        f"{calibrated_row['bss_vs_monthly_climo']:+.4f}"
    )
    print(
        "Thresholded point-model baseline reported: "
        f"Brier={thresholded['brier']:.5f}, BSS={thresholded['bss_vs_monthly_climo']:+.4f}"
    )
    if bin_02 is not None:
        print(
            "Stage 1 reliability 0.2-bin sanity: "
            f"count={int(bin_02['count'])}, mean_pred={bin_02['mean_pred']:.3f}, "
            f"obs_freq={bin_02['obs_freq']:.3f}"
        )
    if calibrated_bin_02 is not None:
        print(
            "Calibrated reliability 0.2-bin sanity: "
            f"count={int(calibrated_bin_02['count'])}, "
            f"mean_pred={calibrated_bin_02['mean_pred']:.3f}, "
            f"obs_freq={calibrated_bin_02['obs_freq']:.3f}"
        )
    print(
        "PR-AUC is secondary only; selection/headline metric is BSS with reliability diagnostics."
    )
    print(f"Saved exceedance outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--cv_fold", type=int, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", default="best_monitor")
    parser.add_argument("--output_dir", default="exceedance_eval")
    parser.add_argument("--cv_stride", type=int, default=5)
    parser.add_argument("--multi_lead_tube", action="store_true")
    parser.add_argument("--prediction_leads", default="12,13,14,15,16,17,18")
    parser.add_argument("--persistence_windows", default="7,14,30")
    parser.add_argument("--max_logistic_samples", type=int, default=1000000)
    parser.add_argument("--calibration_split", choices=["auto", "train", "val", "test"], default="auto")
    parser.add_argument("--allow_eval_split_calibration", action="store_true")
    parser.add_argument("--max_calibration_cases", type=int, default=256)
    parser.add_argument("--max_calibration_samples", type=int, default=250000)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-4)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
