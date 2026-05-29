#!/usr/bin/env python3
"""Validate one plotted ground-truth target slice without running the model.

Reads only one or a few memory-mapped slices from data_cache/heat_index.npy.
Useful for checking suspicious validation plots for wrong date, fill values,
normalization surprises, time discontinuities, and row orientation.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np


BASE_DATE = datetime(1981, 5, 1)
DEFAULT_CACHE_DIR = "data_cache"
DEFAULT_DATA_PATH = "/blue/nessie/mostafarezaali/Teleconnection/VDM_Training_Data_Extended_v2.nc"


def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def date_for_time_value(value: float) -> datetime:
    return BASE_DATE + timedelta(days=float(value))


def date_strings(time_values: np.ndarray) -> List[str]:
    return [date_for_time_value(v).strftime("%Y-%m-%d") for v in time_values]


def find_date_index(time_values: np.ndarray, date_text: str) -> int:
    target = parse_date(date_text).date()
    for i, value in enumerate(time_values):
        if date_for_time_value(value).date() == target:
            return int(i)
    raise ValueError(f"Date {date_text} was not found in time_values.npy")


def compute_doy(value: float) -> int:
    dt = date_for_time_value(value)
    return int(dt.timetuple().tm_yday)


def detect_runs(time_values: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    start = 0
    for i in range(1, len(time_values)):
        if time_values[i] - time_values[i - 1] > 1.5:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(time_values) - 1))
    return runs


def valid_indices(runs: Sequence[Tuple[int, int]], lead: int, min_history: int) -> List[int]:
    out = []
    for start, end in runs:
        first = start + int(min_history)
        last = end - int(lead)
        out.extend(range(first, last + 1))
    return out


def split_indices(
    all_valid: Sequence[int],
    time_values: np.ndarray,
    cv_fold: int,
    cv_stride: int,
) -> Tuple[List[int], List[int], List[int]]:
    val_offset = (int(cv_fold) + 1) % int(cv_stride)
    test_offset = int(cv_fold) % int(cv_stride)
    years = np.array([date_for_time_value(time_values[i]).year for i in all_valid])
    unique_years = sorted(set(int(y) for y in years))
    val_years = set(unique_years[val_offset::cv_stride])
    test_years = set(unique_years[test_offset::cv_stride])
    train_years = set(unique_years) - val_years - test_years
    train = [i for i, y in zip(all_valid, years) if int(y) in train_years]
    val = [i for i, y in zip(all_valid, years) if int(y) in val_years]
    test = [i for i, y in zip(all_valid, years) if int(y) in test_years]
    return train, val, test


def finite_land(mask: np.ndarray, field: np.ndarray) -> np.ndarray:
    return (mask > 0.5) & np.isfinite(field)


def masked_values(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = np.asarray(field[finite_land(mask, field)], dtype=np.float64)
    return vals[np.isfinite(vals)]


def stats_line(name: str, field: np.ndarray, mask: np.ndarray) -> str:
    vals = masked_values(field, mask)
    if vals.size == 0:
        return f"{name:<16s} no valid land values"
    q = np.nanpercentile(vals, [0, 1, 5, 50, 95, 99, 100])
    return (
        f"{name:<16s} min={q[0]:8.3f} p01={q[1]:8.3f} p05={q[2]:8.3f} "
        f"p50={q[3]:8.3f} p95={q[4]:8.3f} p99={q[5]:8.3f} max={q[6]:8.3f}"
    )


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_land(mask, a) & np.isfinite(b)
    if np.sum(valid) < 10:
        return float("nan")
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def land_mean(field: np.ndarray, mask: np.ndarray) -> float:
    vals = masked_values(field, mask)
    return float(np.nanmean(vals)) if vals.size else float("nan")


def row_band_mean(field: np.ndarray, mask: np.ndarray, lo_frac: float, hi_frac: float) -> float:
    h = field.shape[0]
    lo = int(round(h * lo_frac))
    hi = int(round(h * hi_frac))
    return land_mean(field[lo:hi], mask[lo:hi])


def try_print_netcdf_lat_metadata(path: str) -> None:
    if not path or not os.path.exists(path):
        print("NetCDF lat metadata: source file not found/skipped")
        return
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        print(f"NetCDF lat metadata: skipped ({exc})")
        return
    lat_names = ("lat", "latitude", "y")
    lon_names = ("lon", "longitude", "x")
    try:
        with Dataset(path, "r") as nc:
            for name in lat_names:
                if name in nc.variables:
                    lat = np.asarray(nc.variables[name][:])
                    print(
                        f"NetCDF latitude variable {name!r}: shape={lat.shape}, "
                        f"first={float(np.ravel(lat)[0]):.3f}, last={float(np.ravel(lat)[-1]):.3f}"
                    )
                    break
            else:
                print("NetCDF latitude variable: not found among lat/latitude/y")
            for name in lon_names:
                if name in nc.variables:
                    lon = np.asarray(nc.variables[name][:])
                    print(
                        f"NetCDF longitude variable {name!r}: shape={lon.shape}, "
                        f"first={float(np.ravel(lon)[0]):.3f}, last={float(np.ravel(lon)[-1]):.3f}"
                    )
                    break
    except Exception as exc:
        print(f"NetCDF lat metadata: failed to inspect ({exc})")


def same_calendar_day_indices(time_values: np.ndarray, target_dt: datetime) -> List[int]:
    out = []
    for i, value in enumerate(time_values):
        dt = date_for_time_value(value)
        if dt.month == target_dt.month and dt.day == target_dt.day:
            out.append(i)
    return out


def maybe_save_png(path: str, z: np.ndarray, anomaly: np.ndarray | None, mask: np.ndarray) -> None:
    if not path:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PNG not saved; matplotlib unavailable: {exc}")
        return

    z_plot = np.where(mask > 0.5, z, np.nan)
    if anomaly is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(z_plot, origin="upper", cmap="hot", vmin=-3, vmax=3, aspect="auto")
        axes[0].set_title("z-score, origin=upper")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(z_plot, origin="lower", cmap="hot", vmin=-3, vmax=3, aspect="auto")
        axes[1].set_title("z-score, origin=lower")
        fig.colorbar(im1, ax=axes[1])
    else:
        anom_plot = np.where(mask > 0.5, anomaly, np.nan)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(z_plot, origin="upper", cmap="hot", vmin=-3, vmax=3, aspect="auto")
        axes[0].set_title("z-score")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(anom_plot, origin="upper", cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        axes[1].set_title("z - train DOY climo")
        fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(z_plot, origin="lower", cmap="hot", vmin=-3, vmax=3, aspect="auto")
        axes[2].set_title("z-score, origin=lower")
        fig.colorbar(im2, ax=axes[2])
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual orientation check to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Optional source NetCDF, used only for lat metadata.")
    parser.add_argument("--norm-stats", default="data_cache/norm_stats_direct15_cv5_val3_test2.npz")
    parser.add_argument("--climo", default="data_cache/local_daily_climo30_direct15_cv5_val3_test2.npy")
    parser.add_argument("--target-date", default="2019-09-30")
    parser.add_argument("--init-date", default=None)
    parser.add_argument("--val-index", type=int, default=None, help="Validation dataset index from the plot title.")
    parser.add_argument("--cv-fold", type=int, default=2)
    parser.add_argument("--cv-stride", type=int, default=5)
    parser.add_argument("--lead", type=int, default=15)
    parser.add_argument("--min-history", type=int, default=19)
    parser.add_argument("--save-png", default=None)
    args = parser.parse_args()

    heat_path = os.path.join(args.cache_dir, "heat_index.npy")
    time_path = os.path.join(args.cache_dir, "time_values.npy")
    if not os.path.exists(heat_path):
        raise FileNotFoundError(f"Missing {heat_path}; run from the Teleconnection work dir.")
    if not os.path.exists(time_path):
        raise FileNotFoundError(f"Missing {time_path}; run from the Teleconnection work dir.")

    heat = np.load(heat_path, mmap_mode="r")
    time_values = np.load(time_path)
    if heat.ndim != 3:
        raise ValueError(f"Expected heat_index.npy shape (H, W, T), got {heat.shape}")

    if args.val_index is not None:
        runs = detect_runs(time_values)
        all_valid = valid_indices(runs, args.lead, args.min_history)
        _, val_indices, _ = split_indices(all_valid, time_values, args.cv_fold, args.cv_stride)
        if args.val_index < 0 or args.val_index >= len(val_indices):
            raise IndexError(f"--val-index {args.val_index} outside validation size {len(val_indices)}")
        init_idx = int(val_indices[args.val_index])
        target_idx = init_idx + int(args.lead)
    elif args.init_date:
        init_idx = find_date_index(time_values, args.init_date)
        target_idx = init_idx + int(args.lead)
    else:
        target_idx = find_date_index(time_values, args.target_date)
        init_idx = target_idx - int(args.lead)

    init_dt = date_for_time_value(time_values[init_idx])
    target_dt = date_for_time_value(time_values[target_idx])
    target_doy = compute_doy(time_values[target_idx])

    norm = np.load(args.norm_stats)
    hi_mean = float(norm["hi_mean"])
    hi_std = float(norm["hi_std"])
    norm.close()

    first = np.asarray(heat[:, :, 0], dtype=np.float32)
    mask = (np.abs(first) > 0.01) & np.isfinite(first)
    raw = np.asarray(heat[:, :, target_idx], dtype=np.float32)
    init_raw = np.asarray(heat[:, :, init_idx], dtype=np.float32)
    z = (np.where(np.isfinite(raw) & (np.abs(raw) > 0.01), raw, hi_mean) - hi_mean) / (hi_std + 1e-8)

    climo = None
    anomaly = None
    if args.climo and os.path.exists(args.climo):
        climo = np.load(args.climo, mmap_mode="r")
        anomaly = z - np.asarray(climo[target_doy], dtype=np.float32)

    print("Truth-slice validation")
    print("======================")
    print(f"heat_index shape: {heat.shape}")
    print(f"Init:   idx={init_idx}, date={init_dt.strftime('%Y-%m-%d')}")
    print(f"Target: idx={target_idx}, date={target_dt.strftime('%Y-%m-%d')}, doy={target_doy}")
    print(f"Lead days from indices: {target_idx - init_idx}")
    print(f"Norm stats: hi_mean={hi_mean:.4f}, hi_std={hi_std:.4f}, path={args.norm_stats}")
    if climo is not None:
        print(f"Loaded local daily climatology: {args.climo}")
    print()

    print("Raw/z-score distribution over land")
    print("==================================")
    print(stats_line("raw target", raw, mask))
    print(stats_line("z target", z, mask))
    if anomaly is not None:
        print(stats_line("z-climo anomaly", anomaly, mask))
    zeros = float(np.mean((np.abs(raw[mask]) <= 0.01) | ~np.isfinite(raw[mask])))
    z_vals = masked_values(z, mask)
    print(f"Missing/zero-like target fraction over land: {100.0 * zeros:.4f}%")
    print(f"Fraction clipped by plot colorbar z<=-3: {100.0 * np.mean(z_vals <= -3.0):.2f}%")
    print(f"Fraction clipped by plot colorbar z>=+3: {100.0 * np.mean(z_vals >= 3.0):.2f}%")
    print()

    print("Temporal sanity")
    print("===============")
    print(f"Init raw land mean:   {land_mean(init_raw, mask):.3f}")
    print(f"Target raw land mean: {land_mean(raw, mask):.3f}")
    if target_idx > 0:
        prev_raw = np.asarray(heat[:, :, target_idx - 1], dtype=np.float32)
        print(
            f"Corr(target, previous day {date_for_time_value(time_values[target_idx - 1]).strftime('%Y-%m-%d')}): "
            f"{corr(raw, prev_raw, mask):.4f}"
        )
    if target_idx + 1 < heat.shape[-1]:
        next_raw = np.asarray(heat[:, :, target_idx + 1], dtype=np.float32)
        print(
            f"Corr(target, next day {date_for_time_value(time_values[target_idx + 1]).strftime('%Y-%m-%d')}): "
            f"{corr(raw, next_raw, mask):.4f}"
        )
    same_day = same_calendar_day_indices(time_values, target_dt)
    if len(same_day) > 3:
        means = np.array([land_mean(np.asarray(heat[:, :, i], dtype=np.float32), mask) for i in same_day])
        rank = 1 + int(np.sum(means < land_mean(raw, mask)))
        q = np.percentile(means, [5, 50, 95])
        print(
            f"Same calendar day land-mean raw over {len(same_day)} years: "
            f"p05={q[0]:.3f}, p50={q[1]:.3f}, p95={q[2]:.3f}; "
            f"target rank={rank}/{len(means)} from cold to warm"
        )
    print()

    print("Orientation sanity")
    print("==================")
    top_raw = row_band_mean(raw, mask, 0.00, 0.20)
    bottom_raw = row_band_mean(raw, mask, 0.80, 1.00)
    top_z = row_band_mean(z, mask, 0.00, 0.20)
    bottom_z = row_band_mean(z, mask, 0.80, 1.00)
    print(f"Array top 20% mean:    raw={top_raw:.3f}, z={top_z:.3f}")
    print(f"Array bottom 20% mean: raw={bottom_raw:.3f}, z={bottom_z:.3f}")
    if top_raw < bottom_raw:
        print("Temperature gradient suggests array row 0 is the colder/northern side for this date.")
    else:
        print("Temperature gradient suggests array row 0 is the warmer/southern side for this date.")
    print("Note: cfm_mesh_train.py currently builds model lat_1d as 25N -> 50N.")
    try_print_netcdf_lat_metadata(args.data)
    print()

    print("Checks")
    print("======")
    if target_dt.strftime("%Y-%m-%d") != args.target_date and args.val_index is None:
        print(f"[CHECK] Target date mismatch: requested {args.target_date}, got {target_dt:%Y-%m-%d}")
    if zeros > 0.001:
        print("[CHECK] Nontrivial zero/missing values on land; raw target may contain fill values.")
    if np.mean(z_vals <= -3.0) > 0.05:
        print("[CHECK] More than 5% of land is below z=-3, so the plot is heavily color-clipped.")
    if anomaly is not None:
        anom_vals = masked_values(anomaly, mask)
        if np.nanpercentile(np.abs(anom_vals), 99) < 2.0:
            print("[OK] After subtracting train-year daily climatology, anomalies are within a normal-looking range.")
    print("[OK] If neighbor-day correlations are high and zero fraction is near 0, the ground-truth slice is probably real.")

    maybe_save_png(args.save_png, z, anomaly, mask)


if __name__ == "__main__":
    main()
