#!/usr/bin/env python3
"""
Compute MJJAS-only daily climatology for the heatwave subdomain.

Run on HiPerGator with:
    python3 -u Subdomain/compute_climatology.py
"""

import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from scipy.ndimage import gaussian_filter1d


WORK_DIR = "/blue/nessie/mostafarezaali/Teleconnection"
TRAINING_DATA_PATH = os.path.join(WORK_DIR, "VDM_Training_Data_Extended_v2.nc")
OUTPUT_PATH = os.path.join(
    WORK_DIR,
    "data_cache",
    "climatology_daily_1981_2015_sub30_40_-105_-90.npz",
)
PLOT_PATH = os.path.join(WORK_DIR, "test_prediction_plots", "climatology_verification.png")

LAT_SLICE = slice(124, 373)
LON_SLICE = slice(502, 803)
TRAIN_YEARS = list(range(1981, 2016))
VAL_YEARS = list(range(2016, 2020))
TEST_YEARS = list(range(2020, 2024))
LEAD_TIME = 15
BASE_DATE = datetime(1981, 5, 1)


def time_to_date(tv):
    return BASE_DATE + timedelta(days=float(tv))


def detect_continuous_runs(time_values):
    runs, start = [], 0
    for i in range(1, len(time_values)):
        if time_values[i] - time_values[i - 1] > 1.5:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(time_values) - 1))
    return runs


def build_valid_indices(runs, lead_time, min_history=2):
    indices = []
    for s, e in runs:
        for t in range(s + min_history, e - lead_time + 1):
            indices.append(t)
    return indices


def load_temperature_subdomain():
    print(f"Opening {TRAINING_DATA_PATH}")
    with NetCDFDataset(TRAINING_DATA_PATH, "r") as nc:
        raw = nc.variables["t2m_prism"][:]
        if raw.ndim == 4:
            t2m = np.array(raw[:, :, 0, :], dtype=np.float32)
        elif raw.ndim == 3:
            t2m = np.array(raw, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected t2m_prism shape: {raw.shape}")
        time_values = np.array(nc.variables["time"][:], dtype=np.float64)

    t2m = np.ascontiguousarray(t2m[LAT_SLICE, LON_SLICE, :], dtype=np.float32)
    print(f"Loaded subdomain t2m_prism: {t2m.shape}, dtype={t2m.dtype}")
    print(f"Loaded time: {time_values.shape}, range=({time_values[0]}, {time_values[-1]})")
    return t2m, time_values


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    t2m, time_values = load_temperature_subdomain()
    h, w, n_time = t2m.shape

    dates = np.array([time_to_date(tv) for tv in time_values], dtype=object)
    years = np.array([dt.year for dt in dates], dtype=np.int32)
    doys = np.array([dt.timetuple().tm_yday for dt in dates], dtype=np.int32)

    train_mask = np.isin(years, np.array(TRAIN_YEARS, dtype=np.int32))
    train_doys = doys[train_mask]
    doy_min, doy_max = int(train_doys.min()), int(train_doys.max())
    print(f"Training-year DOY range: {doy_min}-{doy_max} ({len(np.unique(train_doys))} unique DOYs)")
    # Leap years put Sep 30 at DOY 274, so allow the MJJAS envelope 121-274.
    if doy_min < 121 or doy_max > 274:
        raise ValueError(f"Unexpected MJJAS DOY range {doy_min}-{doy_max}")

    clim_mean = np.full((366, h, w), np.nan, dtype=np.float32)
    clim_std = np.full((366, h, w), np.nan, dtype=np.float32)
    valid_doys = np.zeros(366, dtype=bool)

    unique_doys = np.array(sorted(np.unique(train_doys)), dtype=np.int32)
    for count, doy in enumerate(unique_doys, start=1):
        t_idx = np.where(train_mask & (doys == doy))[0]
        stack = np.array(t2m[:, :, t_idx], dtype=np.float32)
        stack = np.moveaxis(stack, -1, 0)
        stack[stack == 0.0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            clim_mean[doy - 1] = np.nanmean(stack, axis=0).astype(np.float32)
            clim_std[doy - 1] = np.nanstd(stack, axis=0).astype(np.float32)
        valid_doys[doy - 1] = True
        if count % 10 == 0 or count == 1 or count == len(unique_doys):
            print(f"  DOY {doy:03d}: {len(t_idx)} samples ({count}/{len(unique_doys)})")

    valid_idx = np.where(valid_doys)[0]
    print(f"Smoothing {len(valid_idx)} valid DOYs with gaussian_filter1d(mode='nearest')")
    clim_mean_valid = gaussian_filter1d(clim_mean[valid_idx], sigma=7, axis=0, mode="nearest")
    clim_std_valid = gaussian_filter1d(clim_std[valid_idx], sigma=7, axis=0, mode="nearest")
    clim_mean[valid_idx] = clim_mean_valid.astype(np.float32)
    clim_std[valid_idx] = clim_std_valid.astype(np.float32)

    land_mask_2d = np.any(np.isfinite(clim_mean[valid_idx]), axis=0)
    for i in valid_idx:
        land_pixels = land_mask_2d & np.isfinite(clim_mean[i])
        clim_std[i][land_pixels] = np.maximum(clim_std[i][land_pixels], 0.1)

    runs = detect_continuous_runs(time_values)
    valid_indices = build_valid_indices(runs, lead_time=LEAD_TIME, min_history=2)
    lookup_doys = []
    for t in valid_indices:
        lookup_doys.extend([doys[t - 2], doys[t - 1], doys[t], doys[t + LEAD_TIME]])
    lookup_doys = np.array(lookup_doys, dtype=np.int32)
    bad = sorted(set(int(d) for d in lookup_doys if d < 1 or d > 366 or not valid_doys[d - 1]))
    if bad:
        raise ValueError(f"Found DOY lookups outside populated MJJAS climatology: {bad[:20]}")
    print(f"Verified all lead-{LEAD_TIME} history/target DOY lookups are within populated MJJAS DOYs.")

    np.savez(
        OUTPUT_PATH,
        clim_mean=clim_mean.astype(np.float32),
        clim_std=clim_std.astype(np.float32),
        valid_doys=valid_doys,
        train_years=np.array(TRAIN_YEARS, dtype=np.int32),
    )
    print(f"Saved climatology to {OUTPUT_PATH}")

    doy_mid = 200
    if not valid_doys[doy_mid - 1] or not valid_doys[doy_mid]:
        raise ValueError("DOY 200/201 unavailable; climatology verification cannot run.")
    land_mid = land_mask_2d & np.isfinite(clim_mean[doy_mid - 1])
    vals = clim_mean[doy_mid - 1][land_mid]
    print(
        f"DOY 200 clim_mean over land: mean={np.nanmean(vals):.4f}, "
        f"std={np.nanstd(vals):.4f}, min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}"
    )
    v200 = clim_mean[doy_mid - 1][land_mid]
    v201 = clim_mean[doy_mid][land_mid]
    finite = np.isfinite(v200) & np.isfinite(v201)
    corr = np.corrcoef(v200[finite], v201[finite])[0, 1]
    print(f"Spatial correlation DOY 200 vs 201 climatology: {corr:.6f} (expect >0.99)")
    if corr <= 0.99:
        print("WARNING: DOY 200/201 climatology correlation is lower than expected.")

    plot_doys = [140, 180, 220, 260]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, doy in zip(axes.ravel(), plot_doys):
        if not valid_doys[doy - 1]:
            raise ValueError(f"Plot DOY {doy} is not populated in climatology.")
        im = ax.imshow(np.where(land_mask_2d, clim_mean[doy - 1], np.nan), cmap="hot", aspect="auto")
        ax.set_title(f"DOY {doy}")
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("MJJAS Daily Climatology Verification")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved verification plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
