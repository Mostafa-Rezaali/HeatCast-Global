#!/usr/bin/env python3
"""
Compute validation baselines for direct 15-day heatwave anomaly prediction.

Run on HiPerGator with:
    python3 -u Subdomain/compute_baselines.py
"""

import os
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from scipy import stats  # noqa: F401 - imported to keep scipy available for cluster diagnostics
from sklearn.linear_model import Ridge


WORK_DIR = "/blue/nessie/mostafarezaali/Teleconnection"
TRAINING_DATA_PATH = os.path.join(WORK_DIR, "VDM_Training_Data_Extended_v2.nc")
CLIMATOLOGY_PATH = os.path.join(
    WORK_DIR,
    "data_cache",
    "climatology_daily_1981_2015_sub30_40_-105_-90.npz",
)
OUTPUT_PATH = os.path.join(WORK_DIR, "test_prediction_plots", "baseline_comparison.txt")

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
    print(f"Loaded subdomain t2m_prism: {t2m.shape}")
    return t2m, time_values


def compute_metrics(all_preds, all_truths, land_mask):
    """
    all_preds: list of (H, W) arrays
    all_truths: list of (H, W) arrays
    Returns dict with R2, corr, MSE, RMSE, bias, variance_ratio
    """
    P = np.stack(all_preds)
    T = np.stack(all_truths)
    valid = (np.broadcast_to(land_mask[None, :, :], P.shape) &
             np.isfinite(P) & np.isfinite(T))

    p_flat = P[valid].ravel()
    t_flat = T[valid].ravel()
    ss_res = np.sum((t_flat - p_flat) ** 2)
    ss_tot = np.sum((t_flat - t_flat.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    corrs = []
    for n in range(len(P)):
        sample_valid = valid[n]
        pv = P[n][sample_valid]
        tv = T[n][sample_valid]
        if tv.std() > 1e-6 and pv.std() > 1e-6:
            corrs.append(np.corrcoef(pv, tv)[0, 1])
    corr = np.mean(corrs) if corrs else 0.0

    mse = np.mean((p_flat - t_flat) ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(p_flat) - np.mean(t_flat)
    var_ratio = np.std(p_flat) / (np.std(t_flat) + 1e-8)
    return {
        "R2": float(r2),
        "Corr": float(corr),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "Bias": float(bias),
        "VarRatio": float(var_ratio),
    }


def assert_valid_doy(valid_doys, doy):
    if doy < 1 or doy > 366 or not valid_doys[doy - 1]:
        raise ValueError(f"DOY {doy} is outside populated MJJAS climatology.")


def raw_at(t2m, t):
    arr = np.array(t2m[:, :, t], dtype=np.float32)
    arr[arr == 0.0] = np.nan
    return arr


def anomaly_at(t2m, clim_mean, clim_std, valid_doys, doys, t):
    doy = int(doys[t])
    assert_valid_doy(valid_doys, doy)
    raw = raw_at(t2m, t)
    return (raw - clim_mean[doy - 1]) / (clim_std[doy - 1] + 1e-6)


def pixel_raw_series(t2m, indices, i, j, offset=0):
    idx = np.array(indices, dtype=np.int64) + int(offset)
    vals = np.array(t2m[i, j, idx], dtype=np.float32)
    vals[vals == 0.0] = np.nan
    return vals


def pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, indices, i, j, offset=0):
    idx = np.array(indices, dtype=np.int64) + int(offset)
    doy_idx = doys[idx].astype(np.int32) - 1
    in_range = (doy_idx >= 0) & (doy_idx < 366)
    bad_mask = ~in_range
    if np.any(in_range):
        bad_mask[in_range] |= ~valid_doys[doy_idx[in_range]]
    if np.any(bad_mask):
        bad = sorted(set((doy_idx[bad_mask] + 1).tolist()))
        raise ValueError(f"Pixel anomaly series requested invalid DOYs: {bad[:20]}")
    raw = pixel_raw_series(t2m, indices, i, j, offset=offset)
    clim = clim_mean[doy_idx, i, j]
    cstd = clim_std[doy_idx, i, j]
    return (raw - clim) / (cstd + 1e-6)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    clim = np.load(CLIMATOLOGY_PATH)
    clim_mean = clim["clim_mean"].astype(np.float32)
    clim_std = clim["clim_std"].astype(np.float32)
    valid_doys = clim["valid_doys"].astype(bool)
    print(f"Loaded climatology: {int(valid_doys.sum())} valid DOYs from {CLIMATOLOGY_PATH}")

    t2m, time_values = load_temperature_subdomain()
    dates = np.array([time_to_date(tv) for tv in time_values], dtype=object)
    years = np.array([dt.year for dt in dates], dtype=np.int32)
    doys = np.array([dt.timetuple().tm_yday for dt in dates], dtype=np.int32)

    runs = detect_continuous_runs(time_values)
    valid_indices = build_valid_indices(runs, lead_time=LEAD_TIME, min_history=2)
    train_set, val_set, test_set = set(TRAIN_YEARS), set(VAL_YEARS), set(TEST_YEARS)
    train_indices = [i for i in valid_indices if years[i] in train_set]
    val_indices = [i for i in valid_indices if years[i] in val_set]
    test_indices = [i for i in valid_indices if years[i] in test_set]
    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    land = (t2m[:, :, 0] != 0.0) & np.isfinite(t2m[:, :, 0])
    print(f"Land pixels: {int(land.sum())} / {land.size}")

    for t in valid_indices:
        for idx in (t - 2, t - 1, t, t + LEAD_TIME):
            assert_valid_doy(valid_doys, int(doys[idx]))
    print("Verified all baseline DOY lookups are inside populated MJJAS climatology.")

    results = {}

    preds, truths = [], []
    for t in val_indices:
        preds.append(raw_at(t2m, t))
        truths.append(raw_at(t2m, t + LEAD_TIME))
    results["Raw persistence"] = compute_metrics(preds, truths, land)

    preds, truths = [], []
    for t in val_indices:
        doy_tgt = int(doys[t + LEAD_TIME])
        assert_valid_doy(valid_doys, doy_tgt)
        preds.append(clim_mean[doy_tgt - 1])
        truths.append(raw_at(t2m, t + LEAD_TIME))
    results["Climatology only"] = compute_metrics(preds, truths, land)

    preds, truths = [], []
    for t in val_indices:
        doy_t = int(doys[t])
        doy_tgt = int(doys[t + LEAD_TIME])
        assert_valid_doy(valid_doys, doy_t)
        assert_valid_doy(valid_doys, doy_tgt)
        anom_t = raw_at(t2m, t) - clim_mean[doy_t - 1]
        preds.append(clim_mean[doy_tgt - 1] + anom_t)
        truths.append(raw_at(t2m, t + LEAD_TIME))
    results["Anomaly persistence"] = compute_metrics(preds, truths, land)

    print("Fitting per-pixel ridge regression baseline...")
    h, w, _ = t2m.shape
    land_points = np.argwhere(land)
    ridge_pred = np.full((len(val_indices), h, w), np.nan, dtype=np.float32)
    ridge_truth = np.full((len(val_indices), h, w), np.nan, dtype=np.float32)

    train_doy = doys[np.array(train_indices)]
    val_doy = doys[np.array(val_indices)]
    x_season_train = np.stack(
        [
            np.sin(2.0 * np.pi * train_doy / 365.25),
            np.cos(2.0 * np.pi * train_doy / 365.25),
        ],
        axis=1,
    ).astype(np.float32)
    x_season_val = np.stack(
        [
            np.sin(2.0 * np.pi * val_doy / 365.25),
            np.cos(2.0 * np.pi * val_doy / 365.25),
        ],
        axis=1,
    ).astype(np.float32)

    for p_idx, (i, j) in enumerate(land_points, start=1):
        x_train_cols = [
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, train_indices, i, j, offset=0),
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, train_indices, i, j, offset=-1),
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, train_indices, i, j, offset=-2),
        ]
        X_train = np.column_stack([*x_train_cols, x_season_train]).astype(np.float32)
        y_train = pixel_anomaly_series(
            t2m, clim_mean, clim_std, valid_doys, doys, train_indices, i, j, offset=LEAD_TIME
        ).astype(np.float32)
        finite_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        if finite_train.sum() < 10:
            continue

        x_val_cols = [
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, val_indices, i, j, offset=0),
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, val_indices, i, j, offset=-1),
            pixel_anomaly_series(t2m, clim_mean, clim_std, valid_doys, doys, val_indices, i, j, offset=-2),
        ]
        X_val = np.column_stack([*x_val_cols, x_season_val]).astype(np.float32)
        finite_val = np.isfinite(X_val).all(axis=1)

        model = Ridge(alpha=1.0)
        model.fit(X_train[finite_train], y_train[finite_train])
        pred_anom = np.full(len(val_indices), np.nan, dtype=np.float32)
        pred_anom[finite_val] = model.predict(X_val[finite_val]).astype(np.float32)

        val_tgt_idx = np.array(val_indices, dtype=np.int64) + LEAD_TIME
        val_tgt_doy_idx = doys[val_tgt_idx].astype(np.int32) - 1
        ridge_pred[:, i, j] = (
            clim_mean[val_tgt_doy_idx, i, j]
            + pred_anom * clim_std[val_tgt_doy_idx, i, j]
        )
        ridge_truth[:, i, j] = pixel_raw_series(t2m, val_indices, i, j, offset=LEAD_TIME)

        if p_idx % 10000 == 0 or p_idx == len(land_points):
            print(f"  Ridge pixels: {p_idx}/{len(land_points)}")

    results["Per-pixel ridge"] = compute_metrics(list(ridge_pred), list(ridge_truth), land)

    header = f"{'Baseline':24s} {'R2':>10s} {'Corr':>10s} {'MSE':>12s} {'RMSE':>10s} {'Bias':>10s} {'VarRatio':>10s}"
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(
            f"{name:24s} {metrics['R2']:10.4f} {metrics['Corr']:10.4f} "
            f"{metrics['MSE']:12.4f} {metrics['RMSE']:10.4f} "
            f"{metrics['Bias']:10.4f} {metrics['VarRatio']:10.4f}"
        )
    output = "\n".join(lines)
    print("\n" + output)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"Saved baseline table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
