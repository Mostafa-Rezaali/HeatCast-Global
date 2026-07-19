"""Area-weighted global HeatCast evaluation for week 3, week 4, and W34.

This module is the global-domain companion to :mod:`exceedance_eval`.  It keeps
threshold estimation fold-safe, applies the NH-land/MJJAS headline mask, and
computes continuous, probabilistic, reliability, spread-error, and tail
diagnostics without changing the legacy CONUS calibration path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

from init_calendar import MJJAS_MONTHS, WEEK3_LEADS, WEEK4_LEADS, W34_LEADS, window_falls_in_months
from spatial_weights import area_weights


GLOBAL_WINDOWS: Mapping[str, Tuple[int, ...]] = {
    "week3": WEEK3_LEADS,
    "week4": WEEK4_LEADS,
    "w34": W34_LEADS,
}
THRESHOLD_QUANTILES: Mapping[str, float] = {"upper_tercile": 2.0 / 3.0, "q95": 0.95}
TAIL_Z = {"q99": 2.3263478740408408, "q999": 3.090232306167813}

# Inclusive latitude bounds and longitude bounds in degrees east.  Wrapped
# longitude boxes are represented by west > east.
GLOBAL_REGION_BOXES: Mapping[str, Tuple[float, float, float, float]] = {
    "europe": (35.0, 72.0, 350.0, 40.0),
    "conus": (24.0, 50.0, 235.0, 294.0),
    "east_asia": (20.0, 55.0, 100.0, 150.0),
    "south_asia": (5.0, 35.0, 60.0, 100.0),
    "middle_east_north_africa": (15.0, 40.0, 340.0, 65.0),
    "boreal_high_latitude_land": (60.0, 90.0, 0.0, 360.0),
}


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    label = int(value)
    return date(label // 10000, (label // 100) % 100, label % 100)


def _lead_indices(prediction_leads: Sequence[int], requested: Sequence[int]) -> Tuple[int, ...]:
    available = tuple(int(value) for value in prediction_leads)
    missing = [int(value) for value in requested if int(value) not in available]
    if missing:
        raise ValueError(f"Requested leads {missing} are absent from prediction leads {available}.")
    return tuple(available.index(int(value)) for value in requested)


def window_means(
    daily_fields,
    prediction_leads: Sequence[int] = W34_LEADS,
    windows: Mapping[str, Sequence[int]] = GLOBAL_WINDOWS,
) -> Dict[str, np.ndarray]:
    """Return lead-window means from arrays shaped ``(..., lead, lat, lon)``."""
    values = np.asarray(daily_fields)
    if values.ndim < 3:
        raise ValueError("Daily fields must include lead, latitude, and longitude dimensions.")
    lead_axis = values.ndim - 3
    if values.shape[lead_axis] != len(tuple(prediction_leads)):
        raise ValueError("Prediction-lead count does not match the daily-field lead dimension.")
    return {
        name: np.mean(np.take(values, _lead_indices(prediction_leads, leads), axis=lead_axis), axis=lead_axis)
        for name, leads in windows.items()
    }


def nh_land_mjjas_mask(
    lat,
    land_mask,
    initialization,
    leads: Sequence[int] = W34_LEADS,
) -> np.ndarray:
    """Return the NH-land headline mask when the full valid window is MJJAS."""
    latitude = np.asarray(lat, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if latitude.ndim != 1 or land.ndim != 2 or land.shape[0] != latitude.size:
        raise ValueError("Expected one-dimensional latitude and a matching 2-D land mask.")
    if not window_falls_in_months(_as_date(initialization), leads, MJJAS_MONTHS):
        return np.zeros_like(land, dtype=bool)
    return land & (latitude[:, None] >= 0.0)


def local_warm_season_mask(
    local_warm_months,
    land_mask,
    initialization,
    leads: Sequence[int],
) -> np.ndarray:
    """Return a global-land supplement mask using cell-local warm months.

    ``local_warm_months`` has shape ``(lat, lon, n_month_slots)`` and must be
    estimated from training years.  This path deliberately cannot reuse the NH
    MJJAS calendar for Southern Hemisphere headline claims.
    """
    months = np.asarray(local_warm_months, dtype=np.int16)
    land = np.asarray(land_mask, dtype=bool)
    if months.ndim != 3 or months.shape[:2] != land.shape:
        raise ValueError("Local warm-season months must have shape (lat, lon, slots).")
    valid_months = sorted({(_as_date(initialization) + timedelta(days=int(lead))).month for lead in leads})
    month_is_local = [(months == month).any(axis=-1) for month in valid_months]
    return land & np.logical_and.reduce(month_is_local)


def region_masks(lat, lon, land_mask) -> Dict[str, np.ndarray]:
    """Build the paper region table on a rectilinear global grid."""
    latitude = np.asarray(lat, dtype=np.float64)
    longitude = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    land = np.asarray(land_mask, dtype=bool)
    if land.shape != (latitude.size, longitude.size):
        raise ValueError("Land mask shape must equal (latitude, longitude).")
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    out: Dict[str, np.ndarray] = {}
    for name, (south, north, west, east) in GLOBAL_REGION_BOXES.items():
        lon_ok = np.ones_like(lon2d, dtype=bool) if east - west >= 360.0 else (
            (lon2d >= west) & (lon2d <= east) if west <= east else (lon2d >= west) | (lon2d <= east)
        )
        out[name] = land & (lat2d >= south) & (lat2d <= north) & lon_ok
    return out


def build_fold_window_thresholds(
    truth_daily,
    initialization_dates: Sequence[object],
    train_years: Sequence[int],
    *,
    prediction_leads: Sequence[int] = W34_LEADS,
    windows: Mapping[str, Sequence[int]] = GLOBAL_WINDOWS,
    quantiles: Mapping[str, float] = THRESHOLD_QUANTILES,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Estimate per-cell/month thresholds from training-year initializations only."""
    truth = np.asarray(truth_daily, dtype=np.float64)
    dates = tuple(_as_date(value) for value in initialization_dates)
    if truth.ndim != 4 or truth.shape[0] != len(dates):
        raise ValueError("truth_daily must have shape (sample, lead, lat, lon) matching initialization_dates.")
    allowed_years = {int(value) for value in train_years}
    if not allowed_years:
        raise ValueError("train_years must not be empty.")
    result: Dict[str, Dict[str, np.ndarray]] = {}
    means = window_means(truth, prediction_leads, windows)
    for window_name, leads in windows.items():
        center = int(np.floor(np.median(np.asarray(leads, dtype=np.float64)) + 0.5))
        center_months = np.asarray([(value + timedelta(days=center)).month for value in dates])
        is_training = np.asarray([value.year in allowed_years for value in dates], dtype=bool)
        by_quantile: Dict[str, np.ndarray] = {}
        for quantile_name, quantile in quantiles.items():
            if not 0.0 < float(quantile) < 1.0:
                raise ValueError(f"Invalid threshold quantile {quantile_name}={quantile}.")
            threshold = np.full((12,) + truth.shape[-2:], np.nan, dtype=np.float32)
            for month in range(1, 13):
                selected = is_training & (center_months == month)
                if np.any(selected):
                    threshold[month - 1] = np.nanquantile(means[window_name][selected], quantile, axis=0)
            by_quantile[quantile_name] = threshold
        result[window_name] = by_quantile
    return result


def _broadcast_weights(lat, shape, mask=None) -> Tuple[np.ndarray, np.ndarray]:
    if len(shape) < 2:
        raise ValueError("Metric arrays require latitude and longitude dimensions.")
    lat_weight = np.asarray(area_weights(lat), dtype=np.float64)
    if shape[-2] != lat_weight.size:
        raise ValueError("Latitude length does not match metric arrays.")
    weights = np.broadcast_to(lat_weight.reshape((1,) * (len(shape) - 2) + (lat_weight.size, 1)), shape).copy()
    valid = np.ones(shape, dtype=bool)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), shape)
    return weights, valid


def weighted_mean(values, lat, mask=None) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights, valid = _broadcast_weights(lat, values.shape, mask)
    valid &= np.isfinite(values)
    total = float(np.sum(weights[valid]))
    return float(np.sum(values[valid] * weights[valid]) / total) if total > 0.0 else float("nan")


def weighted_tac(forecast, truth, lat, mask=None) -> float:
    """Return weighted anomaly correlation over samples and space."""
    x = np.asarray(forecast, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Forecast and truth shapes must match.")
    weights, valid = _broadcast_weights(lat, x.shape, mask)
    valid &= np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return float("nan")
    w = weights[valid]
    w /= np.sum(w)
    xv, yv = x[valid], y[valid]
    xa, ya = xv - np.sum(w * xv), yv - np.sum(w * yv)
    denominator = np.sqrt(np.sum(w * xa * xa) * np.sum(w * ya * ya))
    return float(np.sum(w * xa * ya) / denominator) if denominator > 0.0 else float("nan")


def mse_skill(forecast, truth, climatology, lat, mask=None) -> Tuple[float, float]:
    """Return forecast MSE and skill relative to supplied climatology."""
    mse = weighted_mean((np.asarray(forecast) - np.asarray(truth)) ** 2, lat, mask)
    reference = weighted_mean((np.asarray(climatology) - np.asarray(truth)) ** 2, lat, mask)
    skill = 1.0 - mse / reference if np.isfinite(reference) and reference > 0.0 else float("nan")
    return mse, float(skill)


def gaussian_crps(mean, sigma, truth) -> np.ndarray:
    """Analytic Gaussian CRPS with positive-sigma validation."""
    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if np.any(sigma <= 0.0):
        raise ValueError("Gaussian sigma must be strictly positive.")
    z = (truth - mean) / sigma
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) - 1.0 / np.sqrt(np.pi))


def ensemble_crps(members, truth, member_axis: int = 0) -> np.ndarray:
    """Finite-ensemble CRPS used for the climatological and ENS references."""
    ensemble = np.moveaxis(np.asarray(members, dtype=np.float64), member_axis, 0)
    observation = np.asarray(truth, dtype=np.float64)
    first = np.mean(np.abs(ensemble - observation), axis=0)
    second = 0.5 * np.mean(np.abs(ensemble[:, None] - ensemble[None, :]), axis=(0, 1))
    return first - second


def brier_metrics(probability, observed, lat, mask=None, reference_probability=None) -> Tuple[float, float]:
    """Return area-weighted Brier score and Brier skill score."""
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    score = weighted_mean((probability - observed) ** 2, lat, mask)
    if reference_probability is None:
        base_rate = weighted_mean(observed, lat, mask)
        reference_probability = np.full_like(observed, base_rate)
    reference = weighted_mean((np.asarray(reference_probability) - observed) ** 2, lat, mask)
    skill = 1.0 - score / reference if np.isfinite(reference) and reference > 0.0 else float("nan")
    return score, float(skill)


def weighted_roc_auc(probability, observed, lat, mask=None) -> float:
    """Return weighted ROC AUC using the positive-negative ranking identity."""
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    weights, valid = _broadcast_weights(lat, probability.shape, mask)
    valid &= np.isfinite(probability)
    p, y, w = probability[valid], observed[valid], weights[valid]
    positive_weight = float(np.sum(w[y]))
    negative_weight = float(np.sum(w[~y]))
    denominator = positive_weight * negative_weight
    if denominator <= 0.0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]
    favorable = 0.0
    negative_below = 0.0
    start = 0
    while start < p.size:
        end = start + 1
        while end < p.size and p[end] == p[start]:
            end += 1
        positive_tie = float(np.sum(w[start:end][y[start:end]]))
        negative_tie = float(np.sum(w[start:end][~y[start:end]]))
        favorable += positive_tie * (negative_below + 0.5 * negative_tie)
        negative_below += negative_tie
        start = end
    return float(favorable / denominator)


def reliability_curve(probability, observed, lat, mask=None, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Return weighted forecast probability, observed frequency, and mass per bin."""
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    weights, valid = _broadcast_weights(lat, probability.shape, mask)
    valid &= np.isfinite(probability) & np.isfinite(observed)
    p, y, w = probability[valid], observed[valid], weights[valid]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    indices = np.clip(np.digitize(p, edges[1:-1]), 0, int(n_bins) - 1)
    predicted = np.full(int(n_bins), np.nan)
    frequency = np.full(int(n_bins), np.nan)
    mass = np.zeros(int(n_bins))
    for index in range(int(n_bins)):
        selected = indices == index
        if np.any(selected):
            mass[index] = np.sum(w[selected])
            predicted[index] = np.average(p[selected], weights=w[selected])
            frequency[index] = np.average(y[selected], weights=w[selected])
    if np.sum(mass) > 0.0:
        mass /= np.sum(mass)
    return {"bin_edges": edges, "forecast_probability": predicted, "observed_frequency": frequency, "weight": mass}


def evaluate_global_windows(
    forecast_daily_mean,
    forecast_daily_sigma,
    truth_daily,
    climatology_daily,
    initialization_dates: Sequence[object],
    lat,
    land_mask,
    thresholds: Mapping[str, Mapping[str, np.ndarray]],
    *,
    lon=None,
    prediction_leads: Sequence[int] = W34_LEADS,
    climatology_ensemble: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Evaluate all global windows under area-weighted NH-land/MJJAS semantics."""
    forecast = window_means(forecast_daily_mean, prediction_leads)
    truth = window_means(truth_daily, prediction_leads)
    climatology = window_means(climatology_daily, prediction_leads)
    sigma_daily = np.asarray(forecast_daily_sigma, dtype=np.float64)
    sigma = {
        name: np.sqrt(np.sum(np.take(sigma_daily, _lead_indices(prediction_leads, leads), axis=1) ** 2, axis=1)) / len(leads)
        for name, leads in GLOBAL_WINDOWS.items()
    }
    dates = tuple(_as_date(value) for value in initialization_dates)
    if len(dates) != np.asarray(truth_daily).shape[0]:
        raise ValueError("Initialization dates do not match the sample dimension.")
    results: Dict[str, Dict[str, object]] = {}
    for name, leads in GLOBAL_WINDOWS.items():
        sample_mask = np.stack([nh_land_mjjas_mask(lat, land_mask, value, leads) for value in dates])
        mean, obs, scale, clim = forecast[name], truth[name], sigma[name], climatology[name]
        mse, mse_ss = mse_skill(mean, obs, clim, lat, sample_mask)
        crps = gaussian_crps(mean, scale, obs)
        crps_score = weighted_mean(crps, lat, sample_mask)
        crpss = float("nan")
        if climatology_ensemble is not None and name in climatology_ensemble:
            reference = ensemble_crps(climatology_ensemble[name], obs, member_axis=0)
            reference_score = weighted_mean(reference, lat, sample_mask)
            if np.isfinite(reference_score) and reference_score > 0.0:
                crpss = 1.0 - crps_score / reference_score
        row: Dict[str, object] = {
            "window": name,
            "leads": tuple(int(value) for value in leads),
            "valid_initializations": int(np.sum(np.any(sample_mask, axis=(1, 2)))),
            "tac": weighted_tac(mean, obs, lat, sample_mask),
            "mse": mse,
            "mse_skill_vs_climatology": mse_ss,
            "crps": crps_score,
            "crpss_vs_climatological_ensemble": crpss,
            "spread": weighted_mean(scale, lat, sample_mask),
            "rmse": float(np.sqrt(mse)),
            "spread_error_ratio": weighted_mean(scale, lat, sample_mask) / np.sqrt(mse) if mse > 0.0 else float("nan"),
        }
        center = int(np.floor(np.median(np.asarray(leads, dtype=np.float64)) + 0.5))
        month = np.asarray([(value + timedelta(days=center)).month for value in dates])
        event_fields: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for threshold_name in ("upper_tercile", "q95"):
            threshold_cube = np.asarray(thresholds[name][threshold_name])
            threshold = np.stack([threshold_cube[value - 1] for value in month])
            event = obs > threshold
            probability = 1.0 - ndtr((threshold - mean) / scale)
            event_fields[threshold_name] = (probability, event)
            brier, bss = brier_metrics(probability, event, lat, sample_mask)
            row[f"{threshold_name}_brier"] = brier
            row[f"{threshold_name}_bss"] = bss
            row[f"{threshold_name}_roc_auc"] = weighted_roc_auc(probability, event, lat, sample_mask)
            row[f"{threshold_name}_reliability"] = reliability_curve(probability, event, lat, sample_mask)
            if threshold_name == "q95":
                tail_mask = sample_mask & event
                row["q95_tail_mae"] = weighted_mean(np.abs(mean - obs), lat, tail_mask)
                for label, z_value in TAIL_Z.items():
                    row[f"q95_tail_containment_{label}"] = weighted_mean(
                        obs <= mean + z_value * scale, lat, tail_mask
                    )
        breakdowns = []
        longitude = (
            np.arange(np.asarray(land_mask).shape[1]) * (360.0 / np.asarray(land_mask).shape[1])
            if lon is None else np.asarray(lon)
        )
        region_table = {"nh_land": np.asarray(land_mask, dtype=bool) & (np.asarray(lat)[:, None] >= 0.0)}
        region_table.update(region_masks(lat, longitude, land_mask))
        for region_name, region_mask in region_table.items():
            for selected_month in MJJAS_MONTHS:
                breakdown_mask = sample_mask & region_mask[None] & (month[:, None, None] == selected_month)
                if not np.any(breakdown_mask):
                    continue
                region_mse, region_skill = mse_skill(mean, obs, clim, lat, breakdown_mask)
                probability_q95, event_q95 = event_fields["q95"]
                region_brier, region_bss = brier_metrics(probability_q95, event_q95, lat, breakdown_mask)
                breakdowns.append({
                    "region": region_name,
                    "month": int(selected_month),
                    "tac": weighted_tac(mean, obs, lat, breakdown_mask),
                    "mse": region_mse,
                    "mse_skill_vs_climatology": region_skill,
                    "crps": weighted_mean(crps, lat, breakdown_mask),
                    "q95_brier": region_brier,
                    "q95_bss": region_bss,
                })
        row["monthly_region_breakdowns"] = breakdowns
        results[name] = row
    return results


def year_block_bootstrap(values, years, *, repetitions: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap a scalar sample statistic by resampling whole years."""
    array = np.asarray(values, dtype=np.float64)
    year = np.asarray(years, dtype=np.int64)
    if array.shape[0] != year.size:
        raise ValueError("The leading value dimension must match years.")
    unique = np.unique(year)
    if unique.size == 0:
        raise ValueError("No years supplied for bootstrap.")
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        blocks = [array[year == selected] for selected in sampled]
        output[index] = np.nanmean(np.concatenate(blocks, axis=0))
    return output
