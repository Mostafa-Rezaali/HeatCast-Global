#!/usr/bin/env python3
"""Fold-safe W34 tail-envelope, tail-shape, joint-event, and storyline analyses.

The GEV analysis adapts the conditioned-ensemble envelope question of Risser
et al. (2026, arXiv:2604.09754) to matched W34 hindcasts.  It does not estimate
climatological return periods.  All entry points operate on exported held-out
hindcasts, assert fold-role disjointness, and support whole-year bootstrap
summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import genextreme

from global_evaluation import year_block_bootstrap
from spatial_weights import area_weights


DEFAULT_TAIL_PROBABILITIES = (0.95, 0.99, 0.999)
DEFAULT_JOINT_AREA_FRACTIONS = (0.10, 0.25, 0.50)
IPCC_LIKELIHOOD_BOUNDS = (
    (0.99, "virtually_certain"),
    (0.90, "very_likely"),
    (0.66, "likely"),
    (0.33, "about_as_likely_as_not"),
    (0.10, "unlikely"),
    (0.01, "very_unlikely"),
    (0.00, "exceptionally_unlikely"),
)


def assert_fold_safe_exports(train_years, test_years, calibration_years=()) -> None:
    """Reject any overlap among model training, calibration, and test years."""
    train = {int(value) for value in train_years}
    test = {int(value) for value in test_years}
    calibration = {int(value) for value in calibration_years}
    if train & test:
        raise RuntimeError(f"Fold leakage: train/test overlap {sorted(train & test)}.")
    if train & calibration:
        raise RuntimeError(f"Fold leakage: train/calibration overlap {sorted(train & calibration)}.")
    if calibration & test:
        raise RuntimeError(f"Fold leakage: calibration/test overlap {sorted(calibration & test)}.")


def gev_quantile(location, scale, shape, probability: float) -> np.ndarray:
    """Return a GEV quantile using the positive-shape-heavy-tail convention."""
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("GEV probability must lie strictly within (0, 1).")
    location = np.asarray(location, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.float64)
    if np.any(scale <= 0.0):
        raise ValueError("GEV scale must be positive.")
    transformed = -np.log(probability)
    gumbel = location - scale * np.log(transformed)
    safe_shape = np.where(np.abs(shape) < 1e-7, 1.0, shape)
    general = location + scale * np.expm1(-shape * np.log(transformed)) / safe_shape
    return np.where(np.abs(shape) < 1e-7, gumbel, general)


def _gev_log_likelihood(parameters, samples, *, gumbel: bool) -> float:
    location = float(parameters[0])
    log_scale = float(parameters[1])
    scale = np.exp(log_scale)
    shape = 0.0 if gumbel else float(parameters[2])
    standardized = (samples - location) / scale
    if abs(shape) < 1e-7:
        logpdf = -log_scale - standardized - np.exp(-standardized)
    else:
        support = 1.0 + shape * standardized
        if np.any(support <= 0.0):
            return -np.inf
        logpdf = -log_scale - (1.0 / shape + 1.0) * np.log(support) - support ** (-1.0 / shape)
    # Weak regularization stabilizes 11-member fits without overwhelming data.
    sample_scale = max(float(np.std(samples)), 1e-3)
    prior = -0.5 * ((location - float(np.mean(samples))) / (5.0 * sample_scale)) ** 2
    prior += -0.5 * ((log_scale - np.log(sample_scale)) / 1.5) ** 2
    if not gumbel:
        if not -0.5 < shape < 0.5:
            return -np.inf
        prior += -0.5 * (shape / 0.25) ** 2
    return float(np.sum(logpdf) + prior)


def _optimize_gev(samples: np.ndarray, *, gumbel: bool) -> np.ndarray:
    try:
        scipy_shape, scipy_loc, scipy_scale = genextreme.fit(samples)
        initial = [float(scipy_loc), float(np.log(max(scipy_scale, 1e-4)))]
        if not gumbel:
            initial.append(float(np.clip(-scipy_shape, -0.35, 0.35)))
    except Exception:
        initial = [float(np.mean(samples)), float(np.log(max(np.std(samples), 1e-3)))]
        if not gumbel:
            initial.append(0.0)
    bounds = [(None, None), (np.log(1e-5), np.log(max(np.ptp(samples) * 20.0, 1.0)))]
    if not gumbel:
        bounds.append((-0.49, 0.49))
    def objective(value):
        log_probability = _gev_log_likelihood(value, samples, gumbel=gumbel)
        return -log_probability if np.isfinite(log_probability) else 1e100

    result = minimize(
        objective,
        np.asarray(initial),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GEV optimization failed: {result.message}.")
    return np.asarray(result.x, dtype=np.float64)


def _metropolis_draws(
    samples: np.ndarray,
    initial: np.ndarray,
    *,
    gumbel: bool,
    draws: int,
    burn: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    sample_scale = max(float(np.std(samples)), 1e-3)
    proposal = np.asarray([0.06 * sample_scale, 0.045] + ([] if gumbel else [0.025]))
    current = np.asarray(initial, dtype=np.float64).copy()
    current_logp = _gev_log_likelihood(current, samples, gumbel=gumbel)
    kept = []
    accepted = 0
    total = int(burn) + int(draws)
    for index in range(total):
        candidate = current + rng.normal(scale=proposal)
        candidate_logp = _gev_log_likelihood(candidate, samples, gumbel=gumbel)
        if np.isfinite(candidate_logp) and np.log(rng.random()) < candidate_logp - current_logp:
            current, current_logp = candidate, candidate_logp
            accepted += 1
        if index >= int(burn):
            kept.append(current.copy())
    return np.asarray(kept), accepted / max(total, 1)


@dataclass(frozen=True)
class BayesianGEVFit:
    """Posterior draws and diagnostics for one cell's conditioned ENS tail."""

    location: float
    scale: float
    shape: float
    method: str
    posterior_draws: np.ndarray
    acceptance_rate: float
    sensitivity_shape: float
    sample_count: int

    def posterior_quantile(self, probability: float) -> np.ndarray:
        return gev_quantile(
            self.posterior_draws[:, 0],
            np.exp(self.posterior_draws[:, 1]),
            self.posterior_draws[:, 2],
            probability,
        )


def fit_bayesian_gev(
    samples,
    *,
    draws: int = 512,
    burn: int = 512,
    seed: int = 0,
    minimum_shape_samples: int = 30,
) -> BayesianGEVFit:
    """Fit a lightweight Bayesian GEV, falling back to Gumbel for small ENS.

    A full-shape MLE is still reported as ``sensitivity_shape`` when the
    operational fit falls back, making the small-ensemble assumption visible.
    """
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 6 or np.ptp(values) <= 1e-8:
        raise ValueError("GEV fitting needs at least six finite, non-constant samples.")
    full = _optimize_gev(values, gumbel=False)
    use_gumbel = values.size < int(minimum_shape_samples) or abs(float(full[2])) >= 0.45
    initial = _optimize_gev(values, gumbel=True) if use_gumbel else full
    posterior, acceptance = _metropolis_draws(
        values,
        initial,
        gumbel=use_gumbel,
        draws=int(draws),
        burn=int(burn),
        seed=int(seed),
    )
    if use_gumbel:
        posterior = np.column_stack([posterior, np.zeros(posterior.shape[0])])
    location = float(np.median(posterior[:, 0]))
    scale = float(np.median(np.exp(posterior[:, 1])))
    shape = float(np.median(posterior[:, 2]))
    return BayesianGEVFit(
        location=location,
        scale=scale,
        shape=shape,
        method="gumbel_fallback" if use_gumbel else "gev",
        posterior_draws=posterior.astype(np.float32),
        acceptance_rate=float(acceptance),
        sensitivity_shape=float(full[2]),
        sample_count=int(values.size),
    )


def ipcc_likelihood_category(probability: float) -> str:
    """Map a containment probability to explicit IPCC-style likelihood bins."""
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        return "undefined"
    for lower, label in IPCC_LIKELIHOOD_BOUNDS:
        if value >= lower:
            return label
    return "undefined"


def gev_envelope_analysis(
    ens_members,
    heatcast_tail_quantile,
    observed,
    land_mask,
    *,
    probability: float = 0.999,
    draws: int = 256,
    burn: int = 256,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Map posterior ENS-GEV containment and reverse observed-tail misses."""
    ens = np.asarray(ens_members, dtype=np.float64)
    heat = np.asarray(heatcast_tail_quantile, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if ens.ndim != 3 or ens.shape[1:] != land.shape or heat.shape != land.shape or truth.shape != land.shape:
        raise ValueError("Expected ENS=(member,lat,lon) and matching 2-D maps.")
    containment = np.full(land.shape, np.nan, dtype=np.float32)
    median_threshold = np.full(land.shape, np.nan, dtype=np.float32)
    gumbel_used = np.zeros(land.shape, dtype=np.uint8)
    reverse_miss = np.zeros(land.shape, dtype=np.uint8)
    category = np.full(land.shape, "undefined", dtype="U32")
    for cell_index, (row, column) in enumerate(np.argwhere(land)):
        try:
            fit = fit_bayesian_gev(
                ens[:, row, column],
                draws=draws,
                burn=burn,
                seed=int(seed) + cell_index,
            )
        except ValueError:
            continue
        posterior_threshold = fit.posterior_quantile(probability)
        containment[row, column] = np.mean(posterior_threshold >= heat[row, column])
        median_threshold[row, column] = np.median(posterior_threshold)
        gumbel_used[row, column] = int(fit.method == "gumbel_fallback")
        reverse_miss[row, column] = int(
            np.isfinite(truth[row, column])
            and truth[row, column] > median_threshold[row, column]
            and truth[row, column] <= heat[row, column]
        )
        category[row, column] = ipcc_likelihood_category(containment[row, column])
    return {
        "ens_contains_heatcast_probability": containment,
        "ens_gev_threshold_median": median_threshold,
        "likelihood_category": category,
        "gumbel_fallback": gumbel_used,
        "heatcast_contains_observed_ens_miss": reverse_miss,
    }


def tail_shape_analysis(
    gaussian_mean,
    gaussian_sigma,
    cfm_samples,
    thresholds: Mapping[str, np.ndarray],
    *,
    mask=None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare analytic Gaussian and empirical CFM exceedance probabilities."""
    mean = np.asarray(gaussian_mean, dtype=np.float64)
    sigma = np.asarray(gaussian_sigma, dtype=np.float64)
    samples = np.asarray(cfm_samples, dtype=np.float64)
    if mean.shape != sigma.shape or samples.ndim != mean.ndim + 1 or samples.shape[1:] != mean.shape:
        raise ValueError("Expected mean/sigma maps and CFM=(member, ...same shape).")
    if np.any(sigma <= 0.0):
        raise ValueError("Gaussian sigma must be positive.")
    selected = np.ones(mean.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for label, threshold in thresholds.items():
        threshold = np.asarray(threshold, dtype=np.float64)
        gaussian = 1.0 - ndtr((threshold - mean) / sigma)
        empirical = np.mean(samples > threshold[None], axis=0)
        divergence = empirical - gaussian
        for array in (gaussian, empirical, divergence):
            array[~selected] = np.nan
        output[str(label)] = {
            "gaussian_probability": gaussian.astype(np.float32),
            "cfm_probability": empirical.astype(np.float32),
            "cfm_minus_gaussian": divergence.astype(np.float32),
        }
    return output


def _member_area_fraction(samples, threshold, region_mask, lat) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    region = np.asarray(region_mask, dtype=bool)
    if samples.ndim != 3 or threshold.shape != region.shape or samples.shape[1:] != region.shape:
        raise ValueError("Expected samples=(member,lat,lon) and matching threshold/region maps.")
    weights = np.broadcast_to(area_weights(lat)[:, None], region.shape)
    denominator = float(np.sum(weights[region]))
    if denominator <= 0.0:
        raise ValueError("Region contains no positive-area cells.")
    exceedance = (samples > threshold[None]) & region[None]
    return np.sum(exceedance * weights[None], axis=(1, 2)) / denominator


def joint_event_probabilities(
    cfm_samples,
    ens_members,
    threshold,
    region_mask,
    lat,
    *,
    area_fractions: Sequence[float] = DEFAULT_JOINT_AREA_FRACTIONS,
    independence_members: int = 10000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Compare coherent CFM/ENS regional events with independent marginals."""
    cfm = np.asarray(cfm_samples, dtype=np.float64)
    ens = np.asarray(ens_members, dtype=np.float64)
    cfm_fraction = _member_area_fraction(cfm, threshold, region_mask, lat)
    ens_fraction = _member_area_fraction(ens, threshold, region_mask, lat)
    marginal = np.mean(cfm > np.asarray(threshold)[None], axis=0)
    rng = np.random.default_rng(int(seed))
    independent = np.empty(int(independence_members), dtype=np.float64)
    weights = np.broadcast_to(area_weights(lat)[:, None], np.asarray(region_mask).shape)
    selected_probability = marginal[np.asarray(region_mask, dtype=bool)]
    selected_weights = weights[np.asarray(region_mask, dtype=bool)]
    selected_weights /= np.sum(selected_weights)
    block = 256
    for start in range(0, independent.size, block):
        stop = min(start + block, independent.size)
        events = rng.random((stop - start, selected_probability.size)) < selected_probability[None]
        independent[start:stop] = events @ selected_weights
    return {
        f"area_ge_{int(round(float(fraction) * 100))}pct": {
            "cfm": float(np.mean(cfm_fraction >= float(fraction))),
            "ens": float(np.mean(ens_fraction >= float(fraction))),
            "independent_marginals": float(np.mean(independent >= float(fraction))),
        }
        for fraction in area_fractions
    }


def storyline_product(
    cfm_samples,
    observed,
    *,
    probability: float = 0.999,
    minimum_members: int = 1000,
    allow_small_fixture: bool = False,
) -> Dict[str, np.ndarray]:
    """Return the W34 plausible-worst-case map for a fixed initialization."""
    samples = np.asarray(cfm_samples, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != truth.shape:
        raise ValueError("Expected CFM samples=(member,lat,lon) and a matching observed map.")
    if samples.shape[0] < int(minimum_members) and not allow_small_fixture:
        raise ValueError(f"Storyline requires at least {minimum_members} CFM members, got {samples.shape[0]}.")
    storyline = np.quantile(samples, float(probability), axis=0)
    return {
        "storyline_quantile": storyline.astype(np.float32),
        "observed": truth.astype(np.float32),
        "storyline_minus_observed": (storyline - truth).astype(np.float32),
        "observed_contained": (truth <= storyline).astype(np.uint8),
    }


def bootstrap_tail_summary(values, years, *, repetitions: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Return a pooled mean and whole-year 95% interval for a tail diagnostic."""
    draws = year_block_bootstrap(values, years, repetitions=repetitions, seed=seed)
    return {
        "value": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(draws, 0.025)),
        "ci_high": float(np.nanquantile(draws, 0.975)),
        "independent_year_blocks": int(np.unique(years).size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=("gev", "tail_shape", "joint", "storyline"), required=True)
    parser.add_argument("--input", required=True, help="Fold-safe exported NPZ; no training data are read here.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gev_draws", type=int, default=256)
    parser.add_argument("--gev_burn", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Run one analysis against an explicit, fold-safe exported NPZ schema."""
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        train_years = np.atleast_1d(data["train_years"]).astype(int)
        test_years = np.atleast_1d(data["test_years"]).astype(int)
        calibration_years = np.atleast_1d(data["calibration_years"]).astype(int) if "calibration_years" in data else ()
        assert_fold_safe_exports(train_years, test_years, calibration_years)
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if args.analysis == "gev":
        result = gev_envelope_analysis(
            arrays["ens_members"],
            arrays["heatcast_tail_quantile"],
            arrays["observed"],
            arrays["land_mask"],
            draws=args.gev_draws,
            burn=args.gev_burn,
            seed=args.seed,
        )
    elif args.analysis == "tail_shape":
        thresholds = {key: arrays[key] for key in ("q95", "q99", "q999")}
        nested = tail_shape_analysis(
            arrays["gaussian_mean"], arrays["gaussian_sigma"], arrays["cfm_samples"], thresholds,
            mask=arrays.get("land_mask"),
        )
        result = {f"{threshold}__{name}": value for threshold, rows in nested.items() for name, value in rows.items()}
    elif args.analysis == "joint":
        summary = joint_event_probabilities(
            arrays["cfm_samples"], arrays["ens_members"], arrays["q95"], arrays["region_mask"], arrays["lat"],
            seed=args.seed,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return
    else:
        result = storyline_product(arrays["cfm_samples"], arrays["observed"])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)


if __name__ == "__main__":
    main()
