"""Synthetic recovery and counting tests for Phase 6 novelty analyses."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genextreme

from novelty_analyses import (
    assert_fold_safe_exports,
    fit_bayesian_gev,
    ipcc_likelihood_category,
    joint_event_probabilities,
    storyline_product,
    tail_shape_analysis,
)


def test_synthetic_gev_draws_recover_known_parameters():
    rng = np.random.default_rng(12)
    samples = genextreme.rvs(c=-0.10, loc=2.0, scale=1.5, size=800, random_state=rng)
    fit = fit_bayesian_gev(samples, draws=300, burn=300, seed=8)
    assert fit.method == "gev"
    assert fit.location == pytest.approx(2.0, abs=0.25)
    assert fit.scale == pytest.approx(1.5, abs=0.25)
    assert fit.shape == pytest.approx(0.10, abs=0.10)
    assert 0.05 < fit.acceptance_rate < 0.8
    quantiles = fit.posterior_quantile(0.999)
    assert quantiles.shape == (300,)
    assert np.all(np.isfinite(quantiles))


def test_eleven_member_ens_uses_reported_gumbel_fallback():
    samples = np.asarray([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0])
    fit = fit_bayesian_gev(samples, draws=80, burn=80, seed=2)
    assert fit.method == "gumbel_fallback"
    assert fit.shape == 0.0
    assert np.isfinite(fit.sensitivity_shape)


def test_joint_spatial_event_counts_coherent_members_not_marginals():
    threshold = np.zeros((1, 4), dtype=np.float32)
    region = np.ones((1, 4), dtype=bool)
    cfm = np.asarray([
        [[1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, -1.0, -1.0]],
        [[-1.0, -1.0, -1.0, -1.0]],
    ])
    ens = np.asarray([
        [[1.0, -1.0, -1.0, -1.0]],
        [[-1.0, -1.0, -1.0, -1.0]],
    ])
    result = joint_event_probabilities(
        cfm,
        ens,
        threshold,
        region,
        np.asarray([0.0]),
        independence_members=2000,
        seed=4,
    )
    assert result["area_ge_10pct"]["cfm"] == pytest.approx(2.0 / 3.0)
    assert result["area_ge_25pct"]["ens"] == pytest.approx(0.5)
    assert result["area_ge_50pct"]["ens"] == 0.0
    assert 0.0 <= result["area_ge_50pct"]["independent_marginals"] <= 1.0


def test_gaussian_cfm_tail_divergence_and_storyline():
    mean = np.zeros((1, 2), dtype=np.float32)
    sigma = np.ones((1, 2), dtype=np.float32)
    samples = np.asarray([
        [[0.0, 0.0]],
        [[2.0, 0.0]],
        [[2.0, 2.0]],
        [[2.0, 2.0]],
    ])
    result = tail_shape_analysis(mean, sigma, samples, {"q95": np.ones((1, 2))})
    np.testing.assert_allclose(result["q95"]["cfm_probability"], [[0.75, 0.5]])
    assert np.all(result["q95"]["cfm_minus_gaussian"] > 0.0)

    storyline = storyline_product(samples, np.asarray([[1.0, 1.0]]), allow_small_fixture=True)
    assert storyline["storyline_quantile"].shape == (1, 2)
    assert storyline["observed_contained"].tolist() == [[1, 1]]


def test_fold_roles_and_ipcc_categories_are_explicit():
    assert_fold_safe_exports((1979, 1980), (1982,), (1981,))
    with pytest.raises(RuntimeError, match="train/test overlap"):
        assert_fold_safe_exports((1979, 1980), (1980,))
    assert ipcc_likelihood_category(0.995) == "virtually_certain"
    assert ipcc_likelihood_category(0.5) == "about_as_likely_as_not"
    assert ipcc_likelihood_category(0.005) == "exceptionally_unlikely"
