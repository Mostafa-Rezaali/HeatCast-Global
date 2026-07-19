"""Data-free tests for weighted global evaluation, stitching, and export."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from export_w34_stack_netcdf import target_lat_lon, write_global_hindcast_netcdf
from global_evaluation import (
    GLOBAL_WINDOWS,
    brier_metrics,
    build_fold_window_thresholds,
    ensemble_crps,
    evaluate_global_windows,
    local_warm_season_mask,
    nh_land_mjjas_mask,
    region_masks,
    weighted_mean,
    weighted_roc_auc,
    window_means,
)
from stitch_exceedance_folds import summarize_global_metric_rows, write_global_metric_tables


def test_week3_week4_w34_window_means_and_mjjas_boundaries():
    daily = np.arange(14, dtype=np.float32).reshape(1, 14, 1, 1)
    means = window_means(daily)
    assert means["week3"].item() == pytest.approx(3.0)
    assert means["week4"].item() == pytest.approx(10.0)
    assert means["w34"].item() == pytest.approx(6.5)

    lat = np.asarray([-30.0, 0.0, 45.0])
    land = np.ones((3, 2), dtype=bool)
    valid = nh_land_mjjas_mask(lat, land, date(2020, 5, 1))
    assert not valid[0].any()
    assert valid[1:].all()
    assert not nh_land_mjjas_mask(lat, land, date(2020, 9, 10)).any()


def test_local_warm_season_never_applies_nh_calendar_globally():
    months = np.asarray([[[5, 6, 7], [12, 1, 2]]], dtype=np.int16)
    land = np.ones((1, 2), dtype=bool)
    may = local_warm_season_mask(months, land, date(2020, 5, 1), (15, 16))
    december = local_warm_season_mask(months, land, date(2020, 12, 1), (15, 16))
    assert may.tolist() == [[True, False]]
    assert december.tolist() == [[False, True]]


def test_fold_thresholds_ignore_poisoned_test_year():
    dates = [date(2000, 5, 1), date(2001, 5, 1), date(2002, 5, 1)]
    truth = np.stack([
        np.full((14, 2, 2), 1.0, dtype=np.float32),
        np.full((14, 2, 2), 3.0, dtype=np.float32),
        np.full((14, 2, 2), 5.0, dtype=np.float32),
    ])
    original = build_fold_window_thresholds(truth, dates, (2000, 2001))
    poisoned = truth.copy()
    poisoned[2] = 1.0e9
    changed = build_fold_window_thresholds(poisoned, dates, (2000, 2001))
    for window in GLOBAL_WINDOWS:
        for threshold in ("upper_tercile", "q95"):
            np.testing.assert_array_equal(original[window][threshold], changed[window][threshold])


def test_area_weighting_toy_crps_brier_and_auc():
    members = np.asarray([0.0, 1.0, 2.0])
    assert ensemble_crps(members, np.asarray(1.0)).item() == pytest.approx(2.0 / 9.0)

    probability = np.asarray([[[2.0 / 3.0]]])
    observed = np.ones_like(probability)
    brier, _ = brier_metrics(probability, observed, np.asarray([0.0]), reference_probability=np.zeros_like(probability))
    assert brier == pytest.approx(1.0 / 9.0)

    p = np.asarray([[[0.1]], [[0.9]], [[0.5]], [[0.5]]])
    y = np.asarray([[[0]], [[1]], [[0]], [[1]]], dtype=bool)
    assert weighted_roc_auc(p, y, np.asarray([0.0])) == pytest.approx(0.875)

    values = np.asarray([[1.0], [9.0]])
    assert weighted_mean(values, np.asarray([0.0, 80.0])) < 3.0


def test_global_training_reductions_use_cosine_latitude_weights():
    import cfm_mesh_train as cfm
    from mode_dispatch import gaussian_crps

    original_domain = cfm.Config.DOMAIN
    try:
        cfm.Config.DOMAIN = "global"
        prediction = torch.tensor([[[[10.0], [1.0], [10.0]]]])
        truth = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction)
        assert cfm.masked_mse_loss(prediction, truth, mask).item() == pytest.approx(1.0, abs=1e-5)

        raw = SimpleNamespace(
            _mesh=SimpleNamespace(global_domain=True, grid_lat=np.asarray([90.0, 0.0, -90.0]))
        )
        model = SimpleNamespace(module=raw)
        sigma = torch.ones_like(prediction)
        weighted = gaussian_crps(prediction, sigma, truth, mask, model=model)
        unweighted = gaussian_crps(prediction, sigma, truth, mask)
        assert weighted.item() < unweighted.item()
    finally:
        cfm.Config.DOMAIN = original_domain


def test_global_regions_and_all_window_evaluation():
    lat = np.asarray([-30.0, 30.0, 65.0])
    lon = np.asarray([0.0, 80.0, 120.0, 250.0, 355.0])
    land = np.ones((3, 5), dtype=bool)
    masks = region_masks(lat, lon, land)
    assert masks["conus"][1, 3]
    assert masks["europe"][2, 4]
    assert masks["south_asia"][1, 1]
    assert masks["east_asia"][1, 2]

    shape = (2, 14, 3, 5)
    truth = np.zeros(shape, dtype=np.float32)
    truth[0, :, 1:, ::2] = 1.0
    truth[1, :, 1:, 1::2] = 2.0
    forecast = truth * 0.8 + 0.1
    sigma = np.full(shape, 0.5, dtype=np.float32)
    climatology = np.zeros(shape, dtype=np.float32)
    thresholds = {
        window: {
            "upper_tercile": np.zeros((12, 3, 5), dtype=np.float32),
            "q95": np.full((12, 3, 5), 0.5, dtype=np.float32),
        }
        for window in GLOBAL_WINDOWS
    }
    results = evaluate_global_windows(
        forecast,
        sigma,
        truth,
        climatology,
        (date(2000, 5, 1), date(2001, 6, 1)),
        lat,
        land,
        thresholds,
    )
    assert set(results) == set(GLOBAL_WINDOWS)
    for row in results.values():
        assert row["valid_initializations"] == 2
        assert np.isfinite(row["tac"])
        assert np.isfinite(row["q95_brier"])
        assert "q95_tail_containment_q999" in row
        assert row["monthly_region_breakdowns"]


def test_stitch_global_fold_rows_and_year_block_bootstrap(tmp_path):
    rows = []
    for window in GLOBAL_WINDOWS:
        rows.extend([
            {"window": window, "fold": 0, "year": 2000, "tac": 0.2, "crps": 1.0},
            {"window": window, "fold": 1, "year": 2001, "tac": 0.4, "crps": 0.8},
        ])
    summary = summarize_global_metric_rows(rows, repetitions=30, seed=4)
    assert len(summary) == 6
    assert {row["window"] for row in summary} == set(GLOBAL_WINDOWS)
    paths = write_global_metric_tables(rows, tmp_path, repetitions=20)
    assert len(paths) == 4
    assert all(path.is_file() for path in paths)

    bad = list(rows)
    bad.append({"window": "w34", "fold": 2, "year": 2000, "tac": 0.5})
    with pytest.raises(RuntimeError, match="owned by folds"):
        summarize_global_metric_rows(bad, repetitions=2)


def test_global_grid_export_is_shape_parameterized(tmp_path):
    class Config:
        DOMAIN = "global"
        RESOLUTION = "1.5deg"

    lat, lon, lat2d, lon2d, source = target_lat_lon(Config, (121, 240))
    assert lat.shape == (121,)
    assert lon.shape == (240,)
    assert lat2d.shape == lon2d.shape == (121, 240)
    assert source == "configured_global_grid:1.5deg"

    small_lat = np.asarray([45.0, 0.0, -45.0], dtype=np.float32)
    small_lon = np.asarray([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    shape = (2, 14, 3, 4)
    mean = np.ones(shape, dtype=np.float32)
    sigma = np.full(shape, 0.5, dtype=np.float32)
    truth = np.full(shape, 1.5, dtype=np.float32)
    probability = {f"{window}_q95": np.full((2, 3, 4), 0.25, dtype=np.float32) for window in GLOBAL_WINDOWS}
    output = tmp_path / "global_hindcast.nc"
    write_global_hindcast_netcdf(
        output,
        (20000501, 20000601),
        small_lat,
        small_lon,
        mean,
        sigma,
        truth,
        exceedance_probabilities=probability,
    )
    from netCDF4 import Dataset

    with Dataset(output) as dataset:
        assert dataset.variables["week3_anomaly_mean"].shape == (3, 4, 2)
        assert dataset.variables["week4_anomaly_sigma"].shape == (3, 4, 2)
        assert dataset.variables["w34_observed_anomaly"].shape == (3, 4, 2)
        assert dataset.variables["prob_w34_q95"].shape == (3, 4, 2)
