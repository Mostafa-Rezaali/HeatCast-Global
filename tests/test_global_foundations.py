"""Data-free contracts for global config, initialization dates, and area weights."""

from datetime import date

import numpy as np
import pytest
import torch

import cfm_mesh_train as cfm
from init_calendar import MJJAS_MONTHS, W34_LEADS, WEEK3_LEADS, WEEK4_LEADS, mjjas_mon_thu, valid_dates
from spatial_weights import area_weights, weighted_spatial_mean


def test_global_config_is_default_and_preserves_w34_distributional_contract():
    assert cfm.Config.DOMAIN == "global"
    assert cfm.Config.RESOLUTION == "1.5deg"
    assert cfm.Config.IMAGE_SIZE == (121, 240)
    assert cfm.Config.MESH_LEVEL == 5
    assert cfm.Config.TARGET_MODE == "climatology_anomaly"
    assert cfm.Config.PREDICT_PERSISTENCE_RESIDUAL is False
    assert cfm.Config.PREDICTION_LEADS == W34_LEADS
    assert cfm.Config.DISTRIBUTIONAL_HEAD is True
    assert cfm.Config.CRPS_LOSS is True
    assert cfm.Config.NUM_GLOBAL_CHANNELS == 0
    assert cfm.Config.CONDITION_DIM == 8


def test_domain_switch_retains_conus_path_and_semantics():
    keys = (
        "DOMAIN", "RESOLUTION", "TARGET_MODE", "MESH_LEVEL", "MESH_REFINEMENT_LEVEL",
        "IMAGE_SIZE", "TRAINING_DATA_PATH", "TARGET_VARIABLE_CANDIDATES", "OUTPUT_DIR",
        "GLOBAL_DATA_PATH", "USE_EXTENDED_GLOBAL_FIELDS", "REQUIRE_EXTENDED_GLOBAL_FIELDS",
        "NUM_GLOBAL_CHANNELS", "TRAIN_ON_CLIMATOLOGY_ANOMALIES",
        "PREDICT_PERSISTENCE_RESIDUAL", "NUM_SPATIAL_CONDITIONS", "CONDITION_DIM",
        "MULTI_LEAD_TUBE", "PREDICTION_LEADS", "TUBE_LOSS_DAILY_WEIGHT",
        "TUBE_LOSS_CENTER_WEIGHT", "TUBE_LOSS_WEEKLY_WEIGHT", "DISTRIBUTIONAL_HEAD",
        "CRPS_LOSS", "IMAGE_CHANNELS", "EARLY_STOP_METRIC", "CHECKPOINT_DIR", "PLOTS_DIR",
        "OUTPUT_NC_FILE", "HINDCAST_STATS_DIR", "HINDCAST_PAPER_DIR", "TRAINING_METRICS_DIR",
        "EXTENDED_GLOBAL_VARIABLES_PATH",
    )
    original = {key: getattr(cfm.Config, key) for key in keys}
    try:
        cfm.configure_domain("conus", "1.5deg", "zscore_persistence")
        assert cfm.Config.DOMAIN == "conus"
        assert cfm.Config.IMAGE_SIZE == (621, 1405)
        assert cfm.Config.PREDICT_PERSISTENCE_RESIDUAL is True
        assert cfm.Config.USE_EXTENDED_GLOBAL_FIELDS is True
        assert cfm.Config.NUM_GLOBAL_CHANNELS == 118
        assert cfm.Config.CONDITION_DIM == 5
        assert cfm.Config.MULTI_LEAD_TUBE is False
        assert cfm.Config.PREDICTION_LEADS == tuple(range(12, 19))
        assert cfm.Config.DISTRIBUTIONAL_HEAD is False
        assert cfm.Config.CRPS_LOSS is False
    finally:
        for key, value in original.items():
            setattr(cfm.Config, key, value)


def test_week_windows_and_year_boundary_math():
    assert WEEK3_LEADS == tuple(range(15, 22))
    assert WEEK4_LEADS == tuple(range(22, 29))
    assert W34_LEADS == tuple(range(15, 29))
    dates = valid_dates(date(2023, 12, 20), (15, 21, 22, 28))
    assert dates == (date(2024, 1, 4), date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 17))


def test_shared_mjjas_calendar_requires_full_w34_window():
    filtered = tuple(mjjas_mon_thu(2024))
    legacy = tuple(mjjas_mon_thu(2024, require_full_w34=False))
    assert filtered
    assert len(filtered) < len(legacy)
    assert all(day.weekday() in (0, 3) for day in filtered)
    assert all(valid.month in MJJAS_MONTHS for day in filtered for valid in valid_dates(day))
    assert any(valid.month == 10 for day in legacy for valid in valid_dates(day))


def test_area_weights_are_normalized_symmetric_and_maskable():
    lat = np.array([-90.0, -60.0, 0.0, 60.0, 90.0])
    weights = area_weights(lat)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] == pytest.approx(weights[-2])
    assert weights[2] > weights[1]

    values = np.array([[100.0, 100.0], [2.0, 4.0], [10.0, 14.0]])
    small_lat = np.array([-90.0, 0.0, 90.0])
    mask = np.array([[True, True], [True, False], [True, True]])
    assert weighted_spatial_mean(values, small_lat, mask) == pytest.approx(2.0)


def test_torch_area_weighted_mean_preserves_gradient():
    values = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], requires_grad=True)
    result = weighted_spatial_mean(values, torch.tensor([-30.0, 30.0]))
    assert torch.allclose(result, torch.tensor([4.0]))
    result.sum().backward()
    assert values.grad is not None
    assert torch.all(values.grad > 0)
