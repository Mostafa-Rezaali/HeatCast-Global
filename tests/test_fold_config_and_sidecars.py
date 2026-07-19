"""Data-free tests for user-pinned fold roles and bounded threshold building."""

from pathlib import Path

import numpy as np
import pytest

import cfm_mesh_train as cfm
from build_global_fold_sidecars import build_streaming_thresholds
from fold_config import GLOBAL_YEARS, validate_fold_table


def fixture_fold_table():
    years = list(GLOBAL_YEARS)
    records = []
    for fold in range(5):
        test = set(years[fold::5])
        calibration = set(years[(fold + 1) % 5::5])
        records.append({
            "fold": fold,
            "train_years": sorted(set(years) - test - calibration),
            "calibration_years": sorted(calibration),
            "test_years": sorted(test),
        })
    return {"folds": records}


def test_approved_fold_table_requires_disjoint_complete_five_fold_partition():
    folds = validate_fold_table(fixture_fold_table())
    assert set(folds) == set(range(5))
    assert set().union(*(set(folds[fold]["test_years"]) for fold in folds)) == set(GLOBAL_YEARS)
    attacked = fixture_fold_table()
    attacked["folds"][0]["train_years"].append(attacked["folds"][0]["test_years"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_fold_table(attacked)


class _ThresholdFixture:
    indices = (0, 1, 2)
    date_labels = np.array([20000501, 20000601, 20000701], dtype=np.int32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        target = np.full((14, 2, 2), float(item), dtype=np.float32)
        return (target,)


def test_streaming_threshold_builder_uses_disk_rows_and_window_months(tmp_path: Path):
    original_size = cfm.Config.IMAGE_SIZE
    original_leads = cfm.Config.PREDICTION_LEADS
    original_tube = cfm.Config.MULTI_LEAD_TUBE
    try:
        cfm.Config.IMAGE_SIZE = (2, 2)
        cfm.Config.MULTI_LEAD_TUBE = True
        cfm.Config.PREDICTION_LEADS = tuple(range(15, 29))
        thresholds, base_rates, files = build_streaming_thresholds(
            _ThresholdFixture(),
            (2000,),
            tmp_path,
            block_pixels=2,
        )
        assert files
        assert thresholds["w34"]["q95"].shape == (12, 2, 2)
        assert np.all(np.isfinite(thresholds["w34"]["q95"][4:7]))
        assert np.all(np.isfinite(base_rates["week3"]["upper_tercile"][4:7]))
    finally:
        cfm.Config.IMAGE_SIZE = original_size
        cfm.Config.PREDICTION_LEADS = original_leads
        cfm.Config.MULTI_LEAD_TUBE = original_tube
