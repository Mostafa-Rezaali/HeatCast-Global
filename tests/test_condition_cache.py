"""Data-free tests for the approved global teleconnection cache builder."""

from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from data_pipeline.build_condition_cache import (
    TELECONNECTION_CHANNELS,
    expand_monthly_indices,
    validate_preserved_legacy_channels,
)


def test_condition_cache_direct_script_entrypoint():
    script = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "build_condition_cache.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--legacy_condtrain" in result.stdout


def _sources():
    return {
        name: {(2000, 1): float(column + 1), (2000, 2): float(column + 2)}
        for column, name in enumerate(TELECONNECTION_CHANNELS)
    }


def test_monthly_indices_expand_in_approved_order():
    labels = np.array([20000101, 20000131, 20000201, 20000229], dtype=np.int32)
    values = expand_monthly_indices(labels, _sources())
    assert TELECONNECTION_CHANNELS == ("pna", "nao", "nino34", "pdo", "ao")
    assert values.shape == (4, 5)
    np.testing.assert_array_equal(values[0], np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(values[1], values[0])
    np.testing.assert_array_equal(values[2], np.arange(2, 7, dtype=np.float32))
    np.testing.assert_array_equal(values[3], values[2])


def test_monthly_indices_reject_missing_and_sentinel_values():
    sources = _sources()
    sources["ao"] = {(2000, 1): -999.0}
    with pytest.raises(RuntimeError, match="do not cover the ERA5 cache axis"):
        expand_monthly_indices(np.array([20000101, 20000201]), sources)


def _write_legacy_condtrain(path: Path, labels: np.ndarray, raw: np.ndarray) -> None:
    from netCDF4 import Dataset, date2num

    normalized = (raw - np.min(raw, axis=0)) / (np.max(raw, axis=0) - np.min(raw, axis=0))
    legacy = np.zeros((labels.size, 5), dtype=np.float64)
    legacy[:, 0] = normalized[:, 0]
    legacy[:, 1] = normalized[:, 1]
    legacy[:, 2] = np.linspace(0.0, 1.0, labels.size)
    legacy[:, 3] = normalized[:, 3]
    legacy[:, 4] = normalized[:, 4]
    dates = [
        datetime(int(label) // 10000, (int(label) // 100) % 100, int(label) % 100)
        for label in labels
    ]
    with Dataset(path, "w") as dataset:
        dataset.createDimension("condition", 5)
        dataset.createDimension("time", labels.size)
        condition = dataset.createVariable("CondTrain", "f8", ("condition", "time"))
        condition[:] = legacy.T
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 1900-01-01 00:00:00.0"
        time[:] = date2num(dates, units=time.units)


def test_legacy_validation_proves_four_preserved_channels(tmp_path: Path):
    labels = np.array([20000101, 20000102, 20000201, 20000202], dtype=np.int32)
    raw = expand_monthly_indices(labels, _sources())
    legacy = tmp_path / "legacy.nc"
    _write_legacy_condtrain(legacy, labels, raw)
    report = validate_preserved_legacy_channels(labels, raw, legacy)
    assert report["matched_dates"] == 4
    for name in ("pna", "nao", "pdo", "ao"):
        assert report[f"{name}_max_abs_difference"] == 0.0
        assert report[f"{name}_correlation"] == pytest.approx(1.0)


def test_legacy_validation_rejects_changed_source(tmp_path: Path):
    labels = np.array([20000101, 20000102, 20000201, 20000202], dtype=np.int32)
    raw = expand_monthly_indices(labels, _sources())
    legacy = tmp_path / "legacy.nc"
    _write_legacy_condtrain(legacy, labels, raw)
    changed = raw.copy()
    changed[0, 0] += 0.25
    with pytest.raises(RuntimeError, match="refusing to build"):
        validate_preserved_legacy_channels(labels, changed, legacy)
