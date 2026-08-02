"""Data-free tests for the approved global teleconnection cache builder."""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from data_pipeline.build_condition_cache import (
    TELECONNECTION_CHANNELS,
    expand_condition_indices,
    expand_monthly_indices,
    parse_daily_cpc_file,
    parse_pdo_workbook,
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
    assert "--pdo_workbook" in result.stdout


def _sources():
    sources = {}
    for column, name in enumerate(TELECONNECTION_CHANNELS):
        if name in ("pna", "nao", "ao"):
            sources[name] = {
                20000101: float(column + 1),
                20000102: float(column + 1.5),
                20000201: float(column + 2),
                20000202: float(column + 2.5),
            }
        else:
            sources[name] = {(2000, 1): float(column + 1), (2000, 2): float(column + 2)}
    return sources


def test_mixed_daily_monthly_indices_expand_in_approved_order():
    labels = np.array([20000101, 20000102, 20000201, 20000202], dtype=np.int32)
    values = expand_condition_indices(labels, _sources())
    assert TELECONNECTION_CHANNELS == ("pna", "nao", "nino34", "pdo", "ao")
    assert values.shape == (4, 5)
    np.testing.assert_array_equal(values[0], np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(values[1, [0, 1, 4]], np.array([1.5, 2.5, 5.5]))
    np.testing.assert_array_equal(values[1, [2, 3]], values[0, [2, 3]])
    np.testing.assert_array_equal(values[2], np.arange(2, 7, dtype=np.float32))
    np.testing.assert_array_equal(values[3, [0, 1, 4]], np.array([2.5, 3.5, 6.5]))
    np.testing.assert_array_equal(values[3, [2, 3]], values[2, [2, 3]])
    assert expand_monthly_indices is expand_condition_indices


def test_monthly_indices_reject_missing_and_sentinel_values():
    sources = _sources()
    sources["ao"] = {20000101: -999.0}
    with pytest.raises(RuntimeError, match="do not cover the ERA5 cache axis"):
        expand_monthly_indices(np.array([20000101, 20000201]), sources)


def test_daily_cpc_parser(tmp_path: Path):
    source = tmp_path / "pna.daily"
    source.write_text("2000 1 1 0.125\n2000 1 2 -0.750\n", encoding="utf-8")
    assert parse_daily_cpc_file(source) == {20000101: 0.125, 20000102: -0.75}


def test_pdo_workbook_parser_without_openpyxl(tmp_path: Path):
    workbook = tmp_path / "PDO.xlsx"
    headers = ["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = [headers, [2024, -1.57, -1.33, -1.52, -2.11, -2.98, -3.15, -3.0, -2.91, -3.56, -3.8, -3.13, -2.03]]
    xml_rows = []
    for row_number, row in enumerate(rows, 1):
        cells = []
        for column, value in enumerate(row):
            reference = f"{chr(ord('A') + column)}{row_number}"
            if isinstance(value, str):
                cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>')
            else:
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    values = parse_pdo_workbook(workbook)
    assert values[(2024, 1)] == -1.57
    assert values[(2024, 12)] == -2.03


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
        assert report[f"{name}_max_abs_difference"] < 3e-8
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
