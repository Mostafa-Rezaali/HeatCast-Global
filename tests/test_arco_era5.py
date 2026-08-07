"""Data-free tests for the Google ARCO-ERA5 streaming downloader."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset as NetCDFDataset

from data_pipeline.download_arco_era5 import (
    ARCO_PROVIDER,
    BASE_GROUPS,
    SUPPORTED_GROUPS,
    _array_metadata_url,
    _parse_time_origin,
    _root_metadata_url,
    _split_gs_url,
    retrieve_arco_task,
    run_arco_tasks,
    selected_dates,
)
from data_pipeline.download_era5 import build_download_tasks


class FakeArcoBackend:
    """Small deterministic in-memory replacement for the remote Zarr store."""

    store_url = "gs://fixture/arco.zarr-v3"

    def __init__(self, fail_once_date=None):
        self.lat = np.asarray([1.0, -1.0])
        self.lon = np.asarray([0.0, 120.0, 240.0])
        self.levels = np.asarray([300, 500, 850])
        self.fail_once_date = fail_once_date
        self.failed = False
        self.surface_starts = []

    def read_surface(self, variable, start, stop):
        self.surface_starts.append(start)
        if self.fail_once_date == start.date() and not self.failed:
            self.failed = True
            raise OSError("fixture transport interruption")
        hours = int((stop - start).total_seconds() // 3600)
        offset = 100.0 if "dewpoint" in variable else 200.0
        values = np.arange(hours, dtype=np.float32)[:, None, None] + offset
        return np.broadcast_to(values, (hours, len(self.lat), len(self.lon))).copy()

    def read_pressure(self, variable, valid_time, level):
        return np.full((len(self.lat), len(self.lon)), float(level), dtype=np.float32)


def _task(tmp_path: Path, group: str, *, heat_index=True):
    tasks = build_download_tasks(
        tmp_path / "raw" / "era5",
        years=(1979,),
        months=(2,),
        enable_heat_index=heat_index,
        chunking="monthly",
    )
    return next(task for task in tasks if task.group == group)


def test_arco_url_and_calendar_contract():
    assert _split_gs_url("gs://bucket/path/to/store") == ("bucket", "path/to/store")
    assert _root_metadata_url("gs://bucket/path/to/store").endswith("/.zattrs")
    assert _array_metadata_url("gs://bucket/path/to/store", "t2m").endswith(
        "/t2m/.zattrs"
    )
    assert _parse_time_origin("hours since 1900-01-01 00:00:00") == datetime(
        1900, 1, 1
    )
    assert "daily_d2m" not in BASE_GROUPS
    assert "daily_d2m" in SUPPORTED_GROUPS
    assert len(selected_dates(1980, (2,))) == 29
    with pytest.raises(ValueError, match="gs://"):
        _split_gs_url("https://storage.googleapis.com/bucket/store")


def test_arco_daily_dewpoint_writes_utc_mean_and_source_metadata(tmp_path: Path):
    task = _task(tmp_path, "daily_d2m")
    backend = FakeArcoBackend()

    message = retrieve_arco_task(
        task,
        backend,
        max_retries=0,
        retry_base_seconds=0.0,
        progress_every=28,
    )

    assert "Google ARCO-ERA5" in message
    with NetCDFDataset(task.target) as dataset:
        assert dataset.getncattr("download_provider") == ARCO_PROVIDER
        assert dataset.getncattr("source") == backend.store_url
        assert dataset.variables["valid_time"][:].tolist() == list(
            range(19790201, 19790229)
        )
        assert dataset.variables["d2m"].shape == (28, 2, 3)
        assert np.allclose(dataset.variables["d2m"][0], 111.5)
        assert dataset.variables["d2m"].units == "K"

    metadata = json.loads(
        Path(task.target).with_suffix(".nc.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["download_provider"] == ARCO_PROVIDER
    assert metadata["arco_store"] == backend.store_url
    assert metadata["validated_bytes"] == Path(task.target).stat().st_size
    assert not Path(task.target + ".part").exists()
    assert not Path(task.target + ".part.progress.json").exists()


def test_arco_resume_continues_after_last_committed_day(tmp_path: Path):
    task = _task(tmp_path, "daily_d2m")
    interrupted = FakeArcoBackend(fail_once_date=selected_dates(1979, (2,))[1])
    with pytest.raises(OSError, match="fixture transport"):
        retrieve_arco_task(
            task,
            interrupted,
            max_retries=0,
            retry_base_seconds=0.0,
            progress_every=28,
        )

    progress = Path(task.target + ".part.progress.json")
    assert json.loads(progress.read_text(encoding="utf-8"))["completed_days"] == 1

    resumed = FakeArcoBackend()
    retrieve_arco_task(
        task,
        resumed,
        max_retries=0,
        retry_base_seconds=0.0,
        progress_every=28,
    )
    assert resumed.surface_starts[0] == datetime(1979, 2, 2)
    with NetCDFDataset(task.target) as dataset:
        assert np.allclose(dataset.variables["d2m"][0], 111.5)
        assert np.allclose(dataset.variables["d2m"][-1], 111.5)


def test_arco_pressure_level_layout_matches_cache_reader(tmp_path: Path):
    task = _task(tmp_path, "pressure_geopotential", heat_index=False)
    retrieve_arco_task(
        task,
        FakeArcoBackend(),
        max_retries=0,
        retry_base_seconds=0.0,
        progress_every=28,
    )
    with NetCDFDataset(task.target) as dataset:
        assert dataset.variables["pressure_level"][:].tolist() == [300, 500]
        assert dataset.variables["z"].shape == (28, 2, 2, 3)
        assert np.allclose(dataset.variables["z"][0, 0], 300.0)
        assert np.allclose(dataset.variables["z"][0, 1], 500.0)


def test_arco_task_pool_skips_completed_compatible_files(tmp_path: Path, capsys):
    task = _task(tmp_path, "daily_d2m")
    backend = FakeArcoBackend()
    run_arco_tasks(
        (task,),
        workers=2,
        backend=backend,
        max_retries=0,
        retry_base_seconds=0.0,
        progress_every=28,
    )
    first_reads = len(backend.surface_starts)
    run_arco_tasks(
        (task,),
        workers=2,
        backend=backend,
        max_retries=0,
        retry_base_seconds=0.0,
        progress_every=28,
    )
    assert len(backend.surface_starts) == first_reads
    assert "exists, skipping" in capsys.readouterr().out
