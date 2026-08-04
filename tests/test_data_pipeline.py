"""Fast synthetic tests for ERA5 tasking, regridding, and lazy cache reads."""

from datetime import date, timedelta
from dataclasses import asdict
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import types

import numpy as np
import pytest
from netCDF4 import Dataset as NetCDFDataset

from data_pipeline.build_cache import CACHE_CHANNELS, DailySlice, LazyGlobalZarrDataset, write_zarr_cache
from data_pipeline.check_cache import check_cached_slice
from data_pipeline.build_heat_index_target import iter_native_heat_index
from data_pipeline.download_era5 import (
    CDS_CLIMATE_API_URL,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PER_DATASET_WORKERS,
    DEFAULT_SEGMENTS_PER_FILE,
    MONTHS,
    PREFERRED_DAILY_DATASET,
    PRESSURE_LEVEL_DATASET,
    build_download_tasks,
    _download_result_in_segments,
    is_retryable_cds_error,
    retrieve_task,
    run_tasks,
    task_complete,
    validate_cds_endpoint,
)
from data_pipeline.regrid import GridSpec, regrid_field
from data_pipeline.validate_pipeline import (
    expected_date_labels,
    validate_cache_store,
    validate_raw_manifest,
)
from ens_target_grid import LazyGlobalChannel, LazyGlobalTruth
from global_dataset import GlobalHeatCastDataset, identity_preprocessor
from spatial_weights import weighted_spatial_mean
from heat_index import heat_index_from_tmax_dewpoint_c


def test_download_manifest_is_chunked_and_uses_pinned_official_datasets(tmp_path: Path):
    assert DEFAULT_DOWNLOAD_WORKERS == 8
    assert DEFAULT_PER_DATASET_WORKERS == 1
    assert DEFAULT_MAX_RETRIES == 12
    assert DEFAULT_SEGMENTS_PER_FILE == 1
    tasks = build_download_tasks(tmp_path, years=(1979,), months=MONTHS)
    assert len(tasks) == 6
    assert {task.group for task in tasks} == {
        "daily_tmax", "daily_t2m", "single_levels",
        "pressure_geopotential", "pressure_850", "static",
    }
    daily = next(task for task in tasks if task.group == "daily_tmax")
    assert daily.dataset == PREFERRED_DAILY_DATASET
    assert daily.request["daily_statistic"] == "daily_maximum"
    assert daily.request["time_zone"] == "utc+00:00"
    assert daily.request["data_format"] == "netcdf"
    assert daily.request["download_format"] == "unarchived"
    assert daily.request["month"] == [f"{value:02d}" for value in MONTHS]
    assert Path(daily.target).name == "daily_tmax_1979.nc"
    assert all(task.year == 1979 for task in tasks)
    pressure = next(task for task in tasks if task.group == "pressure_850")
    assert pressure.dataset == PRESSURE_LEVEL_DATASET == "reanalysis-era5-pressure-levels"
    assert pressure.request["time"] == ["00:00"]
    assert pressure.request["data_format"] == "netcdf"
    blocked = next(
        task for task in build_download_tasks(
            tmp_path, years=(1979,), months=(1,), pressure_dataset=None
        )
        if task.group == "pressure_850"
    )
    with pytest.raises(RuntimeError, match="Pressure-level CDS dataset is empty"):
        retrieve_task(object(), blocked)


def test_yearly_chunking_reduces_full_archive_to_231_requests(tmp_path: Path):
    annual = build_download_tasks(tmp_path, years=range(1979, 2025), months=MONTHS)
    monthly = build_download_tasks(
        tmp_path,
        years=(1979,),
        months=MONTHS,
        chunking="monthly",
    )
    assert len(annual) == 46 * 5 + 1 == 231
    assert len(monthly) == 12 * 5 + 1 == 61
    assert Path(monthly[0].target).name == "daily_tmax_197901.nc"


def test_heat_index_download_adds_only_daily_mean_dewpoint(tmp_path: Path):
    tasks = build_download_tasks(
        tmp_path,
        years=(1979,),
        months=MONTHS,
        enable_heat_index=True,
    )
    assert len(tasks) == 7
    dewpoint = next(task for task in tasks if task.group == "daily_d2m")
    assert dewpoint.dataset == PREFERRED_DAILY_DATASET
    assert dewpoint.request["variable"] == "2m_dewpoint_temperature"
    assert dewpoint.request["daily_statistic"] == "daily_mean"
    assert Path(dewpoint.target).name == "daily_d2m_1979.nc"
    single = next(task for task in tasks if task.group == "single_levels")
    assert "2m_dewpoint_temperature" not in single.request["variable"]

    full = build_download_tasks(
        tmp_path,
        years=range(1979, 2025),
        months=MONTHS,
        enable_heat_index=True,
    )
    assert len(full) == 46 * 6 + 1 == 277


def test_native_heat_index_stream_uses_daily_tmax_and_daily_mean_dewpoint(tmp_path: Path):
    raw_root = tmp_path / "raw" / "era5"
    for group, variable, value in (
        ("daily_tmax", "t2m", 303.15),
        ("daily_d2m", "d2m", 293.15),
    ):
        path = raw_root / group / "1979" / f"{group}_197901.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        with NetCDFDataset(path, "w") as output:
            output.createDimension("valid_time", 1)
            output.createDimension("latitude", 2)
            output.createDimension("longitude", 4)
            output.createVariable("valid_time", "i8", ("valid_time",))[:] = [19790101]
            output.createVariable("latitude", "f4", ("latitude",))[:] = [45.0, -45.0]
            output.createVariable("longitude", "f4", ("longitude",))[:] = [0, 90, 180, 270]
            field = output.createVariable(
                variable, "f4", ("valid_time", "latitude", "longitude")
            )
            field.units = "K"
            field[:] = value
    records = list(iter_native_heat_index(raw_root, (1979,), (1,), chunking="monthly"))
    assert len(records) == 1
    valid_date, values, lat, lon = records[0]
    assert valid_date == date(1979, 1, 1)
    assert values.shape == (2, 4)
    expected = heat_index_from_tmax_dewpoint_c(np.array([30.0]), np.array([20.0]))[0]
    assert np.allclose(values, expected, atol=1e-5)
    assert np.array_equal(lat, [45.0, -45.0])
    assert np.array_equal(lon, [0.0, 90.0, 180.0, 270.0])


def test_era5_endpoint_preflight_rejects_ecds_without_exposing_key(
    tmp_path: Path, monkeypatch
):
    bad = tmp_path / "ecds.rc"
    bad.write_text(
        "url: https://ecds.ecmwf.int/api\nkey: secret-fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(bad))
    with pytest.raises(RuntimeError, match="separate ECMWF ECDS/S2S") as error:
        validate_cds_endpoint()
    assert "secret-fixture-token" not in str(error.value)

    good = tmp_path / "era5.rc"
    good.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: secret-fixture-token\n",
        encoding="utf-8",
    )
    assert validate_cds_endpoint(good) == good


def test_download_tasks_execute_concurrently(tmp_path: Path, monkeypatch, capsys):
    import data_pipeline.download_era5 as downloader

    config = tmp_path / "era5.rc"
    config.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(config))
    monkeypatch.setitem(
        sys.modules,
        "cdsapi",
        types.SimpleNamespace(Client=lambda: object()),
    )

    barrier = threading.Barrier(2)
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def fake_prepare(_client, task):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        barrier.wait(timeout=2.0)
        with lock:
            state["active"] -= 1
        return task

    monkeypatch.setattr(downloader, "prepare_task", fake_prepare)
    monkeypatch.setattr(
        downloader,
        "download_prepared_task",
        lambda _result, task, *_args: f"retrieved fixture {task.group}",
    )
    all_tasks = build_download_tasks(tmp_path, years=(1979,), months=(1,))
    tasks = (
        next(task for task in all_tasks if task.group == "daily_tmax"),
        next(task for task in all_tasks if task.group == "single_levels"),
    )
    run_tasks(tasks, workers=2)
    output = capsys.readouterr().out
    assert state["maximum"] == 2
    assert "2 download workers" in output
    assert "[2/2]" in output


def test_same_dataset_requests_are_serialized(tmp_path: Path, monkeypatch):
    import data_pipeline.download_era5 as downloader

    config = tmp_path / "era5.rc"
    config.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(config))
    monkeypatch.setitem(
        sys.modules,
        "cdsapi",
        types.SimpleNamespace(Client=lambda: object()),
    )
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def observed_prepare(_client, task):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return task

    monkeypatch.setattr(downloader, "prepare_task", observed_prepare)
    monkeypatch.setattr(
        downloader,
        "download_prepared_task",
        lambda _result, task, *_args: f"retrieved fixture {task.group}",
    )
    tasks = build_download_tasks(tmp_path, years=(1979,), months=(1,))[:2]
    run_tasks(tasks, workers=2, per_dataset_workers=1)
    assert tasks[0].dataset == tasks[1].dataset
    assert state["maximum"] == 1


def test_same_dataset_requests_use_configured_parallel_lanes(tmp_path: Path, monkeypatch):
    import data_pipeline.download_era5 as downloader

    config = tmp_path / "era5.rc"
    config.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(config))
    monkeypatch.setitem(
        sys.modules,
        "cdsapi",
        types.SimpleNamespace(Client=lambda: object()),
    )
    barrier = threading.Barrier(2)
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def observed_prepare(_client, task):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        barrier.wait(timeout=2.0)
        with lock:
            state["active"] -= 1
        return task

    monkeypatch.setattr(downloader, "prepare_task", observed_prepare)
    monkeypatch.setattr(
        downloader,
        "download_prepared_task",
        lambda _result, task, *_args: f"retrieved fixture {task.group}",
    )
    tasks = tuple(
        next(
            task for task in build_download_tasks(tmp_path, years=(year,), months=(1,))
            if task.group == "daily_tmax"
        )
        for year in (1979, 1980)
    )
    run_tasks(tasks, workers=2, per_dataset_workers=2)
    assert state["maximum"] == 2


def test_ready_cds_result_downloads_with_parallel_http_ranges(tmp_path: Path, monkeypatch):
    payload = b"abcdefghijklmnopqrstuvwxyz"
    observed = []

    class Response:
        status_code = 206

        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield self.content

    def get(_url, *, headers, stream, timeout):
        assert stream is True
        assert timeout == (30, 300)
        start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        observed.append((start, stop))
        return Response(payload[start:stop + 1])

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    result = types.SimpleNamespace(
        location="https://signed.example/result.nc",
        content_length=len(payload),
    )
    target = tmp_path / "result.nc.part"
    assert _download_result_in_segments(result, target, segments=4) is True
    assert target.read_bytes() == payload
    assert sorted(observed) == [(0, 5), (6, 12), (13, 18), (19, 25)]
    assert not tuple(tmp_path.glob("*.segment*"))


def test_congested_dataset_lane_does_not_starve_other_datasets(
    tmp_path: Path, monkeypatch
):
    import data_pipeline.download_era5 as downloader

    config = tmp_path / "era5.rc"
    config.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(config))
    monkeypatch.setitem(
        sys.modules,
        "cdsapi",
        types.SimpleNamespace(Client=lambda: object()),
    )
    daily_release = threading.Event()
    single_downloaded = threading.Event()

    def fake_prepare(_client, task):
        if task.dataset == PREFERRED_DAILY_DATASET:
            assert daily_release.wait(timeout=5.0)
        return task

    def fake_download(_result, task, *_args):
        if task.group == "single_levels":
            single_downloaded.set()
        return f"retrieved fixture {task.group}"

    monkeypatch.setattr(downloader, "prepare_task", fake_prepare)
    monkeypatch.setattr(downloader, "download_prepared_task", fake_download)
    all_tasks = build_download_tasks(tmp_path, years=(1979, 1980), months=(1,))
    daily_tasks = tuple(
        task for task in all_tasks if task.dataset == PREFERRED_DAILY_DATASET
    )
    single_task = next(task for task in all_tasks if task.group == "single_levels")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_tasks, daily_tasks + (single_task,), workers=2)
        try:
            assert single_downloaded.wait(timeout=2.0)
        finally:
            daily_release.set()
        future.result(timeout=5.0)


def test_same_dataset_next_request_starts_while_previous_result_downloads(
    tmp_path: Path, monkeypatch
):
    import data_pipeline.download_era5 as downloader

    monkeypatch.setattr(downloader, "validate_download_file", lambda _path, _task: None)
    tasks = tuple(
        next(
            task for task in build_download_tasks(
                tmp_path, years=(year,), months=(1,)
            )
            if task.group == "daily_tmax"
        )
        for year in (1979, 1980)
    )
    request_gate = threading.BoundedSemaphore(1)
    second_request_started = threading.Event()
    lock = threading.Lock()
    request_count = 0

    class Result:
        def __init__(self, request_number):
            self.request_number = request_number

        def download(self, target):
            if self.request_number == 1:
                assert second_request_started.wait(timeout=2.0)
            Path(target).write_bytes(b"fixture")

    class Client:
        def retrieve(self, _dataset, _request):
            nonlocal request_count
            with lock:
                request_count += 1
                request_number = request_count
            if request_number == 2:
                second_request_started.set()
            return Result(request_number)

    client = Client()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(retrieve_task, client, task, request_gate=request_gate)
            for task in tasks
        ]
        assert all("retrieved:" in future.result(timeout=5.0) for future in futures)
    assert second_request_started.is_set()


def test_cds_queue_limit_retries_with_backoff(tmp_path: Path, monkeypatch, capsys):
    import data_pipeline.download_era5 as downloader

    config = tmp_path / "era5.rc"
    config.write_text(
        f"url: {CDS_CLIMATE_API_URL}\nkey: fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CDSAPI_RC", str(config))
    monkeypatch.setitem(
        sys.modules,
        "cdsapi",
        types.SimpleNamespace(Client=lambda: object()),
    )
    attempts = []
    sleeps = []

    def throttled_once(_client, task):
        attempts.append(task.group)
        if len(attempts) == 1:
            raise RuntimeError(
                "Number queued requests for this dataset is temporarily limited."
            )
        return task

    monkeypatch.setattr(downloader, "prepare_task", throttled_once)
    monkeypatch.setattr(
        downloader,
        "download_prepared_task",
        lambda _result, task, *_args: f"retrieved fixture {task.group}",
    )
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    task = build_download_tasks(tmp_path, years=(1979,), months=(1,))[0]
    run_tasks(
        (task,),
        workers=1,
        max_retries=2,
        retry_base_seconds=0.25,
    )
    assert len(attempts) == 2
    assert sleeps == [0.25]
    assert "retry 1/2 in 0s" in capsys.readouterr().out
    assert is_retryable_cds_error(RuntimeError("unrelated failure")) is False


def test_download_task_is_atomic_idempotent_and_records_source(tmp_path: Path):
    task = next(
        task for task in build_download_tasks(
            tmp_path, years=(1979,), months=(1,), pressure_dataset="fixture-pressure-levels"
        )
        if task.group == "daily_tmax"
    )

    class Client:
        calls = 0

        def retrieve(self, dataset, request):
            self.calls += 1
            assert dataset == PREFERRED_DAILY_DATASET
            class Result:
                @staticmethod
                def download(target):
                    with NetCDFDataset(target, "w") as output:
                        output.createDimension("valid_time", 1)
                        output.createDimension("latitude", 2)
                        output.createDimension("longitude", 3)
                        output.createVariable("valid_time", "i8", ("valid_time",))[:] = [19790101]
                        output.createVariable("latitude", "f4", ("latitude",))[:] = [45.0, -45.0]
                        output.createVariable("longitude", "f4", ("longitude",))[:] = [0.0, 120.0, 240.0]
                        output.createVariable(
                            "t2m", "f4", ("valid_time", "latitude", "longitude")
                        )[:] = 280.0
            return Result()

    client = Client()
    assert "retrieved:" in retrieve_task(client, task)
    assert task_complete(task)
    assert "exists, skipping:" in retrieve_task(client, task)
    assert client.calls == 1
    assert not Path(task.target).with_suffix(".nc.part").exists()


def test_post_download_manifest_validation_is_offline_and_detects_partials(tmp_path: Path):
    task = next(
        task for task in build_download_tasks(tmp_path, years=(1979,), months=(1,))
        if task.group == "daily_tmax"
    )
    target = Path(task.target)
    target.parent.mkdir(parents=True)
    with NetCDFDataset(target, "w") as output:
        output.createDimension("valid_time", 1)
        output.createDimension("latitude", 2)
        output.createDimension("longitude", 3)
        output.createVariable("valid_time", "i8", ("valid_time",))[:] = [19790101]
        output.createVariable("latitude", "f4", ("latitude",))[:] = [45.0, -45.0]
        output.createVariable("longitude", "f4", ("longitude",))[:] = [0.0, 120.0, 240.0]
        output.createVariable("t2m", "f4", ("valid_time", "latitude", "longitude"))[:] = 280.0
    record = asdict(task)
    record["months"] = list(task.months)
    target.with_suffix(".nc.metadata.json").write_text(
        json.dumps({"task": record, "target_source": task.source_choice, "utc_days": True}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([record]), encoding="utf-8")
    summary = validate_raw_manifest(manifest, expected_task_count=1)
    assert summary.task_count == 1
    assert summary.total_bytes == target.stat().st_size

    partial = target.parent / "orphan.nc.part"
    partial.write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="partial files"):
        validate_raw_manifest(manifest, expected_task_count=1)


def test_conservative_regrid_preserves_global_area_mean_and_caches_weights(tmp_path: Path):
    source_lat = np.array([-67.5, -22.5, 22.5, 67.5])
    source_lon = np.arange(0.0, 360.0, 45.0)
    target = GridSpec(
        lat=np.array([-45.0, 45.0]),
        lon=np.arange(0.0, 360.0, 90.0),
        resolution="fixture",
    )
    lat_term = np.sin(np.deg2rad(source_lat))[:, None]
    lon_term = np.cos(np.deg2rad(source_lon))[None, :]
    field = 2.0 + lat_term + lon_term
    weights = tmp_path / "conservative_weights.npz"
    regridded = regrid_field(
        field,
        source_lat,
        source_lon,
        target,
        method="conservative",
        weights_path=weights,
        prefer_xesmf=False,
    )
    assert weights.is_file()
    source_mean = weighted_spatial_mean(field, source_lat)
    target_mean = weighted_spatial_mean(regridded, target.lat)
    assert target_mean == pytest.approx(source_mean, abs=1e-6)
    second = regrid_field(
        field, source_lat, source_lon, target,
        method="conservative", weights_path=weights, prefer_xesmf=False,
    )
    assert np.array_equal(regridded, second)


def test_bilinear_regrid_wraps_zero_and_360_longitude():
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.broadcast_to(np.cos(np.deg2rad(source_lon)), (2, 4))
    target = GridSpec(np.array([0.0, 30.0]), np.array([359.0, 1.0]), "fixture")
    output = regrid_field(
        field, source_lat, source_lon, target,
        method="bilinear", prefer_xesmf=False,
    )
    assert np.allclose(output[:, 0], output[:, 1], atol=1e-6)
    assert np.all(output > 0.98)


def test_npz_weight_cache_never_routes_through_xesmf(tmp_path: Path, monkeypatch):
    import data_pipeline.regrid as regrid_module

    monkeypatch.setattr(
        regrid_module,
        "_xesmf_regrid",
        lambda *_args, **_kwargs: pytest.fail("xESMF received a SciPy .npz cache path"),
    )
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.array([0.0, 90.0, 180.0, 270.0])
    target = GridSpec(source_lat, source_lon, "fixture")
    output = regrid_module.regrid_field(
        np.ones((2, 4), dtype=np.float32),
        source_lat,
        source_lon,
        target,
        method="conservative",
        weights_path=tmp_path / "weights.npz",
    )
    assert np.allclose(output, 1.0)


def test_cache_regridding_forces_reusable_scipy_backend(tmp_path: Path, monkeypatch):
    import data_pipeline.build_cache as cache_module

    class Coordinate:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, _key):
            return self.values

    class Dataset:
        variables = {
            "latitude": Coordinate([-45.0, 45.0]),
            "longitude": Coordinate([0.0, 90.0, 180.0, 270.0]),
        }

    observed = {}

    def fake_regrid(field, _lat, _lon, _target, **kwargs):
        observed.update(kwargs)
        return np.asarray(field)

    monkeypatch.setattr(cache_module, "regrid_field", fake_regrid)
    field = np.ones((2, 4), dtype=np.float32)
    cache_module._regrid(field, Dataset(), GridSpec(
        np.array([-45.0, 45.0]), np.array([0.0, 90.0, 180.0, 270.0]), "fixture"
    ), "bilinear", tmp_path, "smooth")
    assert observed["prefer_xesmf"] is False


def test_cache_regrids_execute_concurrently_on_in_memory_arrays(tmp_path: Path, monkeypatch):
    import data_pipeline.build_cache as cache_module

    barrier = threading.Barrier(2)
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def observed_regrid(field, *_args, **_kwargs):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        barrier.wait(timeout=2.0)
        with lock:
            state["active"] -= 1
        return np.asarray(field)

    monkeypatch.setattr(cache_module, "_regrid_array", observed_regrid)
    requests = {
        name: (np.full((2, 3), value), np.arange(2), np.arange(3), "bilinear", name)
        for value, name in enumerate(("first", "second"))
    }
    grid = GridSpec(np.array([-45.0, 45.0]), np.array([0.0, 120.0, 240.0]), "fixture")
    with ThreadPoolExecutor(max_workers=2) as executor:
        output = cache_module._execute_regrid_requests(requests, grid, tmp_path, executor)
    assert state["maximum"] == 2
    assert set(output) == set(requests)


def test_cache_can_queue_multiple_days_for_a_large_worker_pool(tmp_path: Path, monkeypatch):
    import data_pipeline.build_cache as cache_module

    barrier = threading.Barrier(4)
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def observed_regrid(field, *_args, **_kwargs):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        barrier.wait(timeout=2.0)
        with lock:
            state["active"] -= 1
        return np.asarray(field)

    monkeypatch.setattr(cache_module, "_regrid_array", observed_regrid)
    grid = GridSpec(
        np.array([-45.0, 45.0]),
        np.array([0.0, 120.0, 240.0]),
        "fixture",
    )

    def requests(offset):
        return {
            name: (
                np.full((2, 3), offset + value),
                np.arange(2),
                np.arange(3),
                "bilinear",
                name,
            )
            for value, name in enumerate(("first", "second"))
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        first = cache_module._submit_regrid_requests(
            requests(0), grid, tmp_path, executor
        )
        second = cache_module._submit_regrid_requests(
            requests(10), grid, tmp_path, executor
        )
        first_output = cache_module._resolve_regrid_futures(first)
        second_output = cache_module._resolve_regrid_futures(second)

    assert state["maximum"] == 4
    assert set(first_output) == {"first", "second"}
    assert set(second_output) == {"first", "second"}


class LogicalLazyArray:
    """Large logical array that allocates only the requested sample slice."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.requests = []

    @property
    def oindex(self):
        return self

    def __getitem__(self, key):
        self.requests.append(key)
        time_key, lat_key, lon_key, channel_key = key
        n_time = len(time_key) if isinstance(time_key, list) else 1
        n_lat = self.shape[1] if isinstance(lat_key, slice) else len(lat_key)
        n_lon = self.shape[2] if isinstance(lon_key, slice) else len(lon_key)
        if isinstance(channel_key, slice):
            return np.zeros((n_time, n_lat, n_lon, self.shape[3]), dtype=np.float32)
        return np.zeros((n_time, n_lat, n_lon), dtype=np.float32)


def test_lazy_dataset_opens_only_on_getitem_and_reads_bounded_times():
    logical = LogicalLazyArray((100_000, 121, 240, len(CACHE_CHANNELS)))
    opens = []

    def opener(path):
        opens.append(path)
        return {"data": logical}

    metadata = {
        "shape": list(logical.shape),
        "channels": list(CACHE_CHANNELS),
        "time_values": [20000101] * logical.shape[0],
    }
    dataset = LazyGlobalZarrDataset(
        "logical.zarr", (100,), opener=opener, metadata=metadata
    )
    assert dataset._store is None
    assert opens == []
    state = dataset.__getstate__()
    assert state["_store"] is None
    sample = dataset[0]
    assert opens == ["logical.zarr"]
    assert sample["context"].shape == (3, 121, 240, len(CACHE_CHANNELS))
    assert sample["target"].shape == (14, 121, 240)
    assert max(len(request[0]) for request in logical.requests) <= 14
    assert sum(array.nbytes for array in (sample["context"].numpy(), sample["target"].numpy())) < 10_000_000


def test_slice_checker_detects_agreement_and_corruption():
    height, width = 2, 3
    data = np.zeros((1, height, width, len(CACHE_CHANNELS)), dtype=np.float32)
    expected = {name: data[0, :, :, index].copy() for index, name in enumerate(CACHE_CHANNELS)}

    class Root(dict):
        attrs = {"channels": list(CACHE_CHANNELS)}

    root = Root(data=data)
    checks = check_cached_slice(Path("fixture.zarr"), 0, expected, opener=lambda _: root)
    assert all(check.passed for check in checks)
    expected["tmax"][0, 0] = 1.0
    checks = check_cached_slice(Path("fixture.zarr"), 0, expected, opener=lambda _: root)
    assert next(check for check in checks if check.channel == "tmax").passed is False


def test_zarr_writer_uses_time_one_chunks_and_resumes(tmp_path: Path, capsys):
    zarr = pytest.importorskip("zarr")
    grid = GridSpec(np.array([45.0, -45.0]), np.array([0.0, 120.0, 240.0]), "fixture")

    def daily(day):
        fields = {
            name: np.full(grid.shape, float(day), dtype=np.float32)
            for name in CACHE_CHANNELS
        }
        return DailySlice(date(2000, 1, 1) + timedelta(days=day), fields)

    store = tmp_path / "cache.zarr"
    write_zarr_cache((daily(0), daily(1)), store, grid, target_source="daily_statistics")
    metadata = write_zarr_cache(
        (daily(0), daily(1), daily(2)), store, grid,
        target_source="daily_statistics", progress_every=1,
    )
    root = zarr.open_group(str(store), mode="r")
    assert root["data"].shape == (3, 2, 3, len(CACHE_CHANNELS))
    assert root["data"].chunks[0] == 1
    assert root["time"][:].tolist() == [20000101, 20000102, 20000103]
    assert metadata["shape"][0] == 3
    assert "CACHE_PROGRESS committed_days=3" in capsys.readouterr().out

    from data_pipeline.build_cache import cache_resume_context_start
    assert cache_resume_context_start(store) == date(1999, 12, 16)

    lazy_soil = LazyGlobalChannel(store, "swvl1_trailing20")
    selected = lazy_soil.read_pixels_times([0, 5], [0, 2])
    assert selected.shape == (2, 2)
    assert selected.tolist() == [[0.0, 2.0], [0.0, 2.0]]


def test_completed_cache_validation_checks_calendar_schema_and_finite_samples(tmp_path: Path):
    zarr = pytest.importorskip("zarr")
    grid = GridSpec(np.array([45.0, -45.0]), np.array([0.0, 120.0, 240.0]), "fixture")
    start = date(2000, 1, 1)

    def daily(day):
        fields = {
            name: np.full(grid.shape, float(day), dtype=np.float32)
            for name in CACHE_CHANNELS
        }
        fields["land_mask"][:] = 1.0
        fields["sst_valid"][:] = 1.0
        return DailySlice(start + timedelta(days=day), fields)

    store = tmp_path / "validated.zarr"
    write_zarr_cache(tuple(daily(day) for day in range(3)), store, grid, target_source="fixture")
    labels = tuple(20000101 + day for day in range(3))
    summary = validate_cache_store(
        store, grid=grid, date_labels=labels, target_source="fixture", sample_count=3
    )
    assert summary.shape == (3, 2, 3, len(CACHE_CHANNELS))
    assert summary.chunks[0] == 1
    assert expected_date_labels((2000,), (2,))[:2] == (20000201, 20000202)

    root = zarr.open_group(str(store), mode="a")
    root["data"][1, 0, 0, CACHE_CHANNELS.index("tmax")] = np.inf
    with pytest.raises(RuntimeError, match="Non-finite cache value"):
        validate_cache_store(
            store, grid=grid, date_labels=labels, target_source="fixture", sample_count=3
        )


def test_global_training_dataset_preserves_date_labels_while_exposing_legacy_offsets(tmp_path: Path):
    pytest.importorskip("zarr")
    grid = GridSpec(np.array([45.0, -45.0]), np.array([0.0, 120.0, 240.0]), "fixture")

    def daily(day):
        fields = {name: np.full(grid.shape, float(day), dtype=np.float32) for name in CACHE_CHANNELS}
        fields["land_mask"][:] = 1.0
        fields["sst_valid"][:] = 1.0
        return DailySlice(date(2000, 1, 1) + timedelta(days=day), fields)

    store = tmp_path / "training.zarr"
    write_zarr_cache(tuple(daily(day) for day in range(31)), store, grid, target_source="fixture")
    dataset = GlobalHeatCastDataset(
        store,
        (2,),
        condition_vectors=np.zeros((31, 8), dtype=np.float32),
        preprocessor=identity_preprocessor(grid.shape),
    )
    assert dataset._store is None
    sample = dataset[0]
    assert sample[0].shape == (14, 2, 3)
    assert sample[1].shape == (1, 2, 3)
    assert sample[4].shape == (23, 2, 3)
    assert sample[5].shape == (8,)
    assert dataset.date_labels[2] == 20000103
    assert dataset.time_values[2] == (date(2000, 1, 3) - date(1981, 5, 1)).days
    assert np.allclose(sample[0][0].numpy(), 17.0)


def test_global_training_dataset_reads_separate_heat_index_target(tmp_path: Path):
    zarr = pytest.importorskip("zarr")
    grid = GridSpec(np.array([45.0, -45.0]), np.array([0.0, 120.0, 240.0]), "fixture")

    def daily(day):
        fields = {name: np.full(grid.shape, float(day), dtype=np.float32) for name in CACHE_CHANNELS}
        fields["land_mask"][:] = 1.0
        fields["sst_valid"][:] = 1.0
        return DailySlice(date(2000, 1, 1) + timedelta(days=day), fields)

    store = tmp_path / "heat-index-training.zarr"
    write_zarr_cache(tuple(daily(day) for day in range(31)), store, grid, target_source="fixture")
    root = zarr.open_group(str(store), mode="a")
    heat = root.create_dataset(
        "heat_index",
        shape=(31,) + grid.shape,
        chunks=(1,) + grid.shape,
        dtype="f4",
    )
    for index in range(31):
        heat[index] = 100.0 + index
    root.attrs["primary_target"] = "heat_index"
    dataset = GlobalHeatCastDataset(
        store,
        (2,),
        condition_vectors=np.zeros((31, 8), dtype=np.float32),
        preprocessor=identity_preprocessor(grid.shape),
        target_array="heat_index",
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample[0][0].numpy(), 117.0)
    np.testing.assert_allclose(sample[1].numpy(), 2.0)
    lazy_truth = LazyGlobalTruth(store)
    assert lazy_truth.channel == "heat_index"
    np.testing.assert_allclose(lazy_truth[:, :, 17], 117.0)
