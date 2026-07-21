"""Fast synthetic tests for ERA5 tasking, regridding, and lazy cache reads."""

from datetime import date, timedelta
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
from data_pipeline.download_era5 import (
    CDS_CLIMATE_API_URL,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PER_DATASET_WORKERS,
    MONTHS,
    PREFERRED_DAILY_DATASET,
    PRESSURE_LEVEL_DATASET,
    build_download_tasks,
    is_retryable_cds_error,
    retrieve_task,
    run_tasks,
    task_complete,
    validate_cds_endpoint,
)
from data_pipeline.regrid import GridSpec, regrid_field
from ens_target_grid import LazyGlobalChannel
from global_dataset import GlobalHeatCastDataset, identity_preprocessor
from spatial_weights import weighted_spatial_mean


def test_download_manifest_is_chunked_and_uses_pinned_official_datasets(tmp_path: Path):
    assert DEFAULT_DOWNLOAD_WORKERS == 8
    assert DEFAULT_PER_DATASET_WORKERS == 1
    assert DEFAULT_MAX_RETRIES == 12
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
        lambda _result, task: f"retrieved fixture {task.group}",
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
        lambda _result, task: f"retrieved fixture {task.group}",
    )
    tasks = build_download_tasks(tmp_path, years=(1979,), months=(1,))[:2]
    run_tasks(tasks, workers=2, per_dataset_workers=1)
    assert tasks[0].dataset == tasks[1].dataset
    assert state["maximum"] == 1


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

    def fake_download(_result, task):
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
        lambda _result, task: f"retrieved fixture {task.group}",
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


def test_zarr_writer_uses_time_one_chunks_and_resumes(tmp_path: Path):
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
    metadata = write_zarr_cache((daily(0), daily(1), daily(2)), store, grid, target_source="daily_statistics")
    root = zarr.open_group(str(store), mode="r")
    assert root["data"].shape == (3, 2, 3, len(CACHE_CHANNELS))
    assert root["data"].chunks[0] == 1
    assert root["time"][:].tolist() == [20000101, 20000102, 20000103]
    assert metadata["shape"][0] == 3

    lazy_soil = LazyGlobalChannel(store, "swvl1_trailing20")
    selected = lazy_soil.read_pixels_times([0, 5], [0, 2])
    assert selected.shape == (2, 2)
    assert selected.tolist() == [[0.0, 2.0], [0.0, 2.0]]


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
