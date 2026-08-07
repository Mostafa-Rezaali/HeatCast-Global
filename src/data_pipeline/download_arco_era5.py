"""Stream Google ARCO-ERA5 into HeatCast's resumable raw NetCDF contract.

The public ARCO-ERA5 Zarr store is read anonymously with TensorStore. Hourly
surface temperature and dewpoint are reduced to UTC-day maxima or means while
all other predictors retain the established 00 UTC sampling contract. Output
paths and task metadata remain compatible with ``download_era5.py`` so a
partly completed CDS archive can be resumed without downloading valid files
again. ECMWF S2S/ENS data are intentionally outside this module.
"""

from __future__ import annotations

import argparse
import calendar
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence
from urllib.request import urlopen

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from cfm_mesh_train import Config
from data_pipeline.download_era5 import (
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_SECONDS,
    MONTHS,
    DownloadTask,
    _metadata_path,
    _task_record,
    build_download_tasks,
    parse_years,
    task_complete,
    validate_download_file,
)


ARCO_ERA5_STORE = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ARCO_PROVIDER = "google_arco_era5"
SUPPORTED_GROUPS = (
    "daily_tmax",
    "daily_t2m",
    "daily_d2m",
    "single_levels",
    "pressure_geopotential",
    "pressure_850",
    "static",
)
BASE_GROUPS = tuple(value for value in SUPPORTED_GROUPS if value != "daily_d2m")
MAX_RETRY_DELAY_SECONDS = 300.0

SURFACE_VARIABLES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "sst": "sea_surface_temperature",
    "msl": "mean_sea_level_pressure",
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}
PRESSURE_VARIABLES = {
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}
VARIABLE_UNITS = {
    "t2m": "K",
    "d2m": "K",
    "swvl1": "m3 m-3",
    "swvl2": "m3 m-3",
    "sst": "K",
    "msl": "Pa",
    "z": "m2 s-2",
    "lsm": "1",
    "t": "K",
    "q": "kg kg-1",
    "u": "m s-1",
    "v": "m s-1",
}

# The HiPerGator HDF5 library is not thread-safe. Remote TensorStore reads may
# overlap, but creation and mutation of local NetCDF files remain serialized.
_NETCDF_WRITE_LOCK = threading.Lock()


def _split_gs_url(url: str) -> tuple[str, str]:
    """Return bucket and object prefix for one ``gs://`` URL."""
    value = str(url)
    if not value.startswith("gs://"):
        raise ValueError(f"ARCO store must use gs://, got {value!r}.")
    bucket, separator, path = value[5:].partition("/")
    if not bucket or not separator or not path:
        raise ValueError(f"Incomplete ARCO store URL {value!r}.")
    return bucket, path.rstrip("/")


def _parse_utc_datetime(value: str) -> datetime:
    """Parse an ARCO ISO timestamp and normalize it to naive UTC."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_time_origin(units: str) -> datetime:
    """Parse the public store's integer-hour coordinate origin."""
    prefix = "hours since "
    value = str(units).strip()
    if not value.lower().startswith(prefix):
        raise RuntimeError(f"Unsupported ARCO time units {value!r}.")
    return _parse_utc_datetime(value[len(prefix):].strip())


def _root_metadata_url(store_url: str) -> str:
    bucket, path = _split_gs_url(store_url)
    return f"https://storage.googleapis.com/{bucket}/{path}/.zattrs"


def _array_metadata_url(store_url: str, variable: str) -> str:
    bucket, path = _split_gs_url(store_url)
    return f"https://storage.googleapis.com/{bucket}/{path}/{variable}/.zattrs"


def fetch_arco_root_metadata(store_url: str = ARCO_ERA5_STORE) -> Mapping:
    """Read the small public root metadata object without cloud credentials."""
    with urlopen(_root_metadata_url(store_url), timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not {"valid_time_start", "valid_time_stop"}.issubset(payload):
        raise RuntimeError(f"ARCO root metadata is incomplete: {store_url}.")
    return payload


def fetch_arco_array_metadata(store_url: str, variable: str) -> Mapping:
    """Read public xarray attributes for one ARCO array."""
    with urlopen(_array_metadata_url(store_url, variable), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_arco_array_dimensions(store_url: str, variable: str) -> tuple[str, ...]:
    """Read xarray's authoritative dimension order for one public array."""
    payload = fetch_arco_array_metadata(store_url, variable)
    dimensions = tuple(str(value) for value in payload.get("_ARRAY_DIMENSIONS", ()))
    if not dimensions:
        raise RuntimeError(
            f"ARCO array {variable!r} lacks _ARRAY_DIMENSIONS metadata."
        )
    return dimensions


class TensorStoreArcoBackend:
    """Worker-safe reader for the public ARCO-ERA5 Zarr-v3 hierarchy."""

    def __init__(self, store_url: str = ARCO_ERA5_STORE):
        try:
            import tensorstore as ts
        except ImportError as exc:
            raise RuntimeError(
                "Google ARCO-ERA5 requires TensorStore. Install it once with "
                "python -m pip install tensorstore."
            ) from exc
        self._ts = ts
        self.store_url = str(store_url).rstrip("/")
        self.bucket, self.path = _split_gs_url(self.store_url)
        metadata = fetch_arco_root_metadata(self.store_url)
        try:
            self.valid_time_start = _parse_utc_datetime(metadata["valid_time_start"])
            self.valid_time_stop = _parse_utc_datetime(metadata["valid_time_stop"])
        except KeyError as exc:
            raise RuntimeError(
                f"ARCO metadata lacks required valid-time attributes: {self.store_url}."
            ) from exc
        self._stores = {}
        self._stores_lock = threading.Lock()
        self._array_metadata = {}
        self._array_metadata_lock = threading.Lock()
        self._context = ts.Context({
            "cache_pool": {"total_bytes_limit": 512 * 1024 * 1024},
        })
        self.lat = self._read_coordinate("latitude")
        self.lon = self._read_coordinate("longitude")
        self.levels = self._read_coordinate("level")
        time_metadata = self._metadata("time")
        self.time_origin = _parse_time_origin(time_metadata.get("units", ""))
        time_store = self._open_array("time")
        required_size = self.hour_index(self.valid_time_stop) + 1
        if int(time_store.shape[0]) < required_size:
            raise RuntimeError(
                f"ARCO time array has {time_store.shape[0]} values but metadata "
                f"requires at least {required_size}."
            )

    def _open_array(self, variable: str):
        with self._stores_lock:
            cached = self._stores.get(variable)
            if cached is not None:
                return cached
            store = self._ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "gcs",
                        "bucket": self.bucket,
                        "path": f"{self.path}/{variable}/",
                    },
                    "recheck_cached_metadata": "open",
                    "recheck_cached_data": False,
                },
                context=self._context,
                read=True,
            ).result()
            self._stores[variable] = store
            return store

    def _read_coordinate(self, name: str) -> np.ndarray:
        return np.asarray(self._open_array(name).read().result())

    def _metadata(self, variable: str) -> Mapping:
        with self._array_metadata_lock:
            cached = self._array_metadata.get(variable)
            if cached is None:
                cached = fetch_arco_array_metadata(self.store_url, variable)
                self._array_metadata[variable] = cached
            return cached

    def _dimensions(self, variable: str) -> tuple[str, ...]:
        dimensions = tuple(
            str(value) for value in self._metadata(variable).get("_ARRAY_DIMENSIONS", ())
        )
        if not dimensions:
            raise RuntimeError(
                f"ARCO array {variable!r} lacks _ARRAY_DIMENSIONS metadata."
            )
        return dimensions

    def hour_index(self, valid_time: datetime) -> int:
        """Return the exact hourly array index, refusing off-axis timestamps."""
        seconds = (valid_time - self.time_origin).total_seconds()
        quotient, remainder = divmod(seconds, 3600.0)
        if (
            remainder
            or quotient < 0
            or valid_time < self.valid_time_start
            or valid_time > self.valid_time_stop
        ):
            raise RuntimeError(
                f"UTC time {valid_time.isoformat()} is outside/aligned incorrectly for "
                f"ARCO coverage {self.valid_time_start.isoformat()}.."
                f"{self.valid_time_stop.isoformat()}."
            )
        return int(quotient)

    def _dimension_positions(
        self, variable: str, expected: Sequence[str]
    ) -> tuple[int, ...]:
        labels = self._dimensions(variable)
        if set(labels) != set(expected):
            raise RuntimeError(
                f"Unexpected ARCO dimensions {labels}; expected {tuple(expected)}."
            )
        return tuple(labels.index(name) for name in expected)

    def read_surface(
        self,
        variable: str,
        start: datetime,
        stop: datetime,
    ) -> np.ndarray:
        """Read ``[start, stop)`` for one surface field as time/lat/lon."""
        store = self._open_array(str(variable))
        positions = self._dimension_positions(
            str(variable), ("time", "latitude", "longitude")
        )
        index = [slice(None)] * 3
        index[positions[0]] = slice(self.hour_index(start), self.hour_index(stop))
        raw = np.asarray(store[tuple(index)].read().result(), dtype=np.float32)
        return np.moveaxis(raw, positions, (0, 1, 2))

    def read_pressure(
        self,
        variable: str,
        valid_time: datetime,
        level: int,
    ) -> np.ndarray:
        """Read one pressure-level field as latitude/longitude."""
        store = self._open_array(str(variable))
        labels = self._dimensions(str(variable))
        expected = ("time", "level", "latitude", "longitude")
        if set(labels) != set(expected):
            raise RuntimeError(f"Unexpected ARCO pressure dimensions {labels}.")
        matches = np.flatnonzero(np.isclose(self.levels.astype(float), float(level)))
        if matches.size != 1:
            raise RuntimeError(f"Pressure level {level} is unavailable in ARCO ERA5.")
        index = [slice(None)] * 4
        index[labels.index("time")] = self.hour_index(valid_time)
        index[labels.index("level")] = int(matches[0])
        raw = np.asarray(store[tuple(index)].read().result(), dtype=np.float32)
        remaining = [name for name in labels if name not in ("time", "level")]
        return np.moveaxis(
            raw,
            (remaining.index("latitude"), remaining.index("longitude")),
            (0, 1),
        )


def selected_dates(year: int, months: Sequence[int]) -> tuple[date, ...]:
    """Return every selected UTC day in deterministic calendar order."""
    output = []
    for month in sorted({int(value) for value in months}):
        output.extend(
            date(int(year), month, day)
            for day in range(1, calendar.monthrange(int(year), month)[1] + 1)
        )
    return tuple(output)


def _create_output(path: Path, task: DownloadTask, backend) -> None:
    """Create an empty fixed-shape NetCDF file for one ARCO task."""
    dates = selected_dates(task.year, task.months)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NetCDFDataset(str(path), "w", format="NETCDF4") as output:
        output.setncattr("source", backend.store_url)
        output.setncattr("download_provider", ARCO_PROVIDER)
        output.setncattr("utc_days", "true")
        output.createDimension("valid_time", 1 if task.group == "static" else len(dates))
        output.createDimension("latitude", len(backend.lat))
        output.createDimension("longitude", len(backend.lon))
        time_var = output.createVariable("valid_time", "i4", ("valid_time",))
        labels = [19790101] if task.group == "static" else [
            value.year * 10000 + value.month * 100 + value.day for value in dates
        ]
        time_var[:] = labels
        output.createVariable("latitude", "f8", ("latitude",))[:] = backend.lat
        output.createVariable("longitude", "f8", ("longitude",))[:] = backend.lon

        if task.group in ("pressure_geopotential", "pressure_850"):
            levels = (300, 500) if task.group == "pressure_geopotential" else (850,)
            output.createDimension("pressure_level", len(levels))
            output.createVariable(
                "pressure_level", "i4", ("pressure_level",)
            )[:] = levels

        if task.group in ("daily_tmax", "daily_t2m"):
            variables = ("t2m",)
        elif task.group == "daily_d2m":
            variables = ("d2m",)
        elif task.group == "single_levels":
            variables = ("swvl1", "swvl2", "sst", "msl")
        elif task.group == "pressure_geopotential":
            variables = ("z",)
        elif task.group == "pressure_850":
            variables = ("t", "q", "u", "v")
        elif task.group == "static":
            variables = ("z", "lsm")
        else:
            raise ValueError(f"Unsupported ARCO group {task.group!r}.")

        dimensions = (
            ("valid_time", "pressure_level", "latitude", "longitude")
            if task.group in ("pressure_geopotential", "pressure_850")
            else ("valid_time", "latitude", "longitude")
        )
        chunks = (
            (1, len(output.dimensions["pressure_level"]), len(backend.lat), len(backend.lon))
            if len(dimensions) == 4
            else (1, len(backend.lat), len(backend.lon))
        )
        for name in variables:
            variable = output.createVariable(
                name,
                "f4",
                dimensions,
                zlib=True,
                complevel=1,
                shuffle=True,
                chunksizes=chunks,
                fill_value=np.float32(np.nan),
            )
            variable.units = VARIABLE_UNITS[name]


def _read_day(task: DownloadTask, backend, valid_date: date) -> Mapping[str, np.ndarray]:
    """Read and reduce one UTC day without holding the local HDF5 lock."""
    start = datetime(valid_date.year, valid_date.month, valid_date.day)
    stop = start + timedelta(days=1)
    if task.group == "daily_tmax":
        values = backend.read_surface(SURFACE_VARIABLES["t2m"], start, stop)
        return {"t2m": np.nanmax(values, axis=0)}
    elif task.group == "daily_t2m":
        values = backend.read_surface(SURFACE_VARIABLES["t2m"], start, stop)
        return {"t2m": np.nanmean(values, axis=0)}
    elif task.group == "daily_d2m":
        values = backend.read_surface(SURFACE_VARIABLES["d2m"], start, stop)
        return {"d2m": np.nanmean(values, axis=0)}
    elif task.group == "single_levels":
        payload = {}
        for output_name in ("swvl1", "swvl2", "sst", "msl"):
            values = backend.read_surface(
                SURFACE_VARIABLES[output_name], start, start + timedelta(hours=1)
            )
            payload[output_name] = values[0]
        return payload
    elif task.group == "pressure_geopotential":
        payload = []
        for level in (300, 500):
            payload.append(
                backend.read_pressure(PRESSURE_VARIABLES["z"], start, level)
            )
        return {"z": np.stack(payload, axis=0)}
    elif task.group == "pressure_850":
        payload = {}
        for output_name in ("t", "q", "u", "v"):
            payload[output_name] = backend.read_pressure(
                PRESSURE_VARIABLES[output_name], start, 850
            )
        return payload
    else:
        raise ValueError(f"Unsupported daily ARCO group {task.group!r}.")


def _read_static(backend) -> Mapping[str, np.ndarray]:
    """Read the two static fields outside the serialized NetCDF writer."""
    valid_time = datetime(1979, 1, 1)
    payload = {}
    for output_name in ("z", "lsm"):
        values = backend.read_surface(
            SURFACE_VARIABLES[output_name], valid_time, valid_time + timedelta(hours=1)
        )
        payload[output_name] = values[0]
    return payload


def _write_payload(output, task: DownloadTask, index: int, payload: Mapping) -> None:
    """Commit one already-read day while holding the local HDF5 lock."""
    if task.group == "pressure_geopotential":
        output.variables["z"][index, :, :, :] = payload["z"]
    elif task.group == "pressure_850":
        for output_name in ("t", "q", "u", "v"):
            output.variables[output_name][index, 0, :, :] = payload[output_name]
    else:
        target_index = 0 if task.group == "static" else index
        for output_name, values in payload.items():
            output.variables[output_name][target_index, :, :] = values


def _progress_path(partial: Path) -> Path:
    return partial.with_suffix(partial.suffix + ".progress.json")


def _read_progress(path: Path, task: DownloadTask) -> int:
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if payload.get("task") != _task_record(task):
        return 0
    return int(payload.get("completed_days", 0))


def _write_progress(path: Path, task: DownloadTask, completed_days: int) -> None:
    payload = {
        "task": _task_record(task),
        "download_provider": ARCO_PROVIDER,
        "completed_days": int(completed_days),
    }
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _retry(
    operation: Callable[[], object],
    *,
    task: DownloadTask,
    label: str,
    max_retries: int,
    retry_base_seconds: float,
) -> object:
    for retry_number in range(int(max_retries) + 1):
        try:
            return operation()
        except Exception as exc:
            if retry_number >= int(max_retries):
                raise
            delay = min(
                float(retry_base_seconds) * (2 ** retry_number),
                MAX_RETRY_DELAY_SECONDS,
            )
            print(
                f"ARCO read failed group={task.group} year={task.year} {label}: "
                f"{exc}; retry {retry_number + 1}/{max_retries} in {delay:.0f}s.",
                flush=True,
            )
            time.sleep(delay)


def retrieve_arco_task(
    task: DownloadTask,
    backend,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    progress_every: int = 10,
) -> str:
    """Resume, validate, and atomically publish one ARCO-backed task."""
    if task.group not in SUPPORTED_GROUPS:
        raise ValueError(f"Unsupported ARCO task group {task.group!r}.")
    if task_complete(task):
        return f"exists, skipping: {task.target}"

    target = Path(task.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    progress_path = _progress_path(partial)
    dates = (date(1979, 1, 1),) if task.group == "static" else selected_dates(
        task.year, task.months
    )
    completed = _read_progress(progress_path, task)
    if completed < 0 or completed > len(dates) or (completed and not partial.is_file()):
        completed = 0
    if completed == 0:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        with _NETCDF_WRITE_LOCK:
            _create_output(partial, task, backend)

    for index in range(completed, len(dates)):
        valid_date = dates[index]

        payload = _retry(
            (lambda: _read_static(backend))
            if task.group == "static"
            else (lambda: _read_day(task, backend, valid_date)),
            task=task,
            label=valid_date.isoformat(),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
        )
        with _NETCDF_WRITE_LOCK:
            with NetCDFDataset(str(partial), "r+") as output:
                _write_payload(output, task, index, payload)
                output.sync()
        _write_progress(progress_path, task, index + 1)
        if (index + 1) % int(progress_every) == 0 or index + 1 == len(dates):
            print(
                f"ARCO_PROGRESS group={task.group} year={task.year} "
                f"completed_days={index + 1}/{len(dates)} last_date={valid_date}",
                flush=True,
            )

    # Validation opens the HDF5 file too, so it must not overlap another
    # worker's local NetCDF mutation on the HiPerGator build.
    with _NETCDF_WRITE_LOCK:
        validate_download_file(partial, task)
    validated_bytes = partial.stat().st_size
    partial.replace(target)
    metadata = {
        "task": _task_record(task),
        "target_source": task.source_choice,
        "download_provider": ARCO_PROVIDER,
        "arco_store": backend.store_url,
        "utc_days": True,
        "validated_bytes": int(validated_bytes),
    }
    metadata_path = _metadata_path(target)
    metadata_partial = metadata_path.with_suffix(metadata_path.suffix + ".part")
    metadata_partial.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata_partial.replace(metadata_path)
    progress_path.unlink(missing_ok=True)
    return f"retrieved from Google ARCO-ERA5: {target}"


def run_arco_tasks(
    tasks: Iterable[DownloadTask],
    workers: int,
    *,
    backend=None,
    store_url: str = ARCO_ERA5_STORE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    progress_every: int = 10,
) -> None:
    """Run bounded parallel ARCO tasks and print deterministic progress."""
    if int(workers) < 1:
        raise ValueError("workers must be at least one.")
    selected = tuple(tasks)
    if not selected:
        raise ValueError("No ARCO tasks were selected.")
    reader = backend if backend is not None else TensorStoreArcoBackend(store_url)
    print(
        f"Starting {len(selected)} Google ARCO-ERA5 tasks with {workers} workers; "
        f"store={reader.store_url}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(
                retrieve_arco_task,
                task,
                reader,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                progress_every=progress_every,
            ): task
            for task in selected
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"[{completed}/{len(selected)}] {future.result()}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path(Config.DATA_ROOT))
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--chunking", choices=("yearly", "monthly"), default="yearly")
    parser.add_argument(
        "--groups",
        default=None,
        help=(
            "Comma-separated raw groups to obtain from Google ARCO-ERA5. "
            "Defaults to all base groups, plus daily_d2m when "
            "--enable_heat_index is set."
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument("--store", default=ARCO_ERA5_STORE)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--retry_base_seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS
    )
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--enable_heat_index", action="store_true", default=False)
    parser.add_argument("--manifest_only", action="store_true")
    args = parser.parse_args()

    requested_groups = (
        tuple(value.strip() for value in args.groups.split(",") if value.strip())
        if args.groups is not None
        else (SUPPORTED_GROUPS if args.enable_heat_index else BASE_GROUPS)
    )
    unknown = sorted(set(requested_groups) - set(SUPPORTED_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported ARCO groups: {unknown}.")
    if "daily_d2m" in requested_groups and not args.enable_heat_index:
        raise ValueError("daily_d2m requires --enable_heat_index.")

    raw_root = args.data_root / "raw" / "era5"
    all_tasks = build_download_tasks(
        raw_root,
        parse_years(args.years),
        tuple(int(value) for value in args.months.split(",") if value.strip()),
        target_source="daily_statistics",
        enable_heat_index=args.enable_heat_index,
        chunking=args.chunking,
    )
    manifest = args.data_root / "manifests" / "era5_download_tasks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps([_task_record(task) for task in all_tasks], indent=2),
        encoding="utf-8",
    )
    selected_tasks = tuple(task for task in all_tasks if task.group in requested_groups)
    print(
        f"Wrote {len(all_tasks)} compatible tasks to {manifest}; "
        f"selected {len(selected_tasks)} for Google ARCO-ERA5.",
        flush=True,
    )
    if args.manifest_only:
        return 0
    run_arco_tasks(
        selected_tasks,
        args.workers,
        store_url=args.store,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
