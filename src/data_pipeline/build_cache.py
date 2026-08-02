"""Stream annual or monthly ERA5 downloads into a time-chunked global cache.

Only a bounded queue of UTC days and a 20-day rolling history are resident.
``LazyGlobalZarrDataset`` reads metadata in its parent process but opens the
zarr store only inside ``__getitem__`` in each DDP worker.
"""

from __future__ import annotations

import argparse
import json
import os
import time as monotonic_time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date
from torch.utils.data import Dataset, get_worker_info

from cfm_mesh_train import Config
from data_pipeline.download_era5 import (
    MONTHS,
    download_target_path,
    month_chunks,
    parse_years,
)
from data_pipeline.regrid import GridSpec, grid_for_resolution, regrid_field


CACHE_CHANNELS: Tuple[str, ...] = (
    "tmax",
    "t2m_mean",
    "swvl1",
    "swvl1_trailing20",
    "swvl2_trailing20",
    "sst",
    "sst_valid",
    "z500",
    "z500_low20",
    "mslp",
    "t850",
    "q850",
    "u850",
    "v850",
    "z300",
    "orography",
    "land_mask",
)

VARIABLE_CANDIDATES = {
    "t2m": ("t2m", "2m_temperature"),
    "d2m": ("d2m", "2m_dewpoint_temperature"),
    "swvl1": ("swvl1", "volumetric_soil_water_layer_1"),
    "swvl2": ("swvl2", "volumetric_soil_water_layer_2"),
    "sst": ("sst", "sea_surface_temperature"),
    "mslp": ("msl", "mean_sea_level_pressure"),
    "z": ("z", "geopotential"),
    "t": ("t", "temperature"),
    "q": ("q", "specific_humidity"),
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "lsm": ("lsm", "land_sea_mask"),
}


_CONSERVATIVE_REGRID_LOCK = Lock()


@dataclass(frozen=True)
class DailySlice:
    """One regridded UTC day ready for an append-only cache write."""

    valid_date: date
    fields: Mapping[str, np.ndarray]


def metadata_path(store_path: Path) -> Path:
    """Return the small parent-process-safe cache metadata path."""
    return Path(store_path) / "heatcast_cache_metadata.json"


def fold_sidecar_path(store_path: Path, fold: int, kind: str) -> Path:
    """Return a fold-specific climatology/normalization/threshold sidecar path."""
    allowed = ("climatology", "normalization", "thresholds")
    if str(kind) not in allowed:
        raise ValueError(f"Sidecar kind must be one of {allowed}, got {kind!r}.")
    return Path(store_path).with_suffix(Path(store_path).suffix + ".sidecars") / f"fold{int(fold)}_{kind}.npz"


def _require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr<3 is required to build or read the ERA5 cache.") from exc
    return zarr


def _coordinate_name(dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Missing coordinate; tried {tuple(candidates)} in {dataset.filepath()}.")


def _variable_name(dataset, key: str) -> str:
    for name in VARIABLE_CANDIDATES[key]:
        if name in dataset.variables:
            return name
    raise KeyError(
        f"Missing ERA5 variable {key!r}; tried {VARIABLE_CANDIDATES[key]} in {dataset.filepath()}."
    )


def _lat_lon(dataset) -> Tuple[np.ndarray, np.ndarray]:
    lat_name = _coordinate_name(dataset, ("latitude", "lat"))
    lon_name = _coordinate_name(dataset, ("longitude", "lon"))
    return (
        np.asarray(dataset.variables[lat_name][:], dtype=np.float64),
        np.asarray(dataset.variables[lon_name][:], dtype=np.float64),
    )


def _time_name(dataset) -> str:
    return _coordinate_name(dataset, ("valid_time", "time", "date"))


def _date_indices(dataset) -> Dict[date, Tuple[int, ...]]:
    name = _time_name(dataset)
    variable = dataset.variables[name]
    values = np.asarray(variable[:])
    units = getattr(variable, "units", None)
    calendar = getattr(variable, "calendar", "standard")
    if units:
        decoded = num2date(values, units=units, calendar=calendar)
        dates = [date(int(item.year), int(item.month), int(item.day)) for item in decoded]
    else:
        labels = values.astype(np.int64).ravel()
        dates = [date(int(label) // 10000, (int(label) // 100) % 100, int(label) % 100) for label in labels]
    output: Dict[date, list] = {}
    for index, value in enumerate(dates):
        output.setdefault(value, []).append(index)
    return {key: tuple(value) for key, value in output.items()}


def _read_day(
    dataset,
    key: str,
    valid_date: date,
    *,
    level: Optional[int] = None,
    reducer="mean",
    date_lookup: Optional[Mapping[date, Tuple[int, ...]]] = None,
) -> np.ndarray:
    variable = dataset.variables[_variable_name(dataset, key)]
    dimensions = list(variable.dimensions)
    index = [slice(None)] * variable.ndim
    if any(name in dimensions for name in ("valid_time", "time", "date")):
        time_name = next(name for name in ("valid_time", "time", "date") if name in dimensions)
        day_indices = (date_lookup if date_lookup is not None else _date_indices(dataset)).get(valid_date)
        if not day_indices:
            raise KeyError(f"No {valid_date.isoformat()} values for {key} in {dataset.filepath()}.")
        index[dimensions.index(time_name)] = list(day_indices)
    if level is not None:
        level_name = next((name for name in ("pressure_level", "level", "plev") if name in dimensions), None)
        if level_name is None:
            raise KeyError(f"Variable {key} has no pressure-level dimension in {dataset.filepath()}.")
        level_values = np.asarray(dataset.variables[level_name][:], dtype=np.float64)
        matches = np.flatnonzero(np.isclose(level_values, float(level)))
        if matches.size != 1:
            raise KeyError(f"Pressure level {level} unavailable for {key} in {dataset.filepath()}.")
        index[dimensions.index(level_name)] = int(matches[0])

    raw_values = variable[tuple(index)]
    values = np.asarray(
        np.ma.filled(raw_values, np.nan) if np.ma.isMaskedArray(raw_values) else raw_values,
        dtype=np.float32,
    )
    lat_name = next(name for name in ("latitude", "lat") if name in dimensions)
    lon_name = next(name for name in ("longitude", "lon") if name in dimensions)
    remaining = [name for position, name in enumerate(dimensions) if not isinstance(index[position], int)]
    values = np.moveaxis(values, (remaining.index(lat_name), remaining.index(lon_name)), (-2, -1))
    while values.ndim > 2:
        values = np.nanmax(values, axis=0) if reducer == "max" else np.nanmean(values, axis=0)
    return values.astype(np.float32, copy=False)


def _read_static(dataset, key: str) -> np.ndarray:
    variable = dataset.variables[_variable_name(dataset, key)]
    raw_values = variable[:]
    values = np.asarray(
        np.ma.filled(raw_values, np.nan) if np.ma.isMaskedArray(raw_values) else raw_values,
        dtype=np.float32,
    )
    dimensions = list(variable.dimensions)
    lat_name = next(name for name in ("latitude", "lat") if name in dimensions)
    lon_name = next(name for name in ("longitude", "lon") if name in dimensions)
    values = np.moveaxis(values, (dimensions.index(lat_name), dimensions.index(lon_name)), (-2, -1))
    while values.ndim > 2:
        values = values[0]
    return values


def _regrid(
    field,
    dataset,
    target: GridSpec,
    method: str,
    weights_dir: Path,
    label: str,
) -> np.ndarray:
    lat, lon = _lat_lon(dataset)
    return _regrid_array(field, lat, lon, target, method, weights_dir, label)


def _regrid_array(
    field,
    lat,
    lon,
    target: GridSpec,
    method: str,
    weights_dir: Path,
    label: str,
) -> np.ndarray:
    """Regrid an in-memory field without touching a NetCDF handle."""
    weights_path = weights_dir / f"{method}_{len(lat)}x{len(lon)}_to_{target.resolution}_{label}.npz"
    return regrid_field(
        field,
        lat,
        lon,
        target,
        method=method,
        weights_path=weights_path if method == "conservative" else None,
        # The cache builder processes tens of thousands of daily fields.
        # Reuse the deterministic SciPy path instead of constructing a new
        # xESMF regridder for every field/day. Conservative weights remain
        # cached atomically in the .npz file above.
        prefer_xesmf=False,
    )


def _execute_regrid_specification(specification, target, weights_dir):
    """Execute one in-memory regrid, protecting the shared weight cache."""
    field, lat, lon, method, label = specification
    if method == "conservative":
        # Multiple queued days share one atomic .npz weight-cache path. Keep
        # its first construction and reads single-threaded; the other eleven
        # predictor regrids per day remain fully concurrent.
        with _CONSERVATIVE_REGRID_LOCK:
            return _regrid_array(field, lat, lon, target, method, weights_dir, label)
    return _regrid_array(field, lat, lon, target, method, weights_dir, label)


def _submit_regrid_requests(requests, target, weights_dir, executor):
    """Submit one day's independent in-memory regrids without waiting."""
    return {
        name: executor.submit(
            _execute_regrid_specification, specification, target, weights_dir
        )
        for name, specification in requests.items()
    }


def _resolve_regrid_futures(futures):
    """Resolve a submitted day in deterministic channel order."""
    return {name: future.result() for name, future in futures.items()}


def _execute_regrid_requests(requests, target, weights_dir, executor=None):
    """Execute independent in-memory regrids serially or on a bounded thread pool."""
    if executor is None:
        return {
            name: _execute_regrid_specification(specification, target, weights_dir)
            for name, specification in requests.items()
        }
    return _resolve_regrid_futures(
        _submit_regrid_requests(requests, target, weights_dir, executor)
    )


def _date_from_label(value: int) -> date:
    label = int(value)
    return date(label // 10000, (label // 100) % 100, label % 100)


def cache_resume_context_start(store_path: Path, history_days: int = 20) -> Optional[date]:
    """Return the first context day needed to resume an existing cache."""
    path = Path(store_path)
    if not path.exists():
        return None
    root = _require_zarr().open_group(str(path), mode="r")
    if "time" not in root:
        return None
    values = np.asarray(root["time"][:], dtype=np.int32)
    zero_positions = np.flatnonzero(values == 0)
    committed = int(zero_positions[0]) if zero_positions.size else int(values.size)
    if committed <= 0:
        return None
    return _date_from_label(int(values[committed - 1])) - timedelta(days=max(0, int(history_days) - 2))


def iter_era5_daily_slices(
    raw_root: Path,
    years: Sequence[int],
    months: Sequence[int],
    target: GridSpec,
    weights_dir: Path,
    *,
    target_source: str,
    chunking: str = "yearly",
    workers: int = 1,
    start_date: Optional[date] = None,
) -> Iterable[DailySlice]:
    """Yield cache-ready days with bounded parallel in-memory regridding."""
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("Cache regrid workers must be a positive integer.")
    static_path = raw_root / "static" / "era5_static.nc"
    if not static_path.is_file():
        raise FileNotFoundError(f"Missing ERA5 static download: {static_path}")
    with NetCDFDataset(static_path) as static_ds:
        orography = _regrid(_read_static(static_ds, "z") / 9.80665, static_ds, target, "bilinear", weights_dir, "static")
        land_fraction = _regrid(_read_static(static_ds, "lsm"), static_ds, target, "bilinear", weights_dir, "static")
        land_mask = (land_fraction >= 0.5).astype(np.float32)

    swvl1_history: deque = deque(maxlen=20)
    swvl2_history: deque = deque(maxlen=20)
    z500_history: deque = deque(maxlen=20)
    selected_months = tuple(sorted({int(value) for value in months}))
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    requests_per_day = 12
    pending_day_limit = max(1, (worker_count + requests_per_day - 1) // requests_per_day)

    def complete_day(valid_date, regridded):
        tmax = regridded["tmax"]
        t2m = regridded["t2m"]
        smooth = {
            key: regridded[key]
            for key in ("swvl1", "swvl2", "sst", "mslp")
        }
        # Soil moisture is undefined over ocean. Zero is neutral there
        # because the land mask remains an explicit input.
        for key in ("swvl1", "swvl2"):
            smooth[key] = np.where(
                np.isfinite(smooth[key]), smooth[key], 0.0
            ).astype(np.float32)
        z500 = regridded["z500"]
        z300 = regridded["z300"]
        pressure850 = {
            key: regridded[f"{key}850"] for key in ("t", "q", "u", "v")
        }
        swvl1_history.append(smooth["swvl1"])
        swvl2_history.append(smooth["swvl2"])
        z500_history.append(z500)
        sst_valid = np.isfinite(smooth["sst"]).astype(np.float32)
        fields = {
            "tmax": tmax,
            "t2m_mean": t2m,
            "swvl1": smooth["swvl1"],
            "swvl1_trailing20": np.nanmean(np.stack(tuple(swvl1_history)), axis=0),
            "swvl2_trailing20": np.nanmean(np.stack(tuple(swvl2_history)), axis=0),
            "sst": np.where(np.isfinite(smooth["sst"]), smooth["sst"], 0.0),
            "sst_valid": sst_valid,
            "z500": z500,
            "z500_low20": np.nanmean(np.stack(tuple(z500_history)), axis=0),
            "mslp": smooth["mslp"],
            "t850": pressure850["t"],
            "q850": pressure850["q"],
            "u850": pressure850["u"],
            "v850": pressure850["v"],
            "z300": z300,
            "orography": orography,
            "land_mask": land_mask,
        }
        nonfinite = tuple(
            name for name, value in fields.items()
            if not np.all(np.isfinite(value))
        )
        if nonfinite:
            raise RuntimeError(
                f"Non-finite regridded ERA5 fields on {valid_date}: {nonfinite}."
            )
        return DailySlice(valid_date=valid_date, fields=fields)

    try:
        for year in sorted({int(value) for value in years}):
            if start_date is not None and year < start_date.year:
                continue
            for chunk_months in month_chunks(selected_months, chunking):
                if start_date is not None and year == start_date.year and max(chunk_months) < start_date.month:
                    continue
                with ExitStack() as stack:
                    pending_days: deque = deque()
                    if target_source == "daily_statistics":
                        tmax_ds = stack.enter_context(NetCDFDataset(
                            download_target_path(
                                raw_root, "daily_tmax", year, chunk_months
                            )
                        ))
                        t2m_ds = stack.enter_context(NetCDFDataset(
                            download_target_path(
                                raw_root, "daily_t2m", year, chunk_months
                            )
                        ))
                        tmax_reducer = "mean"
                        t2m_reducer = "mean"
                    elif target_source == "hourly_fallback":
                        hourly_ds = stack.enter_context(NetCDFDataset(
                            download_target_path(
                                raw_root, "hourly_t2m", year, chunk_months
                            )
                        ))
                        tmax_ds = hourly_ds
                        t2m_ds = hourly_ds
                        tmax_reducer = "max"
                        t2m_reducer = "mean"
                    else:
                        raise ValueError(f"Unknown target_source={target_source!r}.")
                    single_ds = stack.enter_context(NetCDFDataset(
                        download_target_path(
                            raw_root, "single_levels", year, chunk_months
                        )
                    ))
                    geopotential_ds = stack.enter_context(
                        NetCDFDataset(download_target_path(
                            raw_root, "pressure_geopotential", year, chunk_months
                        ))
                    )
                    pressure850_ds = stack.enter_context(NetCDFDataset(
                        download_target_path(
                            raw_root, "pressure_850", year, chunk_months
                        )
                    ))
                    coordinates = {
                        "tmax": _lat_lon(tmax_ds),
                        "t2m": _lat_lon(t2m_ds),
                        "single": _lat_lon(single_ds),
                        "geopotential": _lat_lon(geopotential_ds),
                        "pressure850": _lat_lon(pressure850_ds),
                    }
                    date_lookups = {
                        "tmax": _date_indices(tmax_ds),
                        "t2m": _date_indices(t2m_ds),
                        "single": _date_indices(single_ds),
                        "geopotential": _date_indices(geopotential_ds),
                        "pressure850": _date_indices(pressure850_ds),
                    }
                    for valid_date in sorted(date_lookups["tmax"]):
                        if (
                            valid_date.year != year
                            or valid_date.month not in chunk_months
                            or (start_date is not None and valid_date < start_date)
                        ):
                            continue
                        requests = {
                            "tmax": (
                                _read_day(
                                    tmax_ds, "t2m", valid_date, reducer=tmax_reducer,
                                    date_lookup=date_lookups["tmax"],
                                ),
                                *coordinates["tmax"], "conservative", "tmax",
                            ),
                            "t2m": (
                                _read_day(
                                    t2m_ds, "t2m", valid_date, reducer=t2m_reducer,
                                    date_lookup=date_lookups["t2m"],
                                ),
                                *coordinates["t2m"], "bilinear", "smooth",
                            ),
                        }
                        for key in ("swvl1", "swvl2", "sst", "mslp"):
                            requests[key] = (
                                _read_day(
                                    single_ds, key, valid_date,
                                    date_lookup=date_lookups["single"],
                                ),
                                *coordinates["single"], "bilinear", "smooth",
                            )
                        for key, level in (("z500", 500), ("z300", 300)):
                            requests[key] = (
                                _read_day(
                                    geopotential_ds, "z", valid_date, level=level,
                                    date_lookup=date_lookups["geopotential"],
                                ),
                                *coordinates["geopotential"], "bilinear", "pressure",
                            )
                        for key in ("t", "q", "u", "v"):
                            requests[f"{key}850"] = (
                                _read_day(
                                    pressure850_ds, key, valid_date, level=850,
                                    date_lookup=date_lookups["pressure850"],
                                ),
                                *coordinates["pressure850"], "bilinear", "pressure",
                            )
                        if executor is None:
                            yield complete_day(
                                valid_date,
                                _execute_regrid_requests(
                                    requests, target, weights_dir, executor=None
                                ),
                            )
                            continue
                        pending_days.append((
                            valid_date,
                            _submit_regrid_requests(
                                requests, target, weights_dir, executor
                            ),
                        ))
                        if len(pending_days) >= pending_day_limit:
                            pending_date, pending_futures = pending_days.popleft()
                            yield complete_day(
                                pending_date,
                                _resolve_regrid_futures(pending_futures),
                            )
                    while pending_days:
                        pending_date, pending_futures = pending_days.popleft()
                        yield complete_day(
                            pending_date,
                            _resolve_regrid_futures(pending_futures),
                        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def write_zarr_cache(
    slices: Iterable[DailySlice],
    store_path: Path,
    grid: GridSpec,
    *,
    target_source: str,
    progress_every: int = 30,
) -> dict:
    """Resume an append-only daily cache with ``time=1`` commit markers."""
    progress_interval = int(progress_every)
    if progress_interval <= 0:
        raise ValueError("Cache progress interval must be a positive integer.")
    zarr = _require_zarr()
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="a")
    expected_tail = (grid.shape[0], grid.shape[1], len(CACHE_CHANNELS))
    if "data" in root:
        data = root["data"]
        time = root["time"]
        if tuple(data.shape[1:]) != expected_tail:
            raise ValueError(f"Existing cache tail shape {data.shape[1:]} != expected {expected_tail}.")
        committed_values = np.asarray(time[:], dtype=np.int32)
        zero_positions = np.flatnonzero(committed_values == 0)
        committed = int(zero_positions[0]) if zero_positions.size else int(committed_values.size)
        data.resize((committed,) + expected_tail)
        time.resize((committed,))
        existing_times = tuple(int(value) for value in committed_values[:committed])
    else:
        data = root.create_dataset(
            "data",
            shape=(0,) + expected_tail,
            chunks=(1, grid.shape[0], grid.shape[1], len(CACHE_CHANNELS)),
            dtype="f4",
        )
        time = root.create_dataset("time", shape=(0,), chunks=(366,), dtype="i4", fill_value=0)
        root.create_dataset("lat", data=grid.lat.astype(np.float32), chunks=(grid.shape[0],))
        root.create_dataset("lon", data=grid.lon.astype(np.float32), chunks=(grid.shape[1],))
        existing_times = ()
        committed = 0
    if any(right <= left for left, right in zip(existing_times, existing_times[1:])):
        raise RuntimeError("Existing cache dates are not strictly increasing.")
    existing_time_set = set(existing_times)
    count = committed
    new_count = 0
    started = monotonic_time.monotonic()
    print(
        f"CACHE_START committed_days={committed} progress_every={progress_interval}",
        flush=True,
    )
    for item in slices:
        date_label = item.valid_date.year * 10000 + item.valid_date.month * 100 + item.valid_date.day
        if date_label in existing_time_set:
            continue
        if existing_times and date_label <= existing_times[-1]:
            raise RuntimeError(
                f"Resume context produced non-cached date {date_label} before "
                f"last committed date {existing_times[-1]}."
            )
        missing = tuple(name for name in CACHE_CHANNELS if name not in item.fields)
        if missing:
            raise KeyError(f"Daily slice {item.valid_date} is missing channels {missing}.")
        stacked = np.stack([np.asarray(item.fields[name], dtype=np.float32) for name in CACHE_CHANNELS], axis=-1)
        if stacked.shape != grid.shape + (len(CACHE_CHANNELS),):
            raise ValueError(f"Daily slice has shape {stacked.shape}, expected {grid.shape + (len(CACHE_CHANNELS),)}.")
        data.resize((count + 1, grid.shape[0], grid.shape[1], len(CACHE_CHANNELS)))
        time.resize((count + 1,))
        data[count, :, :, :] = stacked
        time[count] = date_label
        count += 1
        new_count += 1
        if new_count == 1 or new_count % progress_interval == 0:
            elapsed = max(monotonic_time.monotonic() - started, 1e-9)
            print(
                "CACHE_PROGRESS "
                f"committed_days={count} new_days={new_count} "
                f"last_date={date_label} rate_days_per_min={60.0 * new_count / elapsed:.2f}",
                flush=True,
            )
    metadata = {
        "schema_version": 1,
        "dimensions": ["time", "lat", "lon", "channel"],
        "shape": [count, grid.shape[0], grid.shape[1], len(CACHE_CHANNELS)],
        "chunks": [1, grid.shape[0], grid.shape[1], len(CACHE_CHANNELS)],
        "channels": list(CACHE_CHANNELS),
        "resolution": grid.resolution,
        "target_source": str(target_source),
        "target_statistic": "daily maximum 2m_temperature",
        "utc_days": True,
        "time_values": [int(value) for value in np.asarray(time[:], dtype=np.int32).tolist()],
    }
    root.attrs.update(metadata)
    metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


class LazyGlobalZarrDataset(Dataset):
    """Worker-local lazy sample reader for time-chunked global zarr data."""

    def __init__(
        self,
        store_path,
        init_indices: Sequence[int],
        *,
        history_days: Sequence[int] = (0, 1, 2),
        prediction_leads: Sequence[int] = tuple(range(15, 29)),
        target_array: str = "tmax",
        opener=None,
        metadata: Optional[Mapping] = None,
    ):
        self.store_path = str(store_path)
        self.init_indices = tuple(int(value) for value in init_indices)
        self.history_days = tuple(int(value) for value in history_days)
        self.prediction_leads = tuple(int(value) for value in prediction_leads)
        self.target_array = str(target_array)
        self._opener = opener
        self._store = None
        self._store_pid = None
        self.metadata = dict(metadata) if metadata is not None else json.loads(
            metadata_path(Path(store_path)).read_text(encoding="utf-8")
        )
        self.channels = tuple(self.metadata["channels"])
        self.time_values = tuple(int(value) for value in self.metadata.get("time_values", ()))
        self.tmax_channel = self.channels.index("tmax")
        n_times = int(self.metadata["shape"][0])
        for index in self.init_indices:
            if index - max(self.history_days) < 0 or index + max(self.prediction_leads) >= n_times:
                raise IndexError(f"Initialization index {index} lacks required history/leads in {n_times} days.")

    def __len__(self):
        return len(self.init_indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_store"] = None
        state["_store_pid"] = None
        return state

    def _ensure_open(self):
        pid = os.getpid()
        if self._store is None or self._store_pid != pid:
            opener = self._opener
            if opener is None:
                zarr = _require_zarr()
                opener = lambda path: zarr.open_group(path, mode="r")
            self._store = opener(self.store_path)
            self._store_pid = pid
        return self._store

    def __getitem__(self, item):
        worker = get_worker_info()
        del worker  # Opening here, rather than in __init__, is the DDP-worker contract.
        root = self._ensure_open()
        data = root["data"]
        init_index = self.init_indices[int(item)]
        history_indices = [init_index - lag for lag in self.history_days]
        target_indices = [init_index + lead for lead in self.prediction_leads]
        context = np.asarray(data.oindex[history_indices, :, :, :], dtype=np.float32)
        if self.target_array == "tmax":
            target = np.asarray(
                data.oindex[target_indices, :, :, self.tmax_channel],
                dtype=np.float32,
            )
        else:
            if self.target_array not in root:
                raise RuntimeError(
                    f"Global target array {self.target_array!r} is missing from {self.store_path}."
                )
            target = np.asarray(root[self.target_array].oindex[target_indices, :, :], dtype=np.float32)
        date_axis = getattr(self, "cache_date_labels", self.time_values)
        if len(date_axis) == int(self.metadata["shape"][0]):
            history_dates = tuple(date_axis[index] for index in history_indices)
            target_dates = tuple(date_axis[index] for index in target_indices)
        else:
            time = root["time"]
            history_dates = tuple(int(value) for value in np.asarray(time.oindex[history_indices]).tolist())
            target_dates = tuple(int(value) for value in np.asarray(time.oindex[target_indices]).tolist())
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "init_index": init_index,
            "history_dates": history_dates,
            "target_dates": target_dates,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path(Config.DATA_ROOT))
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resolution", choices=tuple(Config.RESOLUTION_SPECS), default=Config.RESOLUTION)
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--chunking", choices=("yearly", "monthly"), default="yearly")
    parser.add_argument("--target_source", choices=("daily_statistics", "hourly_fallback"), default="daily_statistics")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=30)
    args = parser.parse_args()
    raw_root = args.raw_dir or args.data_root / "raw" / "era5"
    output = args.output or args.data_root / "cache" / f"era5_{args.resolution}.zarr"
    grid = grid_for_resolution(args.resolution)
    resume_start = cache_resume_context_start(output)
    print(
        f"CACHE_PLAN workers={args.workers} resume_context_start="
        f"{resume_start.isoformat() if resume_start is not None else 'beginning'}",
        flush=True,
    )
    slices = iter_era5_daily_slices(
        raw_root,
        parse_years(args.years),
        tuple(int(value) for value in args.months.split(",") if value.strip()),
        grid,
        args.data_root / "regrid_weights",
        target_source=args.target_source,
        chunking=args.chunking,
        workers=args.workers,
        start_date=resume_start,
    )
    metadata = write_zarr_cache(
        slices,
        output,
        grid,
        target_source=args.target_source,
        progress_every=args.progress_every,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
