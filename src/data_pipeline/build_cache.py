"""Stream monthly ERA5 downloads into a time-chunked global zarr cache.

Only one UTC day and a bounded 20-day rolling history are resident at once.
``LazyGlobalZarrDataset`` reads metadata in its parent process but opens the
zarr store only inside ``__getitem__`` in each DDP worker.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date
from torch.utils.data import Dataset, get_worker_info

from cfm_mesh_train import Config
from data_pipeline.download_era5 import MONTHS, parse_years
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


def _read_day(dataset, key: str, valid_date: date, *, level: Optional[int] = None, reducer="mean") -> np.ndarray:
    variable = dataset.variables[_variable_name(dataset, key)]
    dimensions = list(variable.dimensions)
    index = [slice(None)] * variable.ndim
    if any(name in dimensions for name in ("valid_time", "time", "date")):
        time_name = next(name for name in ("valid_time", "time", "date") if name in dimensions)
        day_indices = _date_indices(dataset).get(valid_date)
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

    values = np.asarray(variable[tuple(index)], dtype=np.float32)
    lat_name = next(name for name in ("latitude", "lat") if name in dimensions)
    lon_name = next(name for name in ("longitude", "lon") if name in dimensions)
    remaining = [name for position, name in enumerate(dimensions) if not isinstance(index[position], int)]
    values = np.moveaxis(values, (remaining.index(lat_name), remaining.index(lon_name)), (-2, -1))
    while values.ndim > 2:
        values = np.nanmax(values, axis=0) if reducer == "max" else np.nanmean(values, axis=0)
    return values.astype(np.float32, copy=False)


def _read_static(dataset, key: str) -> np.ndarray:
    variable = dataset.variables[_variable_name(dataset, key)]
    values = np.asarray(variable[:], dtype=np.float32)
    dimensions = list(variable.dimensions)
    lat_name = next(name for name in ("latitude", "lat") if name in dimensions)
    lon_name = next(name for name in ("longitude", "lon") if name in dimensions)
    values = np.moveaxis(values, (dimensions.index(lat_name), dimensions.index(lon_name)), (-2, -1))
    while values.ndim > 2:
        values = values[0]
    return values


def _monthly_path(raw_root: Path, group: str, year: int, month: int) -> Path:
    return raw_root / group / str(int(year)) / f"{group}_{int(year):04d}{int(month):02d}.nc"


def _regrid(
    field,
    dataset,
    target: GridSpec,
    method: str,
    weights_dir: Path,
    label: str,
) -> np.ndarray:
    lat, lon = _lat_lon(dataset)
    weights_path = weights_dir / f"{method}_{len(lat)}x{len(lon)}_to_{target.resolution}_{label}.npz"
    return regrid_field(
        field,
        lat,
        lon,
        target,
        method=method,
        weights_path=weights_path if method == "conservative" else None,
    )


def iter_era5_daily_slices(
    raw_root: Path,
    years: Sequence[int],
    months: Sequence[int],
    target: GridSpec,
    weights_dir: Path,
    *,
    target_source: str,
) -> Iterable[DailySlice]:
    """Yield one cache-ready day while retaining at most 20 rolling fields."""
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
    for year in sorted({int(value) for value in years}):
        for month in sorted({int(value) for value in months}):
            with ExitStack() as stack:
                if target_source == "daily_statistics":
                    tmax_ds = stack.enter_context(NetCDFDataset(_monthly_path(raw_root, "daily_tmax", year, month)))
                    t2m_ds = stack.enter_context(NetCDFDataset(_monthly_path(raw_root, "daily_t2m", year, month)))
                    tmax_reducer = "mean"
                    t2m_reducer = "mean"
                elif target_source == "hourly_fallback":
                    hourly_ds = stack.enter_context(NetCDFDataset(_monthly_path(raw_root, "hourly_t2m", year, month)))
                    tmax_ds = hourly_ds
                    t2m_ds = hourly_ds
                    tmax_reducer = "max"
                    t2m_reducer = "mean"
                else:
                    raise ValueError(f"Unknown target_source={target_source!r}.")
                single_ds = stack.enter_context(NetCDFDataset(_monthly_path(raw_root, "single_levels", year, month)))
                geopotential_ds = stack.enter_context(
                    NetCDFDataset(_monthly_path(raw_root, "pressure_geopotential", year, month))
                )
                pressure850_ds = stack.enter_context(NetCDFDataset(_monthly_path(raw_root, "pressure_850", year, month)))

                for valid_date in sorted(_date_indices(tmax_ds)):
                    if valid_date.year != year or valid_date.month != month:
                        continue
                    tmax = _regrid(
                        _read_day(tmax_ds, "t2m", valid_date, reducer=tmax_reducer),
                        tmax_ds, target, "conservative", weights_dir, "tmax",
                    )
                    t2m = _regrid(
                        _read_day(t2m_ds, "t2m", valid_date, reducer=t2m_reducer),
                        t2m_ds, target, "bilinear", weights_dir, "smooth",
                    )
                    smooth = {}
                    for key in ("swvl1", "swvl2", "sst", "mslp"):
                        smooth[key] = _regrid(
                            _read_day(single_ds, key, valid_date),
                            single_ds, target, "bilinear", weights_dir, "smooth",
                        )
                    z500 = _regrid(
                        _read_day(geopotential_ds, "z", valid_date, level=500),
                        geopotential_ds, target, "bilinear", weights_dir, "pressure",
                    )
                    z300 = _regrid(
                        _read_day(geopotential_ds, "z", valid_date, level=300),
                        geopotential_ds, target, "bilinear", weights_dir, "pressure",
                    )
                    pressure850 = {
                        key: _regrid(
                            _read_day(pressure850_ds, key, valid_date, level=850),
                            pressure850_ds, target, "bilinear", weights_dir, "pressure",
                        )
                        for key in ("t", "q", "u", "v")
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
                    yield DailySlice(valid_date=valid_date, fields=fields)


def write_zarr_cache(
    slices: Iterable[DailySlice],
    store_path: Path,
    grid: GridSpec,
    *,
    target_source: str,
) -> dict:
    """Resume an append-only daily cache with ``time=1`` commit markers."""
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
    count = 0
    for item in slices:
        date_label = item.valid_date.year * 10000 + item.valid_date.month * 100 + item.valid_date.day
        if count < committed:
            if existing_times[count] != date_label:
                raise RuntimeError(
                    f"Resume date mismatch at index {count}: cache={existing_times[count]}, source={date_label}."
                )
            count += 1
            continue
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
        opener=None,
        metadata: Optional[Mapping] = None,
    ):
        self.store_path = str(store_path)
        self.init_indices = tuple(int(value) for value in init_indices)
        self.history_days = tuple(int(value) for value in history_days)
        self.prediction_leads = tuple(int(value) for value in prediction_leads)
        self._opener = opener
        self._store = None
        self._store_pid = None
        self.metadata = dict(metadata) if metadata is not None else json.loads(
            metadata_path(Path(store_path)).read_text(encoding="utf-8")
        )
        self.channels = tuple(self.metadata["channels"])
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
        target = np.asarray(data.oindex[target_indices, :, :, self.tmax_channel], dtype=np.float32)
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
            "init_index": init_index,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path(Config.DATA_ROOT))
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resolution", choices=tuple(Config.RESOLUTION_SPECS), default=Config.RESOLUTION)
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--target_source", choices=("daily_statistics", "hourly_fallback"), default="daily_statistics")
    args = parser.parse_args()
    raw_root = args.raw_dir or args.data_root / "raw" / "era5"
    output = args.output or args.data_root / "cache" / f"era5_{args.resolution}.zarr"
    grid = grid_for_resolution(args.resolution)
    slices = iter_era5_daily_slices(
        raw_root,
        parse_years(args.years),
        tuple(int(value) for value in args.months.split(",") if value.strip()),
        grid,
        args.data_root / "regrid_weights",
        target_source=args.target_source,
    )
    metadata = write_zarr_cache(slices, output, grid, target_source=args.target_source)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
