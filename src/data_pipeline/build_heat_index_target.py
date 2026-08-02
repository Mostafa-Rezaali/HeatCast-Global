#!/usr/bin/env python3
"""Stream ERA5 Tmax/dewpoint into the primary global heat-index target.

Heat index is calculated on the native ERA5 grid with the exact HeatIndex
NOAA/Steadman implementation, conservatively regridded, and written as a
separate time-chunked array in the existing predictor zarr store. The process
is resume-safe and never materializes the full target in CPU memory.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Iterable, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from data_pipeline.build_cache import (
    _date_indices,
    _lat_lon,
    _read_day,
    _require_zarr,
    metadata_path,
)
from data_pipeline.download_era5 import (
    MONTHS,
    download_target_path,
    month_chunks,
    parse_years,
)
from data_pipeline.regrid import grid_for_resolution, regrid_field
from heat_index import heat_index_from_tmax_dewpoint_c


TARGET_ARRAY = "heat_index"
COMPLETE_ARRAY = "heat_index_complete"


def _temperature_to_celsius(values, units: str, label: str) -> np.ndarray:
    """Convert an ERA5 temperature field to Celsius with explicit unit checks."""
    array = np.asarray(values, dtype=np.float32)
    normalized = str(units).strip().lower()
    if normalized in ("k", "kelvin", "degrees_k", "degree_k"):
        return array - np.float32(273.15)
    if normalized in ("c", "degc", "celsius", "degrees_celsius", "degree_celsius"):
        return array
    raise RuntimeError(f"Unsupported {label} units {units!r}; expected Kelvin or Celsius.")


def _variable_units(dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in dataset.variables:
            units = getattr(dataset.variables[name], "units", None)
            if units:
                return str(units)
    raise KeyError(f"No variable among {tuple(candidates)} with units in {dataset.filepath()}.")


def iter_native_heat_index(
    raw_root: Path,
    years: Sequence[int],
    months: Sequence[int],
    *,
    chunking: str,
) -> Iterable[Tuple[date, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield date, native HI(C), latitude, and longitude from bounded files."""
    selected_months = tuple(sorted({int(value) for value in months}))
    for year in sorted({int(value) for value in years}):
        for selected in month_chunks(selected_months, chunking):
            with ExitStack() as stack:
                tmax_ds = stack.enter_context(NetCDFDataset(
                    download_target_path(raw_root, "daily_tmax", year, selected)
                ))
                dewpoint_ds = stack.enter_context(NetCDFDataset(
                    download_target_path(raw_root, "daily_d2m", year, selected)
                ))
                tmax_dates = _date_indices(tmax_ds)
                dewpoint_dates = _date_indices(dewpoint_ds)
                if set(tmax_dates) != set(dewpoint_dates):
                    raise RuntimeError(
                        f"Tmax/dewpoint date mismatch for {year} months {selected}: "
                        f"tmax_only={sorted(set(tmax_dates) - set(dewpoint_dates))[:5]}, "
                        f"dewpoint_only={sorted(set(dewpoint_dates) - set(tmax_dates))[:5]}."
                    )
                tmax_units = _variable_units(tmax_ds, ("t2m", "2m_temperature"))
                dewpoint_units = _variable_units(
                    dewpoint_ds,
                    ("d2m", "2m_dewpoint_temperature"),
                )
                tmax_lat, tmax_lon = _lat_lon(tmax_ds)
                d2m_lat, d2m_lon = _lat_lon(dewpoint_ds)
                if not np.array_equal(tmax_lat, d2m_lat) or not np.array_equal(tmax_lon, d2m_lon):
                    raise RuntimeError("Native ERA5 Tmax and dewpoint grids do not match.")
                for valid_date in sorted(tmax_dates):
                    tmax = _temperature_to_celsius(
                        _read_day(
                            tmax_ds,
                            "t2m",
                            valid_date,
                            reducer="mean",
                            date_lookup=tmax_dates,
                        ),
                        tmax_units,
                        "Tmax",
                    )
                    dewpoint = _temperature_to_celsius(
                        _read_day(
                            dewpoint_ds,
                            "d2m",
                            valid_date,
                            reducer="mean",
                            date_lookup=dewpoint_dates,
                        ),
                        dewpoint_units,
                        "daily-mean dewpoint",
                    )
                    heat_index = heat_index_from_tmax_dewpoint_c(tmax, dewpoint).astype(np.float32)
                    if not np.all(np.isfinite(heat_index)):
                        raise RuntimeError(f"Non-finite native heat index on {valid_date}.")
                    yield valid_date, heat_index, tmax_lat, tmax_lon


def build_heat_index_target(
    store_path: Path,
    raw_root: Path,
    years: Sequence[int],
    months: Sequence[int],
    *,
    resolution: str,
    chunking: str,
    workers: int,
    progress_every: int,
    weights_dir: Path,
) -> dict:
    """Build or resume the separate heat-index target array in one zarr store."""
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be positive.")
    progress_interval = int(progress_every)
    if progress_interval <= 0:
        raise ValueError("progress_every must be positive.")
    zarr = _require_zarr()
    root = zarr.open_group(str(store_path), mode="a")
    labels = np.asarray(root["time"][:], dtype=np.int32)
    grid = grid_for_resolution(resolution)
    if tuple(root["data"].shape[1:3]) != grid.shape:
        raise RuntimeError(
            f"Predictor cache grid {root['data'].shape[1:3]} does not match {grid.shape}."
        )
    if TARGET_ARRAY in root:
        target = root[TARGET_ARRAY]
        complete = root[COMPLETE_ARRAY]
        if tuple(target.shape) != (len(labels),) + grid.shape:
            raise RuntimeError(f"Existing heat-index shape is invalid: {target.shape}.")
    else:
        target = root.create_dataset(
            TARGET_ARRAY,
            shape=(len(labels),) + grid.shape,
            chunks=(1,) + grid.shape,
            dtype="f4",
            fill_value=np.nan,
        )
        complete = root.create_dataset(
            COMPLETE_ARRAY,
            shape=(len(labels),),
            chunks=(366,),
            dtype="u1",
            fill_value=0,
        )
    label_to_index = {int(value): index for index, value in enumerate(labels)}
    weights_path = Path(weights_dir) / f"conservative_721x1440_to_{resolution}_tmax.npz"
    weights_lock = Lock()

    def regrid_item(item):
        valid_date, native, source_lat, source_lon = item
        if weights_path.exists():
            result = regrid_field(
                native,
                source_lat,
                source_lon,
                grid,
                method="conservative",
                weights_path=weights_path,
                prefer_xesmf=False,
            )
        else:
            with weights_lock:
                result = regrid_field(
                    native,
                    source_lat,
                    source_lon,
                    grid,
                    method="conservative",
                    weights_path=weights_path,
                    prefer_xesmf=False,
                )
        return valid_date, result

    selected_labels = set()
    pending = deque()
    built = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for item in iter_native_heat_index(
            raw_root,
            years,
            months,
            chunking=chunking,
        ):
            valid_date = item[0]
            label = valid_date.year * 10000 + valid_date.month * 100 + valid_date.day
            if label not in label_to_index:
                raise RuntimeError(f"Heat-index date {label} is absent from predictor cache.")
            selected_labels.add(label)
            index = label_to_index[label]
            if int(complete[index]) == 1:
                continue
            pending.append(executor.submit(regrid_item, item))
            if len(pending) < max(1, 2 * worker_count):
                continue
            future = pending.popleft()
            completed_date, field = future.result()
            completed_label = (
                completed_date.year * 10000 + completed_date.month * 100 + completed_date.day
            )
            completed_index = label_to_index[completed_label]
            target[completed_index, :, :] = np.asarray(field, dtype=np.float32)
            complete[completed_index] = 1
            built += 1
            if built == 1 or built % progress_interval == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    "HEAT_INDEX_PROGRESS "
                    f"new_days={built} last_date={completed_label} "
                    f"rate_days_per_min={60.0 * built / elapsed:.2f}",
                    flush=True,
                )
        while pending:
            completed_date, field = pending.popleft().result()
            completed_label = (
                completed_date.year * 10000 + completed_date.month * 100 + completed_date.day
            )
            completed_index = label_to_index[completed_label]
            target[completed_index, :, :] = np.asarray(field, dtype=np.float32)
            complete[completed_index] = 1
            built += 1

    incomplete = [label for label in sorted(selected_labels) if int(complete[label_to_index[label]]) != 1]
    if incomplete:
        raise RuntimeError(f"Heat-index target remains incomplete: {incomplete[:10]}.")
    root.attrs.update({
        "primary_target": TARGET_ARRAY,
        "heat_index_units": "degrees_Celsius",
        "heat_index_temperature": "ERA5 daily maximum 2m_temperature",
        "heat_index_humidity": "RH derived from ERA5 daily-mean 2m_dewpoint_temperature at Tmax",
        "heat_index_formula": "NOAA/Steadman simple expression plus Rothfusz regression and adjustments",
        "heat_index_regridding": "computed on native ERA5 grid then first-order conservative",
    })
    metadata_file = metadata_path(Path(store_path))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata.update({key: root.attrs[key] for key in (
        "primary_target",
        "heat_index_units",
        "heat_index_temperature",
        "heat_index_humidity",
        "heat_index_formula",
        "heat_index_regridding",
    )})
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "store": str(Path(store_path).resolve()),
        "target": TARGET_ARRAY,
        "shape": list(target.shape),
        "completed_days": int(np.count_nonzero(np.asarray(complete[:], dtype=np.uint8))),
        "new_days": int(built),
        "first_date": int(labels[0]),
        "last_date": int(labels[-1]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--resolution", choices=("1.5deg", "0.25deg"), default="1.5deg")
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--chunking", choices=("yearly", "monthly"), default="yearly")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=30)
    args = parser.parse_args(argv)
    months = tuple(int(value) for value in str(args.months).split(",") if value.strip())
    result = build_heat_index_target(
        args.data_root / "cache" / f"era5_{args.resolution}.zarr",
        args.data_root / "raw" / "era5",
        parse_years(args.years),
        months,
        resolution=args.resolution,
        chunking=args.chunking,
        workers=args.workers,
        progress_every=args.progress_every,
        weights_dir=args.data_root / "regrid_weights",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"GLOBAL HEAT-INDEX TARGET COMPLETE: {result['store']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
