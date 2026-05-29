#!/usr/bin/env python3
"""
Download targeted ERA5 global fields and append them to Global_Coarse_Conditions.nc.

This script is intentionally separate from training. It reads the exact MJJAS
dates from VDM_Training_Data_Extended_v2.nc, downloads ERA5 fields on a 1-degree
global grid, and writes one daily global field per training timestep.

Default output:
    /blue/nessie/mostafarezaali/Teleconnection/Global_Coarse_Conditions_Extended.nc

Requirements on HiPerGator:
    - cdsapi installed in torch_b200 or another env
    - a configured ~/.cdsapirc for the Copernicus CDS API

Recommended first check:
    python3 -u download_extend_global_fields.py --dry_run

Recommended run:
    python3 -u download_extend_global_fields.py --preset balanced --download_only --max_workers 4
    python3 -u download_extend_global_fields.py --preset balanced --merge_only --overwrite_output

After this succeeds, update the training/JEPA config to point GLOBAL_DATA_PATH at
the extended file and add the printed era5_* variable names to GLOBAL_VARIABLES.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset as NetCDFDataset, num2date


BASE_DATE = datetime(1981, 5, 1)
DEFAULT_WORK_DIR = "/blue/nessie/mostafarezaali/Teleconnection"
DEFAULT_TRAINING_DATA = os.path.join(DEFAULT_WORK_DIR, "VDM_Training_Data_Extended_v2.nc")
DEFAULT_SOURCE_GLOBAL = os.path.join(DEFAULT_WORK_DIR, "Global_Coarse_Conditions.nc")
DEFAULT_OUTPUT_GLOBAL = os.path.join(DEFAULT_WORK_DIR, "Global_Coarse_Conditions_Extended.nc")
DEFAULT_DOWNLOAD_DIR = os.path.join(DEFAULT_WORK_DIR, "data_cache", "era5_global_downloads")

ERA5_PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
ERA5_SINGLE_DATASET = "reanalysis-era5-single-levels"

PRESSURE_SHORT = {
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "specific_humidity": "q",
}

SINGLE_SHORT = {
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "sea_surface_temperature": "sst",
    "total_column_water_vapour": "tcwv",
    "2m_temperature": "t2m",
}

PRESETS = {
    # Good first expansion: circulation, blocking, thermal structure, moisture,
    # and surface pressure drivers. Specific humidity is restricted to levels
    # where it carries meaningful heatwave/soil-moisture feedback signal.
    "balanced": {
        "pressure": {
            "geopotential": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "10"],
            "u_component_of_wind": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "10"],
            "v_component_of_wind": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "10"],
            "temperature": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "10"],
            "specific_humidity": ["1000", "925", "850", "700", "500"],
        },
        "single": [
            "mean_sea_level_pressure",
            "surface_pressure",
            "sea_surface_temperature",
            "total_column_water_vapour",
            "2m_temperature",
        ],
    },
    # Smaller diagnostic expansion. Useful if you want to see whether global
    # geopotential/blocking fields help before paying the full storage cost.
    "z_only": {
        "pressure": {
            "geopotential": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "10"],
        },
        "single": ["mean_sea_level_pressure", "surface_pressure"],
    },
    # Broad expansion. This is large. Use only if storage and global-encoder
    # capacity are acceptable.
    "broad": {
        "pressure": {
            "geopotential": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "50", "10"],
            "u_component_of_wind": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "50", "10"],
            "v_component_of_wind": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "50", "10"],
            "temperature": ["1000", "925", "850", "700", "500", "300", "250", "200", "100", "50", "10"],
            "specific_humidity": ["1000", "925", "850", "700", "500", "300"],
        },
        "single": [
            "mean_sea_level_pressure",
            "surface_pressure",
            "sea_surface_temperature",
            "total_column_water_vapour",
            "2m_temperature",
        ],
    },
}


def time_to_date(tv: float) -> datetime:
    return BASE_DATE + timedelta(days=float(tv))


def load_training_dates(path: str, start_year: int, end_year: int) -> Tuple[np.ndarray, List[datetime]]:
    with NetCDFDataset(path, "r") as nc:
        time_values = np.array(nc.variables["time"][:], dtype=np.float64)
    dates = [time_to_date(tv) for tv in time_values]
    keep = np.array([start_year <= d.year <= end_year for d in dates], dtype=bool)
    if not np.all(keep):
        print(f"Using {int(keep.sum())}/{len(keep)} timesteps for years {start_year}-{end_year}.")
    return np.where(keep)[0], [d for d, ok in zip(dates, keep) if ok]


def group_dates_by_month(indices: np.ndarray, dates: Sequence[datetime]) -> Dict[Tuple[int, int], List[Tuple[int, datetime]]]:
    grouped: Dict[Tuple[int, int], List[Tuple[int, datetime]]] = defaultdict(list)
    for idx, dt in zip(indices, dates):
        grouped[(dt.year, dt.month)].append((int(idx), dt))
    return dict(grouped)


def unique_days(month_items: Sequence[Tuple[int, datetime]]) -> List[str]:
    days = sorted({dt.day for _, dt in month_items})
    return [f"{day:02d}" for day in days]


def pressure_output_name(cds_var: str, level: str) -> str:
    return f"era5_{PRESSURE_SHORT[cds_var]}_{level}"


def single_output_name(cds_var: str) -> str:
    return f"era5_{SINGLE_SHORT[cds_var]}"


def build_output_names(preset: dict) -> List[str]:
    names = []
    for cds_var, levels in preset["pressure"].items():
        for level in levels:
            names.append(pressure_output_name(cds_var, level))
    for cds_var in preset["single"]:
        names.append(single_output_name(cds_var))
    return names


def grouped_pressure_requests(preset: dict) -> Dict[Tuple[str, ...], List[str]]:
    by_levels: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for var_name, levels in preset["pressure"].items():
        by_levels[tuple(levels)].append(var_name)
    return dict(by_levels)


def request_payload(
    variables: Sequence[str],
    year: int,
    month: int,
    days: Sequence[str],
    times: Sequence[str],
    legacy_cds: bool,
    levels: Sequence[str] | None = None,
) -> dict:
    payload = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": list(days),
        "time": list(times),
        "grid": ["1.0", "1.0"],
    }
    if levels is not None:
        payload["pressure_level"] = list(levels)
    if legacy_cds:
        payload["product_type"] = "reanalysis"
        payload["year"] = f"{year:04d}"
        payload["month"] = f"{month:02d}"
        payload["format"] = "netcdf"
    else:
        payload["data_format"] = "netcdf"
        payload["download_format"] = "unarchived"
    return payload


def request_id(prefix: str, variables: Sequence[str], levels: Sequence[str] | None) -> str:
    raw = json.dumps({"vars": list(variables), "levels": list(levels or [])}, sort_keys=True)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def done_path(target: str) -> str:
    return target + ".done"


def is_complete_netcdf(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with NetCDFDataset(path, "r") as nc:
            _ = list(nc.dimensions.keys())
            _ = list(nc.variables.keys())
        return True
    except Exception:
        return False


def mark_done(target: str) -> None:
    with open(done_path(target), "w", encoding="utf-8") as f:
        f.write(f"completed_utc={datetime.utcnow().isoformat()}Z\n")
        f.write(f"size_bytes={os.path.getsize(target)}\n")


def is_completed_download(target: str) -> bool:
    if not is_complete_netcdf(target):
        return False
    if not os.path.exists(done_path(target)):
        mark_done(target)
    return True


def retrieve_file(dataset: str, payload: dict, target: str, dry_run: bool, overwrite: bool) -> str:
    if os.path.exists(target) and not overwrite and is_completed_download(target):
        print(f"  complete, skipping: {target}", flush=True)
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if dry_run:
        print(f"\nDRY RUN request -> {target}")
        print(f"  dataset: {dataset}")
        print(json.dumps(payload, indent=2))
        return target
    import cdsapi

    tmp_target = f"{target}.part.{os.getpid()}"
    if os.path.exists(tmp_target):
        os.remove(tmp_target)

    print(f"  downloading: {target}", flush=True)
    client = cdsapi.Client()
    client.retrieve(dataset, payload, tmp_target)
    if not is_complete_netcdf(tmp_target):
        if os.path.exists(tmp_target):
            os.remove(tmp_target)
        raise RuntimeError(f"Downloaded file is not a readable NetCDF: {target}")

    os.replace(tmp_target, target)
    mark_done(target)
    print(f"  complete: {target}", flush=True)
    return target


def build_download_tasks(args, preset: dict, month_groups: Dict[Tuple[int, int], List[Tuple[int, datetime]]]) -> List[tuple]:
    tasks: List[tuple] = []
    pressure_groups = grouped_pressure_requests(preset)
    for (year, month), items in sorted(month_groups.items()):
        days = unique_days(items)
        for levels, variables in pressure_groups.items():
            rid = request_id("pressure", variables, levels)
            target = os.path.join(args.download_dir, f"{year:04d}{month:02d}_{rid}.nc")
            payload = request_payload(variables, year, month, days, args.times, args.legacy_cds, levels=levels)
            tasks.append((ERA5_PRESSURE_DATASET, payload, target))

        if preset["single"]:
            rid = request_id("single", preset["single"], None)
            target = os.path.join(args.download_dir, f"{year:04d}{month:02d}_{rid}.nc")
            payload = request_payload(preset["single"], year, month, days, args.times, args.legacy_cds, levels=None)
            tasks.append((ERA5_SINGLE_DATASET, payload, target))
    return tasks


def download_monthly_files(args, preset: dict, month_groups: Dict[Tuple[int, int], List[Tuple[int, datetime]]]) -> List[str]:
    tasks = build_download_tasks(args, preset, month_groups)
    targets = [target for _, _, target in tasks]
    if args.dry_run or args.max_workers <= 1:
        for dataset, payload, target in tasks:
            retrieve_file(dataset, payload, target, args.dry_run, args.overwrite_downloads)
        return targets

    print(f"Starting parallel downloads: {len(tasks)} requests, max_workers={args.max_workers}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(retrieve_file, dataset, payload, target, False, args.overwrite_downloads)
            for dataset, payload, target in tasks
        ]
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"  download progress: {completed}/{len(tasks)}", flush=True)
    return targets


def list_completed_downloads(download_dir: str) -> List[str]:
    if not os.path.isdir(download_dir):
        raise FileNotFoundError(download_dir)
    paths = sorted(
        os.path.join(download_dir, name)
        for name in os.listdir(download_dir)
        if name.endswith(".nc")
    )
    completed = []
    skipped = []
    for path in paths:
        if is_completed_download(path):
            completed.append(path)
        else:
            skipped.append(path)
    if skipped:
        print("Skipping incomplete or unreadable downloaded files:")
        for path in skipped:
            print(f"  {path}")
    return completed


def find_name(nc: NetCDFDataset, candidates: Iterable[str]) -> str:
    lower_map = {name.lower(): name for name in nc.variables.keys()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of {list(candidates)} in {list(nc.variables.keys())[:30]}...")


def find_variable(nc: NetCDFDataset, long_name: str, short_name: str) -> str:
    candidates = [short_name, long_name, long_name.replace("_", " ")]
    lower_map = {name.lower(): name for name in nc.variables.keys()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for name, var in nc.variables.items():
        attrs = [getattr(var, "long_name", ""), getattr(var, "standard_name", "")]
        if any(long_name.replace("_", " ") in str(attr).lower() for attr in attrs):
            return name
    raise KeyError(f"Could not find ERA5 variable {long_name} ({short_name}) in {list(nc.variables.keys())}")


def axis_index(dims: Sequence[str], candidates: Iterable[str]) -> int:
    lower = [d.lower() for d in dims]
    for cand in candidates:
        if cand.lower() in lower:
            return lower.index(cand.lower())
    for i, dim in enumerate(lower):
        for cand in candidates:
            if cand.lower() in dim:
                return i
    raise KeyError(f"Could not find axis {list(candidates)} in dimensions {dims}")


def as_float_array(var) -> np.ndarray:
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.array(arr, dtype=np.float32)


def squeeze_extra_axes(arr: np.ndarray, dims: List[str], keep_axes: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
    keep = set(keep_axes)
    for axis in reversed(range(len(dims))):
        if axis in keep:
            continue
        if arr.shape[axis] == 1:
            arr = np.take(arr, 0, axis=axis)
        else:
            arr = np.nanmean(arr, axis=axis)
        dims.pop(axis)
        keep = {a - 1 if a > axis else a for a in keep}
    return arr, dims


def read_dates_from_download(nc: NetCDFDataset) -> List[datetime.date]:
    time_name = find_name(nc, ["time", "valid_time"])
    time_var = nc.variables[time_name]
    times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
    return [datetime(int(t.year), int(t.month), int(t.day), int(getattr(t, "hour", 0))).date() for t in times]


def daily_rows(arr: np.ndarray, dates: Sequence[datetime.date], daily_mean: bool) -> Dict[datetime.date, np.ndarray]:
    by_date: Dict[datetime.date, List[int]] = defaultdict(list)
    for row, dt in enumerate(dates):
        by_date[dt].append(row)
    out = {}
    for dt, rows in by_date.items():
        if len(rows) > 1 and not daily_mean:
            raise ValueError(
                f"Downloaded file has {len(rows)} rows for {dt}. "
                "Pass --daily_mean when requesting multiple synoptic times."
            )
        out[dt] = np.nanmean(arr[rows], axis=0).astype(np.float32)
    return out


def find_pressure_level_index(level_values: np.ndarray, requested_level: str, nc_path: str) -> int:
    """Match ERA5 pressure levels robustly whether NetCDF stores 1000 or 1000.0."""
    try:
        numeric_levels = np.array(level_values, dtype=np.float64)
        requested = float(requested_level)
        matches = np.where(np.isclose(numeric_levels, requested, rtol=0.0, atol=1e-4))[0]
        if len(matches) > 0:
            return int(matches[0])
    except (TypeError, ValueError):
        pass

    string_levels = np.array(level_values).astype(str)
    requested_clean = str(requested_level).rstrip("0").rstrip(".")
    cleaned = np.array([value.rstrip("0").rstrip(".") for value in string_levels])
    matches = np.where(cleaned == requested_clean)[0]
    if len(matches) > 0:
        return int(matches[0])

    raise KeyError(
        f"Level {requested_level} not found in {nc_path}; "
        f"available={string_levels.tolist()}"
    )


def output_coord_dims(dst: NetCDFDataset, n_time: int) -> Tuple[str, str, str]:
    time_dim = None
    lat_dim = None
    lon_dim = None
    for name, dim in dst.dimensions.items():
        lname = name.lower()
        if lname in {"time", "valid_time"} and len(dim) == n_time:
            time_dim = name
        elif lname in {"lat", "latitude"} or "lat" in lname:
            lat_dim = name
        elif lname in {"lon", "longitude"} or "lon" in lname:
            lon_dim = name
    if time_dim is None:
        time_dim = next((name for name, dim in dst.dimensions.items() if len(dim) == n_time), None)
    if lat_dim is None:
        lat_dim = next((name for name, dim in dst.dimensions.items() if len(dim) == 181), None)
    if lon_dim is None:
        lon_dim = next((name for name, dim in dst.dimensions.items() if len(dim) == 360), None)
    if None in (time_dim, lat_dim, lon_dim):
        raise ValueError(f"Could not infer output dims. Available: {list(dst.dimensions.keys())}")
    return time_dim, lat_dim, lon_dim


def create_or_get_output_var(dst: NetCDFDataset, name: str, dims: Tuple[str, str, str], compression: int):
    if name in dst.variables:
        return dst.variables[name]
    kwargs = {}
    if compression > 0:
        kwargs.update({"zlib": True, "complevel": int(compression), "shuffle": True})
    var = dst.createVariable(
        name,
        "f4",
        dims,
        fill_value=np.float32(np.nan),
        chunksizes=(1, 181, 360),
        **kwargs,
    )
    var.units = "ERA5 native units"
    var.grid = "1.0 degree global, latitude descending if inherited from ERA5/CDS"
    return var


def ensure_output_file(source: str, output: str, overwrite: bool) -> None:
    if os.path.abspath(source) == os.path.abspath(output):
        raise ValueError("Refusing to append in-place. Use a separate --output file.")
    if os.path.exists(output):
        if not overwrite:
            print(f"Using existing output file: {output}")
            return
        os.remove(output)
    print(f"Copying coarse database:\n  from {source}\n  to   {output}")
    shutil.copy2(source, output)


def write_pressure_file(
    nc_path: str,
    dst: NetCDFDataset,
    preset: dict,
    output_dims: Tuple[str, str, str],
    date_to_index: Dict[datetime.date, int],
    daily_mean: bool,
    compression: int,
) -> int:
    writes = 0
    with NetCDFDataset(nc_path, "r") as nc:
        level_name = find_name(nc, ["pressure_level", "level", "isobaricinhpa"])
        levels = np.array(nc.variables[level_name][:])
        dates = read_dates_from_download(nc)

        for cds_var, requested_levels in preset["pressure"].items():
            try:
                nc_var_name = find_variable(nc, cds_var, PRESSURE_SHORT[cds_var])
            except KeyError:
                continue
            src = nc.variables[nc_var_name]
            arr = as_float_array(src)
            dims = list(src.dimensions)
            t_axis = axis_index(dims, ["time", "valid_time"])
            lev_axis = axis_index(dims, ["pressure_level", "level", "isobaricinhpa"])
            lat_axis = axis_index(dims, ["latitude", "lat"])
            lon_axis = axis_index(dims, ["longitude", "lon"])
            arr, dims = squeeze_extra_axes(arr, dims, [t_axis, lev_axis, lat_axis, lon_axis])
            t_axis = axis_index(dims, ["time", "valid_time"])
            lev_axis = axis_index(dims, ["pressure_level", "level", "isobaricinhpa"])
            lat_axis = axis_index(dims, ["latitude", "lat"])
            lon_axis = axis_index(dims, ["longitude", "lon"])
            arr = np.transpose(arr, (t_axis, lev_axis, lat_axis, lon_axis))

            for level in requested_levels:
                level_idx = find_pressure_level_index(levels, level, nc_path)
                field_by_date = daily_rows(arr[:, level_idx, :, :], dates, daily_mean)
                out_name = pressure_output_name(cds_var, level)
                out_var = create_or_get_output_var(dst, out_name, output_dims, compression)
                for dt, field in field_by_date.items():
                    if dt not in date_to_index:
                        continue
                    out_var[date_to_index[dt], :, :] = field
                    writes += 1
    return writes


def write_single_file(
    nc_path: str,
    dst: NetCDFDataset,
    preset: dict,
    output_dims: Tuple[str, str, str],
    date_to_index: Dict[datetime.date, int],
    daily_mean: bool,
    compression: int,
) -> int:
    writes = 0
    with NetCDFDataset(nc_path, "r") as nc:
        dates = read_dates_from_download(nc)
        for cds_var in preset["single"]:
            try:
                nc_var_name = find_variable(nc, cds_var, SINGLE_SHORT[cds_var])
            except KeyError:
                continue
            src = nc.variables[nc_var_name]
            arr = as_float_array(src)
            dims = list(src.dimensions)
            t_axis = axis_index(dims, ["time", "valid_time"])
            lat_axis = axis_index(dims, ["latitude", "lat"])
            lon_axis = axis_index(dims, ["longitude", "lon"])
            arr, dims = squeeze_extra_axes(arr, dims, [t_axis, lat_axis, lon_axis])
            t_axis = axis_index(dims, ["time", "valid_time"])
            lat_axis = axis_index(dims, ["latitude", "lat"])
            lon_axis = axis_index(dims, ["longitude", "lon"])
            arr = np.transpose(arr, (t_axis, lat_axis, lon_axis))
            field_by_date = daily_rows(arr, dates, daily_mean)
            out_name = single_output_name(cds_var)
            out_var = create_or_get_output_var(dst, out_name, output_dims, compression)
            for dt, field in field_by_date.items():
                if dt not in date_to_index:
                    continue
                out_var[date_to_index[dt], :, :] = field
                writes += 1
    return writes


def merge_downloads(args, preset: dict, targets: Sequence[str], indices: np.ndarray, dates: Sequence[datetime]) -> None:
    ensure_output_file(args.source_global, args.output, args.overwrite_output)
    date_to_index = {dt.date(): int(idx) for idx, dt in zip(indices, dates)}
    with NetCDFDataset(args.output, "a") as dst:
        output_dims = output_coord_dims(dst, n_time=max(date_to_index.values()) + 1)
        total_writes = 0
        for nc_path in targets:
            if not os.path.exists(nc_path):
                raise FileNotFoundError(f"Missing downloaded file: {nc_path}")
            with NetCDFDataset(nc_path, "r") as probe:
                has_level = any(name.lower() in {"pressure_level", "level", "isobaricinhpa"} for name in probe.variables)
            print(f"  merging: {os.path.basename(nc_path)}")
            if has_level:
                total_writes += write_pressure_file(
                    nc_path, dst, preset, output_dims, date_to_index, args.daily_mean, args.compression
                )
            else:
                total_writes += write_single_file(
                    nc_path, dst, preset, output_dims, date_to_index, args.daily_mean, args.compression
                )
        dst.history = getattr(dst, "history", "") + (
            f"\nAppended ERA5 fields with download_extend_global_fields.py on {datetime.utcnow().isoformat()}Z"
        )
    print(f"Merged {total_writes} daily fields into {args.output}")


def write_variable_report(args, output_names: Sequence[str]) -> str:
    report_path = os.path.join(os.path.dirname(args.output), "data_cache", "extended_global_variables.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Add these names to Config.GLOBAL_VARIABLES after existing fields.\n")
        f.write("# Then set Config.GLOBAL_DATA_PATH to the extended NetCDF file.\n\n")
        f.write(f"GLOBAL_DATA_PATH = {args.output!r}\n\n")
        f.write("NEW_GLOBAL_VARIABLES = [\n")
        for name in output_names:
            f.write(f"    {name!r},\n")
        f.write("]\n")
    return report_path


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_data", default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--source_global", default=DEFAULT_SOURCE_GLOBAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_GLOBAL)
    parser.add_argument("--download_dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="balanced")
    parser.add_argument("--start_year", type=int, default=1981)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--times", type=parse_csv, default=["00:00"], help="Comma-separated synoptic times.")
    parser.add_argument("--daily_mean", action="store_true", help="Average multiple synoptic times into one daily field.")
    parser.add_argument("--legacy_cds", action="store_true", help="Use older CDS request key format='netcdf'.")
    parser.add_argument("--dry_run", action="store_true", help="Print CDS requests and do not download or merge.")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true", help="Skip download and merge files already in download_dir.")
    parser.add_argument("--overwrite_downloads", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Parallel CDS download workers. Existing completed NetCDFs are never re-downloaded unless --overwrite_downloads is set.",
    )
    parser.add_argument("--compression", type=int, default=2, help="NetCDF zlib compression level, 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.times) > 1 and not args.daily_mean:
        raise ValueError("Multiple --times require --daily_mean so output still aligns to daily training timesteps.")
    if not os.path.exists(args.training_data):
        raise FileNotFoundError(args.training_data)
    if not os.path.exists(args.source_global):
        raise FileNotFoundError(args.source_global)

    preset = PRESETS[args.preset]
    output_names = build_output_names(preset)
    print("=" * 88)
    print("ERA5 global-field extension")
    print("=" * 88)
    print(f"Preset:          {args.preset}")
    print(f"Training data:   {args.training_data}")
    print(f"Source coarse:   {args.source_global}")
    print(f"Output coarse:   {args.output}")
    print(f"Download dir:    {args.download_dir}")
    print(f"Times:           {args.times}  daily_mean={args.daily_mean}")
    print(f"Max workers:     {args.max_workers}")
    print(f"New variables:   {len(output_names)}")
    print(", ".join(output_names))

    indices, dates = load_training_dates(args.training_data, args.start_year, args.end_year)
    month_groups = group_dates_by_month(indices, dates)
    expected_months = sum(1 for y in range(args.start_year, args.end_year + 1) for m in range(5, 10))
    print(f"Training timesteps: {len(indices)}")
    print(f"Monthly request groups: {len(month_groups)} (MJJAS full range would be {expected_months})")

    targets: List[str]
    if args.merge_only:
        targets = list_completed_downloads(args.download_dir)
    else:
        targets = download_monthly_files(args, preset, month_groups)

    report_path = write_variable_report(args, output_names)
    print(f"Variable report written to {report_path}")

    if args.dry_run:
        print("\nDry run complete. No files were downloaded or merged.")
        return
    if args.download_only:
        print("\nDownload complete. Merge was skipped because --download_only was set.")
        return

    merge_downloads(args, preset, targets, indices, dates)
    print("\nDone.")
    print("Next steps:")
    print(f"  1. Inspect {args.output}")
    print("  2. Add the printed era5_* names to GLOBAL_VARIABLES")
    print("  3. Set NUM_GLOBAL_CHANNELS = len(GLOBAL_VARIABLES)")
    print("  4. Delete stale global caches before training:")
    print("     rm -rf /blue/nessie/mostafarezaali/Teleconnection/data_cache_sub30_40_-105_-90/global")
    print("     rm -rf /dev/shm/cfm_cache*")


if __name__ == "__main__":
    main()
