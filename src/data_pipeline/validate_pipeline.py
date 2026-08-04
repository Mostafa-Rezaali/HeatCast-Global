"""Validate downloaded ERA5 tasks and the completed global Zarr cache.

The raw audit replays the deterministic download manifest without contacting
CDS.  The cache audit checks the complete date/grid/schema contract and reads
only a bounded set of representative daily slices, preserving the project's
lazy-I/O requirement.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from cfm_mesh_train import Config
from data_pipeline.build_cache import CACHE_CHANNELS, _require_zarr, metadata_path
from data_pipeline.download_era5 import (
    MONTHS,
    DownloadTask,
    parse_years,
    task_complete,
)
from data_pipeline.regrid import GridSpec, grid_for_resolution


@dataclass(frozen=True)
class RawValidationSummary:
    """Compact result of a deterministic raw-download manifest audit."""

    task_count: int
    total_bytes: int
    group_counts: Mapping[str, int]
    partial_files: tuple[str, ...]


@dataclass(frozen=True)
class CacheValidationSummary:
    """Compact result of a bounded Zarr schema and content audit."""

    store: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    first_date: int
    last_date: int
    sampled_indices: tuple[int, ...]


def _task_from_record(record: Mapping) -> DownloadTask:
    """Reconstruct one immutable task from its JSON-stable manifest record."""
    required = {
        "group", "year", "months", "dataset", "request", "target", "source_choice"
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Download manifest record is missing fields {missing}.")
    return DownloadTask(
        group=str(record["group"]),
        year=int(record["year"]),
        months=tuple(int(value) for value in record["months"]),
        dataset=None if record["dataset"] is None else str(record["dataset"]),
        request=dict(record["request"]),
        target=str(record["target"]),
        source_choice=str(record["source_choice"]),
    )


def validate_raw_manifest(
    manifest_path: Path,
    *,
    expected_task_count: Optional[int] = None,
) -> RawValidationSummary:
    """Validate every task, metadata sidecar, and NetCDF header in a manifest."""
    path = Path(manifest_path)
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read ERA5 download manifest {path}: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"ERA5 download manifest {path} is empty or not a list.")
    tasks = tuple(_task_from_record(record) for record in records)
    if expected_task_count is not None and len(tasks) != int(expected_task_count):
        raise RuntimeError(
            f"ERA5 manifest has {len(tasks)} tasks; expected {int(expected_task_count)}."
        )
    targets = tuple(Path(task.target) for task in tasks)
    if len(set(targets)) != len(targets):
        raise RuntimeError("ERA5 download manifest contains duplicate target paths.")

    # The audit, unlike a downloader resume, must reopen every NetCDF header.
    incomplete = [str(task.target) for task in tasks if not task_complete(task, deep=True)]
    if incomplete:
        preview = "\n".join(incomplete[:20])
        suffix = "" if len(incomplete) <= 20 else f"\n... and {len(incomplete) - 20} more"
        raise RuntimeError(
            f"{len(incomplete)}/{len(tasks)} ERA5 tasks failed NetCDF/metadata validation:\n"
            f"{preview}{suffix}"
        )

    common_root = targets[0].parent
    try:
        common_root = Path(os.path.commonpath([str(target) for target in targets]))
    except ValueError:
        pass
    if common_root.is_file():
        common_root = common_root.parent
    partials = tuple(
        str(candidate)
        for candidate in sorted(common_root.rglob("*.part"))
        if candidate.is_file()
    ) if common_root.exists() else ()
    if partials:
        raise RuntimeError(
            f"Raw ERA5 tree still contains {len(partials)} partial files; first={partials[0]}"
        )

    groups: dict[str, int] = {}
    for task in tasks:
        groups[task.group] = groups.get(task.group, 0) + 1
    return RawValidationSummary(
        task_count=len(tasks),
        total_bytes=sum(target.stat().st_size for target in targets),
        group_counts=dict(sorted(groups.items())),
        partial_files=partials,
    )


def expected_date_labels(years: Sequence[int], months: Sequence[int]) -> tuple[int, ...]:
    """Return the exact monotonically increasing UTC-day labels for a cache."""
    year_set = set(int(value) for value in years)
    month_set = set(int(value) for value in months)
    if not year_set or not month_set or min(month_set) < 1 or max(month_set) > 12:
        raise ValueError("Years and months must be non-empty, with months within 1-12.")
    current = date(min(year_set), 1, 1)
    stop = date(max(year_set) + 1, 1, 1)
    labels = []
    while current < stop:
        if current.year in year_set and current.month in month_set:
            labels.append(current.year * 10000 + current.month * 100 + current.day)
        current += timedelta(days=1)
    return tuple(labels)


def _sample_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    count = max(1, min(int(count), int(length)))
    return tuple(sorted({int(value) for value in np.linspace(0, length - 1, count)}))


def validate_cache_store(
    store_path: Path,
    *,
    grid: GridSpec,
    date_labels: Sequence[int],
    target_source: str,
    sample_count: int = 9,
) -> CacheValidationSummary:
    """Validate a complete cache while reading only bounded representative days."""
    path = Path(store_path)
    metadata_file = metadata_path(path)
    if not metadata_file.is_file():
        raise RuntimeError(f"Missing cache metadata file {metadata_file}.")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read cache metadata {metadata_file}: {exc}") from exc

    root = _require_zarr().open_group(str(path), mode="r")
    required_arrays = {"data", "time", "lat", "lon"}
    missing_arrays = sorted(required_arrays - set(root.array_keys()))
    if missing_arrays:
        raise RuntimeError(f"Cache is missing arrays {missing_arrays}.")
    data = root["data"]
    time = root["time"]
    expected_dates = tuple(int(value) for value in date_labels)
    expected_shape = (len(expected_dates),) + grid.shape + (len(CACHE_CHANNELS),)
    expected_chunks = (1,) + grid.shape + (len(CACHE_CHANNELS),)
    if tuple(data.shape) != expected_shape:
        raise RuntimeError(f"Cache data shape {tuple(data.shape)} != {expected_shape}.")
    if tuple(data.chunks) != expected_chunks:
        raise RuntimeError(f"Cache chunks {tuple(data.chunks)} != {expected_chunks}.")
    actual_dates = tuple(int(value) for value in np.asarray(time[:], dtype=np.int32))
    if actual_dates != expected_dates:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual_dates, expected_dates)) if pair[0] != pair[1]),
            min(len(actual_dates), len(expected_dates)),
        )
        raise RuntimeError(
            f"Cache UTC-day axis differs from the requested calendar at index {mismatch}."
        )
    if not np.allclose(np.asarray(root["lat"][:]), grid.lat, atol=1e-6, rtol=0.0):
        raise RuntimeError("Cache latitude coordinates do not match the configured grid.")
    if not np.allclose(np.asarray(root["lon"][:]), grid.lon, atol=1e-6, rtol=0.0):
        raise RuntimeError("Cache longitude coordinates do not match the configured grid.")
    channels = tuple(str(value) for value in root.attrs.get("channels", ()))
    if channels != CACHE_CHANNELS:
        raise RuntimeError(f"Cache channels {channels} != authoritative {CACHE_CHANNELS}.")
    if str(root.attrs.get("target_source")) != str(target_source):
        raise RuntimeError(
            f"Cache target_source={root.attrs.get('target_source')!r} != {target_source!r}."
        )
    if tuple(metadata.get("shape", ())) != expected_shape:
        raise RuntimeError("Cache metadata shape disagrees with the Zarr arrays.")

    sampled = _sample_indices(len(expected_dates), sample_count)
    land_index = CACHE_CHANNELS.index("land_mask")
    sst_index = CACHE_CHANNELS.index("sst")
    sst_valid_index = CACHE_CHANNELS.index("sst_valid")
    for index in sampled:
        daily = np.asarray(data[index, :, :, :], dtype=np.float32)
        if not np.all(np.isfinite(daily)):
            bad = np.argwhere(~np.isfinite(daily))[0]
            channel = CACHE_CHANNELS[int(bad[-1])]
            raise RuntimeError(
                f"Non-finite cache value on {expected_dates[index]} in channel {channel}."
            )
        land = daily[..., land_index]
        sst_valid = daily[..., sst_valid_index]
        if not np.all(np.isin(land, (0.0, 1.0))):
            raise RuntimeError(f"land_mask is not binary on {expected_dates[index]}.")
        if not np.all(np.isin(sst_valid, (0.0, 1.0))):
            raise RuntimeError(f"sst_valid is not binary on {expected_dates[index]}.")
        if not np.all(daily[..., sst_index][sst_valid == 0.0] == 0.0):
            raise RuntimeError(f"Invalid SST cells are not zero-filled on {expected_dates[index]}.")

    return CacheValidationSummary(
        store=str(path),
        shape=tuple(int(value) for value in data.shape),
        chunks=tuple(int(value) for value in data.chunks),
        first_date=expected_dates[0],
        last_date=expected_dates[-1],
        sampled_indices=sampled,
    )


def _write_report(path: Optional[Path], payload: Mapping) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("raw", "cache"))
    parser.add_argument("--data_root", type=Path, default=Path(Config.DATA_ROOT))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--resolution", choices=tuple(Config.RESOLUTION_SPECS), default=Config.RESOLUTION)
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--target_source", choices=("daily_statistics", "hourly_fallback"), default="daily_statistics")
    parser.add_argument("--expected_tasks", type=int, default=None)
    parser.add_argument("--sample_count", type=int, default=9)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.stage == "raw":
        summary = validate_raw_manifest(
            args.manifest or args.data_root / "manifests" / "era5_download_tasks.json",
            expected_task_count=args.expected_tasks,
        )
    else:
        years = parse_years(args.years)
        months = tuple(int(value) for value in args.months.split(",") if value.strip())
        summary = validate_cache_store(
            args.cache or args.data_root / "cache" / f"era5_{args.resolution}.zarr",
            grid=grid_for_resolution(args.resolution),
            date_labels=expected_date_labels(years, months),
            target_source=args.target_source,
            sample_count=args.sample_count,
        )
    payload = asdict(summary)
    _write_report(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
