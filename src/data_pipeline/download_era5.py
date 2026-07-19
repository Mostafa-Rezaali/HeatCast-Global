"""Build and execute resumable monthly CDS requests for HeatCast-Global ERA5.

The downloader writes only external runtime data below ``Config.DATA_ROOT``.
Requests are idempotent, use atomic ``.part`` files, and record the selected
Tmax source in adjacent JSON metadata. CI exercises request construction with
fixtures and never contacts CDS.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from cfm_mesh_train import Config


PREFERRED_DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
HOURLY_SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"
PRESSURE_LEVEL_DATASET = None  # TODO(USER): pin the CDS pressure-level dataset identifier.
YEAR_RANGE: Tuple[int, ...] = tuple(range(1979, 2025))
MONTHS: Tuple[int, ...] = tuple(range(1, 13))
HOURS: Tuple[str, ...] = tuple(f"{hour:02d}:00" for hour in range(24))

SINGLE_LEVEL_VARIABLES: Tuple[str, ...] = (
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "sea_surface_temperature",
    "mean_sea_level_pressure",
)
PRESSURE_850_VARIABLES: Tuple[str, ...] = (
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
)


@dataclass(frozen=True)
class DownloadTask:
    """One atomic monthly or static CDS retrieval."""

    group: str
    year: int
    month: int
    dataset: Optional[str]
    request: dict
    target: str
    source_choice: str


def _days_for_month() -> list[str]:
    """Return a superset accepted by CDS; invalid month-days are ignored server-side."""
    return [f"{day:02d}" for day in range(1, 32)]


def _target(raw_root: Path, group: str, year: int, month: int) -> Path:
    return raw_root / group / str(int(year)) / f"{group}_{int(year):04d}{int(month):02d}.nc"


def _daily_request(year: int, month: int, statistic: str) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(int(year)),
        "month": f"{int(month):02d}",
        "day": _days_for_month(),
        "daily_statistic": str(statistic),
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "format": "netcdf",
    }


def _hourly_request(year: int, month: int, variables: Sequence[str], pressure_levels=None) -> dict:
    request = {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": str(int(year)),
        "month": f"{int(month):02d}",
        "day": _days_for_month(),
        "time": list(HOURS),
        "format": "netcdf",
    }
    if pressure_levels is not None:
        request["pressure_level"] = list(pressure_levels)
    return request


def build_download_tasks(
    raw_root: Path,
    years: Sequence[int] = YEAR_RANGE,
    months: Sequence[int] = MONTHS,
    *,
    target_source: str = "daily_statistics",
    pressure_dataset: Optional[str] = PRESSURE_LEVEL_DATASET,
    enable_heat_index: bool = False,
) -> Tuple[DownloadTask, ...]:
    """Return deterministic, chunked request tasks without contacting CDS."""
    if target_source not in ("daily_statistics", "hourly_fallback"):
        raise ValueError("target_source must be 'daily_statistics' or 'hourly_fallback'.")
    tasks = []
    for year in sorted({int(value) for value in years}):
        if year < 1979 or year > 2024:
            raise ValueError(f"ERA5 year must be within 1979-2024, got {year}.")
        for month in sorted({int(value) for value in months}):
            if month < 1 or month > 12:
                raise ValueError(f"Invalid calendar month {month}.")
            if target_source == "daily_statistics":
                daily_specs = (
                    ("daily_tmax", "daily_maximum"),
                    ("daily_t2m", "daily_mean"),
                )
                for group, statistic in daily_specs:
                    tasks.append(DownloadTask(
                        group=group,
                        year=year,
                        month=month,
                        dataset=PREFERRED_DAILY_DATASET,
                        request=_daily_request(year, month, statistic),
                        target=str(_target(raw_root, group, year, month)),
                        source_choice=target_source,
                    ))
            else:
                group = "hourly_t2m"
                tasks.append(DownloadTask(
                    group=group,
                    year=year,
                    month=month,
                    dataset=HOURLY_SINGLE_LEVEL_DATASET,
                    request=_hourly_request(year, month, ("2m_temperature",)),
                    target=str(_target(raw_root, group, year, month)),
                    source_choice=target_source,
                ))

            single_variables = list(SINGLE_LEVEL_VARIABLES)
            if enable_heat_index:
                single_variables.append("2m_dewpoint_temperature")
            tasks.append(DownloadTask(
                group="single_levels",
                year=year,
                month=month,
                dataset=HOURLY_SINGLE_LEVEL_DATASET,
                request=_hourly_request(year, month, single_variables),
                target=str(_target(raw_root, "single_levels", year, month)),
                source_choice=target_source,
            ))
            tasks.append(DownloadTask(
                group="pressure_geopotential",
                year=year,
                month=month,
                dataset=pressure_dataset,
                request=_hourly_request(year, month, ("geopotential",), ("300", "500")),
                target=str(_target(raw_root, "pressure_geopotential", year, month)),
                source_choice=target_source,
            ))
            tasks.append(DownloadTask(
                group="pressure_850",
                year=year,
                month=month,
                dataset=pressure_dataset,
                request=_hourly_request(year, month, PRESSURE_850_VARIABLES, ("850",)),
                target=str(_target(raw_root, "pressure_850", year, month)),
                source_choice=target_source,
            ))

    # Static ERA5 surface geopotential and land-sea mask are retrieved once.
    static_target = raw_root / "static" / "era5_static.nc"
    tasks.append(DownloadTask(
        group="static",
        year=1979,
        month=1,
        dataset=HOURLY_SINGLE_LEVEL_DATASET,
        request={
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask"],
            "year": "1979",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        target=str(static_target),
        source_choice=target_source,
    ))
    return tuple(tasks)


def _metadata_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".metadata.json")


def task_complete(task: DownloadTask) -> bool:
    """Return whether a non-empty target and matching task metadata exist."""
    target = Path(task.target)
    metadata_path = _metadata_path(target)
    if not target.is_file() or target.stat().st_size <= 0 or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("task") == asdict(task)


def retrieve_task(client, task: DownloadTask) -> str:
    """Retrieve one task atomically, or skip it when its contract matches."""
    if task_complete(task):
        return f"exists, skipping: {task.target}"
    if not task.dataset:
        raise RuntimeError(
            "Pressure-level CDS dataset is not pinned. Set --pressure_dataset after resolving "
            "TODO(USER) in docs/DECISIONS_NEEDED.md."
        )
    target = Path(task.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        client.retrieve(task.dataset, task.request, str(partial))
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError(f"CDS retrieval produced no data: {partial}")
        partial.replace(target)
        metadata = {
            "task": asdict(task),
            "target_source": task.source_choice,
            "utc_days": True,
        }
        metadata_path = _metadata_path(target)
        metadata_partial = metadata_path.with_suffix(metadata_path.suffix + ".part")
        metadata_partial.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        metadata_partial.replace(metadata_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return f"retrieved: {target}"


def run_tasks(tasks: Iterable[DownloadTask], workers: int) -> None:
    """Run bounded independent CDS requests with one client per worker task."""
    if int(workers) < 1:
        raise ValueError("workers must be at least one.")

    def execute(task):
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError("Install cdsapi and configure ~/.cdsapirc on HiPerGator.") from exc
        return retrieve_task(cdsapi.Client(), task)

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            print(future.result(), flush=True)


def parse_years(text: str) -> Tuple[int, ...]:
    """Parse inclusive ``START-END`` or comma-separated years."""
    value = str(text).strip()
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", 1))
        return tuple(range(start, end + 1))
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path(Config.DATA_ROOT))
    parser.add_argument("--years", default="1979-2024")
    parser.add_argument("--months", default=",".join(str(value) for value in MONTHS))
    parser.add_argument("--target_source", choices=("daily_statistics", "hourly_fallback"), default="daily_statistics")
    parser.add_argument("--pressure_dataset", default=PRESSURE_LEVEL_DATASET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--enable_heat_index", action="store_true", default=Config.ENABLE_HEAT_INDEX)
    parser.add_argument("--manifest_only", action="store_true")
    args = parser.parse_args()
    raw_root = args.data_root / "raw" / "era5"
    tasks = build_download_tasks(
        raw_root,
        parse_years(args.years),
        tuple(int(value) for value in args.months.split(",") if value.strip()),
        target_source=args.target_source,
        pressure_dataset=args.pressure_dataset,
        enable_heat_index=args.enable_heat_index,
    )
    manifest = args.data_root / "manifests" / "era5_download_tasks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([asdict(task) for task in tasks], indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} idempotent tasks to {manifest}")
    if args.manifest_only:
        return 0
    run_tasks(tasks, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
