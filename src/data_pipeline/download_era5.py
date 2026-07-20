"""Build and execute resumable monthly CDS requests for HeatCast-Global ERA5.

The downloader writes only external runtime data below ``Config.DATA_ROOT``.
Requests are idempotent, use atomic ``.part`` files, and record the selected
Tmax source in adjacent JSON metadata. CI exercises request construction with
fixtures and never contacts CDS.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from netCDF4 import Dataset as NetCDFDataset

from cfm_mesh_train import Config


PREFERRED_DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
HOURLY_SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"
PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"
CDS_CLIMATE_API_URL = "https://cds.climate.copernicus.eu/api"
DEFAULT_DOWNLOAD_WORKERS = 8
DEFAULT_PER_DATASET_WORKERS = 1
DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_BASE_SECONDS = 60.0
MAX_RETRY_DELAY_SECONDS = 900.0
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
NETCDF_VARIABLE_CANDIDATES = {
    "2m_temperature": ("t2m", "2m_temperature"),
    "2m_dewpoint_temperature": ("d2m", "2m_dewpoint_temperature"),
    "volumetric_soil_water_layer_1": ("swvl1", "volumetric_soil_water_layer_1"),
    "volumetric_soil_water_layer_2": ("swvl2", "volumetric_soil_water_layer_2"),
    "sea_surface_temperature": ("sst", "sea_surface_temperature"),
    "mean_sea_level_pressure": ("msl", "mean_sea_level_pressure"),
    "geopotential": ("z", "geopotential"),
    "temperature": ("t", "temperature"),
    "specific_humidity": ("q", "specific_humidity"),
    "u_component_of_wind": ("u", "u_component_of_wind"),
    "v_component_of_wind": ("v", "v_component_of_wind"),
    "land_sea_mask": ("lsm", "land_sea_mask"),
}


@dataclass(frozen=True)
class DownloadTask:
    """One atomic annual, monthly, or static CDS retrieval."""

    group: str
    year: int
    months: Tuple[int, ...]
    dataset: Optional[str]
    request: dict
    target: str
    source_choice: str


def era5_cds_rc_path() -> Path:
    """Return the ERA5 credential file without reusing the ECMWF ECDS file."""
    return Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc-era5")).expanduser()


def _configured_cds_url(path: Path) -> Optional[str]:
    """Read only the non-secret URL field from a cdsapi YAML configuration."""
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "url":
            return value.strip().strip("'\"")
    return None


def validate_cds_endpoint(path: Optional[Path] = None) -> Path:
    """Fail before task submission unless ERA5 is routed to Climate Data Store."""
    config_path = Path(path).expanduser() if path is not None else era5_cds_rc_path()
    if not config_path.is_file() or config_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Missing ERA5 CDS credentials at {config_path}. Create this separate file "
            f"with url: {CDS_CLIMATE_API_URL} and your CDS personal access token."
        )
    configured_url = _configured_cds_url(config_path)
    if configured_url is None:
        raise RuntimeError(f"Missing 'url:' in ERA5 CDS configuration {config_path}.")
    if configured_url.rstrip("/") != CDS_CLIMATE_API_URL:
        raise RuntimeError(
            f"Wrong endpoint in {config_path}: {configured_url}. ERA5 collection IDs "
            f"must use {CDS_CLIMATE_API_URL}; https://ecds.ecmwf.int/api is the "
            "separate ECMWF ECDS/S2S service. Keep separate credential files."
        )
    return config_path


def _days_for_month() -> list[str]:
    """Return a superset accepted by CDS; invalid month-days are ignored server-side."""
    return [f"{day:02d}" for day in range(1, 32)]


def month_chunks(
    months: Sequence[int],
    chunking: str,
) -> Tuple[Tuple[int, ...], ...]:
    """Return one annual month group or twelve independent monthly groups."""
    selected = tuple(sorted({int(value) for value in months}))
    if not selected or any(value < 1 or value > 12 for value in selected):
        raise ValueError(f"Months must be within 1-12, got {selected}.")
    if str(chunking) == "yearly":
        return (selected,)
    if str(chunking) == "monthly":
        return tuple((value,) for value in selected)
    raise ValueError("chunking must be 'yearly' or 'monthly'.")


def download_target_path(
    raw_root: Path,
    group: str,
    year: int,
    months: Sequence[int],
) -> Path:
    """Return a collision-free target for one annual or monthly request."""
    selected = tuple(int(value) for value in months)
    if selected == MONTHS:
        label = f"{int(year):04d}"
    elif len(selected) == 1:
        label = f"{int(year):04d}{selected[0]:02d}"
    else:
        label = f"{int(year):04d}_m{''.join(f'{value:02d}' for value in selected)}"
    return raw_root / group / str(int(year)) / f"{group}_{label}.nc"


def _daily_request(year: int, months: Sequence[int], statistic: str) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(int(year)),
        "month": [f"{int(month):02d}" for month in months],
        "day": _days_for_month(),
        "daily_statistic": str(statistic),
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _hourly_request(
    year: int,
    months: Sequence[int],
    variables: Sequence[str],
    pressure_levels=None,
    *,
    times: Sequence[str] = HOURS,
) -> dict:
    request = {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": str(int(year)),
        "month": [f"{int(month):02d}" for month in months],
        "day": _days_for_month(),
        "time": list(times),
        "data_format": "netcdf",
        "download_format": "unarchived",
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
    chunking: str = "yearly",
) -> Tuple[DownloadTask, ...]:
    """Return deterministic, chunked request tasks without contacting CDS."""
    if target_source not in ("daily_statistics", "hourly_fallback"):
        raise ValueError("target_source must be 'daily_statistics' or 'hourly_fallback'.")
    chunks = month_chunks(months, chunking)
    tasks = []
    for year in sorted({int(value) for value in years}):
        if year < 1979 or year > 2024:
            raise ValueError(f"ERA5 year must be within 1979-2024, got {year}.")
        for selected_months in chunks:
            if target_source == "daily_statistics":
                daily_specs = (
                    ("daily_tmax", "daily_maximum"),
                    ("daily_t2m", "daily_mean"),
                )
                for group, statistic in daily_specs:
                    tasks.append(DownloadTask(
                        group=group,
                        year=year,
                        months=selected_months,
                        dataset=PREFERRED_DAILY_DATASET,
                        request=_daily_request(year, selected_months, statistic),
                        target=str(
                            download_target_path(
                                raw_root, group, year, selected_months
                            )
                        ),
                        source_choice=target_source,
                    ))
            else:
                group = "hourly_t2m"
                tasks.append(DownloadTask(
                    group=group,
                    year=year,
                    months=selected_months,
                    dataset=HOURLY_SINGLE_LEVEL_DATASET,
                    request=_hourly_request(
                        year, selected_months, ("2m_temperature",)
                    ),
                    target=str(
                        download_target_path(raw_root, group, year, selected_months)
                    ),
                    source_choice=target_source,
                ))

            single_variables = list(SINGLE_LEVEL_VARIABLES)
            if enable_heat_index:
                single_variables.append("2m_dewpoint_temperature")
            tasks.append(DownloadTask(
                group="single_levels",
                year=year,
                months=selected_months,
                dataset=HOURLY_SINGLE_LEVEL_DATASET,
                request=_hourly_request(
                    year, selected_months, single_variables, times=("00:00",)
                ),
                target=str(
                    download_target_path(
                        raw_root, "single_levels", year, selected_months
                    )
                ),
                source_choice=target_source,
            ))
            tasks.append(DownloadTask(
                group="pressure_geopotential",
                year=year,
                months=selected_months,
                dataset=pressure_dataset,
                request=_hourly_request(
                    year,
                    selected_months,
                    ("geopotential",),
                    ("300", "500"),
                    times=("00:00",),
                ),
                target=str(
                    download_target_path(
                        raw_root, "pressure_geopotential", year, selected_months
                    )
                ),
                source_choice=target_source,
            ))
            tasks.append(DownloadTask(
                group="pressure_850",
                year=year,
                months=selected_months,
                dataset=pressure_dataset,
                request=_hourly_request(
                    year,
                    selected_months,
                    PRESSURE_850_VARIABLES,
                    ("850",),
                    times=("00:00",),
                ),
                target=str(
                    download_target_path(
                        raw_root, "pressure_850", year, selected_months
                    )
                ),
                source_choice=target_source,
            ))

    # Static ERA5 surface geopotential and land-sea mask are retrieved once.
    static_target = raw_root / "static" / "era5_static.nc"
    tasks.append(DownloadTask(
        group="static",
        year=1979,
        months=(1,),
        dataset=HOURLY_SINGLE_LEVEL_DATASET,
        request={
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask"],
            "year": "1979",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        target=str(static_target),
        source_choice=target_source,
    ))
    return tuple(tasks)


def _metadata_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".metadata.json")


def _task_record(task: DownloadTask) -> dict:
    """Return the JSON-stable task contract used for manifests and resumes."""
    record = asdict(task)
    record["months"] = list(task.months)
    return record


def validate_download_file(path: Path, task: DownloadTask) -> None:
    """Reject archives, corrupt files, and NetCDF payloads missing requested fields."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"CDS retrieval produced no data: {path}")
    try:
        with NetCDFDataset(str(path)) as dataset:
            available = set(dataset.variables)
            if not any(name in available for name in ("latitude", "lat")):
                raise RuntimeError("missing latitude coordinate")
            if not any(name in available for name in ("longitude", "lon")):
                raise RuntimeError("missing longitude coordinate")
            requested_variables = task.request.get("variable", ())
            if isinstance(requested_variables, str):
                requested_variables = (requested_variables,)
            missing = []
            for requested in requested_variables:
                candidates = NETCDF_VARIABLE_CANDIDATES.get(str(requested), (str(requested),))
                if not any(candidate in available for candidate in candidates):
                    missing.append(str(requested))
            if missing:
                raise RuntimeError(f"missing requested variables {missing}; available={sorted(available)}")
            if not dataset.dimensions or any(len(value) <= 0 for value in dataset.dimensions.values()):
                raise RuntimeError("empty NetCDF dimensions")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Invalid NetCDF payload for {task.group}: {path}: {exc}") from exc


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
    if metadata.get("task") != _task_record(task):
        return False
    try:
        validate_download_file(target, task)
    except RuntimeError:
        return False
    return True


def retrieve_task(client, task: DownloadTask) -> str:
    """Retrieve one task atomically, or skip it when its contract matches."""
    if task_complete(task):
        return f"exists, skipping: {task.target}"
    if not task.dataset:
        raise RuntimeError(
            "Pressure-level CDS dataset is empty. Pass --pressure_dataset explicitly."
        )
    target = Path(task.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        client.retrieve(task.dataset, task.request, str(partial))
        validate_download_file(partial, task)
        partial.replace(target)
        metadata = {
            "task": _task_record(task),
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


def is_retryable_cds_error(error: Exception) -> bool:
    """Return whether CDS reports transient queue pressure or server failure."""
    message = str(error).lower()
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    queue_markers = (
        "number queued requests for this dataset is temporarily limited",
        "queued requests per user",
        "too many requests",
        "temporarily unavailable",
    )
    return status_code in (429, 500, 502, 503, 504) or any(
        marker in message for marker in queue_markers
    )


def run_tasks(
    tasks: Iterable[DownloadTask],
    workers: int,
    *,
    per_dataset_workers: int = DEFAULT_PER_DATASET_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
) -> None:
    """Run parallel CDS requests with per-dataset limits and queue backoff."""
    if int(workers) < 1:
        raise ValueError("workers must be at least one.")
    if int(per_dataset_workers) < 1:
        raise ValueError("per_dataset_workers must be at least one.")
    if int(max_retries) < 0:
        raise ValueError("max_retries cannot be negative.")
    if float(retry_base_seconds) < 0:
        raise ValueError("retry_base_seconds cannot be negative.")
    pending_tasks = tuple(tasks)
    config_path = validate_cds_endpoint()
    os.environ["CDSAPI_RC"] = str(config_path)
    dataset_gates = {
        dataset: threading.BoundedSemaphore(int(per_dataset_workers))
        for dataset in {task.dataset for task in pending_tasks}
    }
    print(
        f"Starting {len(pending_tasks)} CDS tasks with "
        f"{int(workers)} local workers, at most {int(per_dataset_workers)} "
        "active request(s) per dataset, and automatic queue backoff.",
        flush=True,
    )

    def execute(task):
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError(
                "Install cdsapi and configure ~/.cdsapirc-era5 on HiPerGator."
            ) from exc
        with dataset_gates[task.dataset]:
            for retry_number in range(int(max_retries) + 1):
                try:
                    return retrieve_task(cdsapi.Client(), task)
                except Exception as exc:
                    if (
                        not is_retryable_cds_error(exc)
                        or retry_number >= int(max_retries)
                    ):
                        raise
                    delay = min(
                        float(retry_base_seconds) * (2 ** retry_number),
                        MAX_RETRY_DELAY_SECONDS,
                    )
                    print(
                        f"CDS queue limited dataset={task.dataset} "
                        f"group={task.group} year={task.year} "
                        f"months={','.join(f'{value:02d}' for value in task.months)}; "
                        f"retry {retry_number + 1}/{int(max_retries)} "
                        f"in {delay:.0f}s.",
                        flush=True,
                    )
                    time.sleep(delay)
        raise AssertionError("CDS retry loop ended unexpectedly.")

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(execute, task): task for task in pending_tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            print(
                f"[{completed}/{len(futures)}] {future.result()}",
                flush=True,
            )


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
    parser.add_argument("--chunking", choices=("yearly", "monthly"), default="yearly")
    parser.add_argument("--target_source", choices=("daily_statistics", "hourly_fallback"), default="daily_statistics")
    parser.add_argument("--pressure_dataset", default=PRESSURE_LEVEL_DATASET)
    parser.add_argument("--workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument(
        "--per_dataset_workers", type=int, default=DEFAULT_PER_DATASET_WORKERS
    )
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--retry_base_seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS
    )
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
        chunking=args.chunking,
    )
    manifest = args.data_root / "manifests" / "era5_download_tasks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps([_task_record(task) for task in tasks], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} idempotent tasks to {manifest}")
    if args.manifest_only:
        return 0
    run_tasks(
        tasks,
        args.workers,
        per_dataset_workers=args.per_dataset_workers,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
