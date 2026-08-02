#!/usr/bin/env python3
"""Build the fold-normalized global teleconnection-vector input cache.

The persisted array combines daily CPC PNA/NAO/AO, monthly Nino3.4, and the
preserved ERSSTv5 PDO workbook on the ERA5 daily time axis. Fold-specific means
and standard deviations remain fitted at model load time, so no test-year
information enters preprocessing statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_driver_tables import parse_nino34_file, read_first_xlsx_sheet


TELECONNECTION_CHANNELS: Tuple[str, ...] = (
    "pna",
    "nao",
    "nino34",
    "pdo",
    "ao",
)

DEFAULT_SOURCE_URLS: Mapping[str, str] = {
    "pna": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.index.b500101.current.ascii",
    "nao": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii",
    "nino34": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
    "ao": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii",
}

DAILY_CHANNELS: Tuple[str, ...] = ("pna", "nao", "ao")
MONTHLY_CHANNELS: Tuple[str, ...] = ("nino34", "pdo")

# CPC publishes the daily indices rounded to three decimals. The historical
# CondTrain file was generated from higher-precision copies, and the observed
# normalized differences are <=1.73e-4 across all 6,579 overlap dates.
LEGACY_TOLERANCES: Mapping[str, float] = {
    "pna": 2.5e-4,
    "nao": 2.5e-4,
    "pdo": 5e-6,
    "ao": 2.5e-4,
}

LEGACY_COLUMN_MAP: Mapping[str, int] = {
    "pna": 0,
    "nao": 1,
    "pdo": 3,
    "ao": 4,
}


def default_source_paths(data_root: Path, pdo_workbook: Path) -> Dict[str, Path]:
    """Return generated source paths under the configured global data root."""
    driver_root = Path(data_root) / "drivers"
    return {
        "pna": driver_root / "teleconnections" / "pna.daily",
        "nao": driver_root / "teleconnections" / "nao.daily",
        "nino34": driver_root / "nino34.txt",
        "pdo": Path(pdo_workbook),
        "ao": driver_root / "teleconnections" / "ao.daily",
    }


def parse_daily_cpc_file(path: Path) -> Mapping[int, float]:
    """Parse CPC ``year month day value`` indices keyed by YYYYMMDD."""
    values: Dict[int, float] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year, month, day = (int(parts[index]) for index in range(3))
            value = float(parts[3])
        except ValueError:
            continue
        label = year * 10000 + month * 100 + day
        if label in values:
            raise RuntimeError(f"{path}:{line_number}: duplicate date {label}.")
        values[label] = value
    if not values:
        raise RuntimeError(f"No daily CPC values parsed from {path}.")
    return values


def parse_pdo_workbook(path: Path) -> Mapping[Tuple[int, int], float]:
    """Parse the preserved ERSSTv5 PDO workbook without openpyxl."""
    headers, rows = read_first_xlsx_sheet(Path(path))
    normalized = [str(value).strip().lower()[:3] for value in headers]
    expected = ["yea", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    if normalized[:13] != expected:
        raise RuntimeError(f"Unexpected PDO workbook columns in {path}: {headers}.")
    values: Dict[Tuple[int, int], float] = {}
    for row_number, row in enumerate(rows, 2):
        if not row or str(row[0]).strip() == "":
            continue
        try:
            year = int(float(str(row[0]).strip()))
        except ValueError as exc:
            raise RuntimeError(f"{path}:{row_number}: invalid PDO year {row[0]!r}.") from exc
        for month in range(1, 13):
            if month >= len(row) or str(row[month]).strip() == "":
                continue
            try:
                values[(year, month)] = float(str(row[month]).strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"{path}:{row_number}: invalid PDO value {row[month]!r} for month {month}."
                ) from exc
    if not values:
        raise RuntimeError(f"No PDO values parsed from {path}.")
    return values


def _parse_source(name: str, path: Path) -> Mapping[object, float]:
    if name in DAILY_CHANNELS:
        return parse_daily_cpc_file(path)
    if name == "nino34":
        return parse_nino34_file(path)
    if name == "pdo":
        return parse_pdo_workbook(path)
    raise KeyError(f"Unsupported condition source: {name}")


def download_source(url: str, target: Path, retries: int = 5) -> None:
    """Download one small public index file with retries and atomic rename."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 HeatCast-Global/1.0",
            "Accept": "text/plain,*/*;q=0.8",
        },
    )
    error = None
    for attempt in range(1, int(retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) < 100:
                raise RuntimeError(f"Downloaded payload is unexpectedly short ({len(payload)} bytes).")
            partial.write_bytes(payload)
            os.replace(partial, target)
            return
        except Exception as exc:  # network failures vary by site and Python version
            error = exc
            if partial.exists():
                partial.unlink()
            if attempt < int(retries):
                time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {error}") from error


def prepare_sources(
    paths: Mapping[str, Path],
    urls: Mapping[str, str] = DEFAULT_SOURCE_URLS,
    refresh: bool = False,
    allow_download: bool = True,
    retries: int = 5,
) -> None:
    """Ensure every configured source exists and passes its production parser."""
    for name in TELECONNECTION_CHANNELS:
        path = Path(paths[name])
        if name == "pdo":
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing preserved PDO workbook: {path}")
            parsed = _parse_source(name, path)
            if not parsed:
                raise RuntimeError(f"No values parsed for {name} from {path}.")
            continue
        if refresh or not path.is_file() or path.stat().st_size == 0:
            if not allow_download:
                raise FileNotFoundError(f"Missing teleconnection source with downloads disabled: {path}")
            print(f"Downloading {name}: {urls[name]} -> {path}", flush=True)
            download_source(urls[name], path, retries=retries)
        parsed = _parse_source(name, path)
        if not parsed:
            raise RuntimeError(f"No values parsed for {name} from {path}.")


def load_cache_dates(store_path: Path) -> np.ndarray:
    """Load only the integer YYYYMMDD axis from the global zarr cache."""
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr<3 is required to read the ERA5 cache axis.") from exc
    root = zarr.open_group(str(store_path), mode="r")
    if "time" not in root:
        raise RuntimeError(f"ERA5 cache is missing its time array: {store_path}")
    dates = np.asarray(root["time"][:], dtype=np.int32)
    if dates.ndim != 1 or dates.size == 0:
        raise RuntimeError(f"Invalid ERA5 cache time axis: shape={dates.shape}.")
    if np.any(np.diff(dates.astype(np.int64)) <= 0):
        raise RuntimeError("ERA5 cache date labels must be strictly increasing.")
    return dates


def expand_condition_indices(
    date_labels: Sequence[int],
    parsed_sources: Mapping[str, Mapping[object, float]],
) -> np.ndarray:
    """Align mixed daily/monthly condition indices to exact YYYYMMDD labels."""
    labels = np.asarray(date_labels, dtype=np.int64)
    output = np.empty((labels.size, len(TELECONNECTION_CHANNELS)), dtype=np.float32)
    missing = []
    invalid = []
    for row, label in enumerate(labels):
        year = int(label) // 10000
        month = (int(label) // 100) % 100
        for column, name in enumerate(TELECONNECTION_CHANNELS):
            values = parsed_sources[name]
            key = int(label) if name in DAILY_CHANNELS else (year, month)
            if key not in values:
                missing.append((int(label), name))
                output[row, column] = np.nan
                continue
            value = float(values[key])
            if not np.isfinite(value) or abs(value) >= 90.0:
                invalid.append((int(label), name, value))
                output[row, column] = np.nan
                continue
            output[row, column] = value
    if missing or invalid:
        raise RuntimeError(
            "Teleconnection sources do not cover the ERA5 cache axis: "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"invalid={invalid[:10]} ({len(invalid)} total)."
        )
    return output


# Backward-compatible name retained for existing callers and fixtures.
expand_monthly_indices = expand_condition_indices


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = np.min(values, axis=0)
    span = np.max(values, axis=0) - minimum
    if np.any(span <= 0):
        raise RuntimeError(f"Cannot min-max normalize constant legacy candidates: span={span}.")
    return (values - minimum) / span


def validate_preserved_legacy_channels(
    date_labels: Sequence[int],
    raw_values: np.ndarray,
    legacy_condtrain_path: Path,
    tolerances: Mapping[str, float] = LEGACY_TOLERANCES,
) -> Dict[str, float]:
    """Prove the four preserved channels reproduce the legacy normalized file."""
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required for legacy CondTrain validation.") from exc

    with Dataset(str(legacy_condtrain_path)) as dataset:
        legacy = np.asarray(dataset.variables["CondTrain"][:], dtype=np.float64)
        time_var = dataset.variables["time"]
        units = getattr(time_var, "units", None)
        if not units:
            raise RuntimeError("Legacy CondTrain time coordinate has no units.")
        dates = num2date(
            time_var[:],
            units=units,
            calendar=getattr(time_var, "calendar", "standard"),
        )
    if legacy.shape[0] == 5:
        legacy = legacy.T
    if legacy.shape != (len(dates), 5):
        raise RuntimeError(f"Unexpected legacy CondTrain shape after orientation: {legacy.shape}.")

    lookup = {int(label): index for index, label in enumerate(np.asarray(date_labels, dtype=np.int64))}
    indices = []
    legacy_labels = []
    for value in dates:
        label = int(f"{int(value.year):04d}{int(value.month):02d}{int(value.day):02d}")
        if label not in lookup:
            raise RuntimeError(f"Legacy CondTrain date is absent from ERA5 cache: {label}.")
        indices.append(lookup[label])
        legacy_labels.append(label)
    candidate = np.asarray(raw_values[np.asarray(indices, dtype=np.int64)], dtype=np.float64)
    report: Dict[str, float] = {}
    failures = []
    for name, legacy_column in LEGACY_COLUMN_MAP.items():
        candidate_column = TELECONNECTION_CHANNELS.index(name)
        reconstructed = _minmax(candidate[:, [candidate_column]])[:, 0]
        expected = legacy[:, legacy_column]
        difference = float(np.max(np.abs(reconstructed - expected)))
        correlation = float(np.corrcoef(reconstructed, expected)[0, 1])
        report[f"{name}_max_abs_difference"] = difference
        report[f"{name}_correlation"] = correlation
        tolerance = float(tolerances[name])
        report[f"{name}_tolerance"] = tolerance
        if (
            not np.isfinite(difference)
            or difference > tolerance
            or not np.isfinite(correlation)
            or correlation < 0.99999
        ):
            failures.append((name, difference, correlation))
    report["matched_dates"] = int(len(legacy_labels))
    if failures:
        raise RuntimeError(
            "Current sources do not reproduce preserved legacy CondTrain channels; "
            f"refusing to build: {failures}."
        )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_condition_cache(
    output_path: Path,
    date_labels: Sequence[int],
    values: np.ndarray,
    source_paths: Mapping[str, Path],
    legacy_report: Mapping[str, float],
) -> Path:
    """Atomically write the raw five-index array and provenance metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".part")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    os.replace(partial, output_path)

    labels = np.asarray(date_labels, dtype=np.int64)
    metadata = {
        "channels": list(TELECONNECTION_CHANNELS),
        "shape": list(values.shape),
        "first_date": int(labels[0]),
        "last_date": int(labels[-1]),
        "normalization": "raw values; mean/std fitted from active fold training initializations",
        "decision": "Legacy OMI PC2 replaced by NOAA PSL Nino3.4 anomaly; user approved 2026-08-02.",
        "sources": {
            name: {
                "path": str(Path(source_paths[name]).resolve()),
                "url": DEFAULT_SOURCE_URLS.get(name),
                "source_kind": (
                    "daily_cpc" if name in DAILY_CHANNELS
                    else "preserved_ersstv5_workbook" if name == "pdo"
                    else "monthly_noaa_psl"
                ),
                "sha256": _sha256(Path(source_paths[name])),
            }
            for name in TELECONNECTION_CHANNELS
        },
        "legacy_validation": dict(legacy_report),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_partial = metadata_path.with_name(metadata_path.name + ".part")
    metadata_partial.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(metadata_partial, metadata_path)
    return metadata_path


def build_condition_cache(
    store_path: Path,
    output_path: Path,
    source_paths: Mapping[str, Path],
    legacy_condtrain_path: Path,
) -> Dict[str, object]:
    """Build and validate the production five-index cache."""
    labels = load_cache_dates(store_path)
    parsed = {name: _parse_source(name, Path(source_paths[name])) for name in TELECONNECTION_CHANNELS}
    values = expand_condition_indices(labels, parsed)
    legacy_report = validate_preserved_legacy_channels(
        labels,
        values,
        legacy_condtrain_path,
    )
    metadata_path = write_condition_cache(output_path, labels, values, source_paths, legacy_report)
    return {
        "output": str(output_path),
        "metadata": str(metadata_path),
        "shape": list(values.shape),
        "channels": list(TELECONNECTION_CHANNELS),
        "first_date": int(labels[0]),
        "last_date": int(labels[-1]),
        "legacy_validation": legacy_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--resolution", default="1.5deg")
    parser.add_argument("--legacy_condtrain", type=Path, required=True)
    parser.add_argument(
        "--pdo_workbook",
        type=Path,
        required=True,
        help="Preserved PDO.xlsx used by the original HeatCast input pipeline.",
    )
    parser.add_argument("--refresh_sources", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--download_retries", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.download_retries < 1:
        raise ValueError("--download_retries must be positive.")
    data_root = args.data_root.resolve()
    paths = default_source_paths(data_root, args.pdo_workbook.resolve())
    prepare_sources(
        paths,
        refresh=bool(args.refresh_sources),
        allow_download=not bool(args.no_download),
        retries=int(args.download_retries),
    )
    store = data_root / "cache" / f"era5_{args.resolution}.zarr"
    output = data_root / "cache" / "teleconnection_5.npy"
    result = build_condition_cache(store, output, paths, args.legacy_condtrain.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"GLOBAL CONDITION CACHE COMPLETE: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
