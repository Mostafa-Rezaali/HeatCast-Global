"""Validate selected global zarr cache slices against regridded raw fields."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data_pipeline.build_cache import CACHE_CHANNELS, _require_zarr


@dataclass(frozen=True)
class SliceCheck:
    """Per-channel raw-vs-cache agreement summary."""

    channel: str
    max_abs_error: float
    mean_abs_error: float
    passed: bool


def check_cached_slice(
    store_path: Path,
    time_index: int,
    expected_fields: Mapping[str, np.ndarray],
    *,
    channels: Sequence[str] = CACHE_CHANNELS,
    atol: float = 1e-5,
    opener=None,
) -> tuple[SliceCheck, ...]:
    """Compare one cache time slice with independently prepared raw fields."""
    if opener is None:
        opener = lambda path: _require_zarr().open_group(str(path), mode="r")
    root = opener(store_path)
    cache_channels = tuple(root.attrs.get("channels", channels))
    checks = []
    for channel in channels:
        if channel not in expected_fields:
            raise KeyError(f"Expected raw slice is missing channel {channel!r}.")
        cached = np.asarray(root["data"][int(time_index), :, :, cache_channels.index(channel)], dtype=np.float64)
        expected = np.asarray(expected_fields[channel], dtype=np.float64)
        if cached.shape != expected.shape:
            raise ValueError(f"Shape mismatch for {channel}: cache={cached.shape}, raw={expected.shape}.")
        difference = np.abs(cached - expected)
        finite = np.isfinite(difference)
        max_error = float(np.max(difference[finite])) if np.any(finite) else 0.0
        mean_error = float(np.mean(difference[finite])) if np.any(finite) else 0.0
        same_missing = np.array_equal(np.isnan(cached), np.isnan(expected))
        checks.append(SliceCheck(channel, max_error, mean_error, same_missing and max_error <= float(atol)))
    return tuple(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected_npz", type=Path, required=True,
                        help="Independent regridded raw slice with arrays named by cache channel.")
    parser.add_argument("--time_index", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    with np.load(args.expected_npz, allow_pickle=False) as expected:
        fields = {name: expected[name] for name in expected.files}
    checks = check_cached_slice(args.cache, args.time_index, fields, atol=args.atol)
    print(json.dumps([asdict(check) for check in checks], indent=2))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
