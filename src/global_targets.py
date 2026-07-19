"""Fold-safe harmonic climatology and normalization for global ERA5 fields.

The climatology uses an intercept plus the first four annual sine/cosine
harmonics. Sufficient statistics are accumulated only from explicitly supplied
training years; validation and test values never enter coefficients or
normalization statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Tuple

import numpy as np


ANNUAL_PERIOD_DAYS = 365.25
DEFAULT_HARMONICS = 4


def label_to_date(value) -> date:
    """Convert date-like or YYYYMMDD values to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    label = int(value)
    return date(label // 10000, (label // 100) % 100, label % 100)


def harmonic_features(day_of_year, n_harmonics: int = DEFAULT_HARMONICS) -> np.ndarray:
    """Return intercept and annual sine/cosine harmonic design features."""
    days = np.asarray(day_of_year, dtype=np.float64)
    columns = [np.ones_like(days, dtype=np.float64)]
    angle = 2.0 * np.pi * days / ANNUAL_PERIOD_DAYS
    for harmonic in range(1, int(n_harmonics) + 1):
        columns.extend((np.sin(harmonic * angle), np.cos(harmonic * angle)))
    return np.stack(columns, axis=-1)


def _feature_for_label(value, n_harmonics: int) -> np.ndarray:
    return harmonic_features(label_to_date(value).timetuple().tm_yday, n_harmonics)


@dataclass
class FoldFieldPreprocessor:
    """Per-pixel harmonic coefficients and fold-safe normalization arrays."""

    channels: Tuple[str, ...]
    anomaly_channels: Tuple[str, ...]
    coefficients: Mapping[str, np.ndarray]
    means: Mapping[str, np.ndarray]
    stds: Mapping[str, np.ndarray]
    train_years: Tuple[int, ...]
    n_harmonics: int = DEFAULT_HARMONICS

    def climatology(self, channel: str, valid_date) -> np.ndarray:
        """Evaluate one channel's smoothed day-of-year climatology."""
        if channel not in self.coefficients:
            return np.zeros_like(self.means[channel], dtype=np.float32)
        feature = _feature_for_label(valid_date, self.n_harmonics)
        return np.tensordot(feature, self.coefficients[channel], axes=(0, 0)).astype(np.float32)

    def anomaly(self, channel: str, valid_date, field) -> np.ndarray:
        """Subtract climatology only for channels marked with ``A``."""
        values = np.asarray(field, dtype=np.float32)
        if channel in self.anomaly_channels:
            values = values - self.climatology(channel, valid_date)
        return values

    def transform(self, channel: str, valid_date, field) -> np.ndarray:
        """Apply the fold-safe anomaly transform and per-pixel normalization."""
        values = self.anomaly(channel, valid_date, field)
        return ((values - self.means[channel]) / self.stds[channel]).astype(np.float32)

    def inverse(self, channel: str, valid_date, normalized) -> np.ndarray:
        """Invert normalization and, when applicable, restore climatology."""
        values = np.asarray(normalized, dtype=np.float32) * self.stds[channel] + self.means[channel]
        if channel in self.anomaly_channels:
            values = values + self.climatology(channel, valid_date)
        return values.astype(np.float32)

    def save(self, path: Path) -> None:
        """Write a pickle-free fold sidecar outside the repository."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "channels": np.asarray(self.channels, dtype="U64"),
            "anomaly_channels": np.asarray(self.anomaly_channels, dtype="U64"),
            "train_years": np.asarray(self.train_years, dtype=np.int16),
            "n_harmonics": np.asarray(self.n_harmonics, dtype=np.int16),
        }
        for channel in self.channels:
            arrays[f"mean__{channel}"] = np.asarray(self.means[channel], dtype=np.float32)
            arrays[f"std__{channel}"] = np.asarray(self.stds[channel], dtype=np.float32)
            if channel in self.coefficients:
                arrays[f"coef__{channel}"] = np.asarray(self.coefficients[channel], dtype=np.float32)
        np.savez_compressed(target, **arrays)
        metadata = {
            "target_mode": "climatology_anomaly",
            "harmonics": int(self.n_harmonics),
            "train_years": list(self.train_years),
            "fold_safe": True,
        }
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path):
        """Load a sidecar written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            channels = tuple(str(value) for value in data["channels"].tolist())
            anomaly_channels = tuple(str(value) for value in data["anomaly_channels"].tolist())
            coefficients = {
                channel: data[f"coef__{channel}"].copy()
                for channel in anomaly_channels
            }
            means = {channel: data[f"mean__{channel}"].copy() for channel in channels}
            stds = {channel: data[f"std__{channel}"].copy() for channel in channels}
            return cls(
                channels=channels,
                anomaly_channels=anomaly_channels,
                coefficients=coefficients,
                means=means,
                stds=stds,
                train_years=tuple(int(value) for value in data["train_years"].tolist()),
                n_harmonics=int(data["n_harmonics"]),
            )


def fit_fold_preprocessor(
    iterator_factory: Callable[[], Iterable[tuple]],
    channels: Sequence[str],
    train_years: Sequence[int],
    *,
    anomaly_channels: Sequence[str],
    n_harmonics: int = DEFAULT_HARMONICS,
    std_floor: float = 1e-6,
) -> FoldFieldPreprocessor:
    """Fit harmonic and normalization statistics from replayable lazy slices."""
    channels = tuple(str(channel) for channel in channels)
    anomaly_channels = tuple(str(channel) for channel in anomaly_channels)
    unknown = sorted(set(anomaly_channels) - set(channels))
    if unknown:
        raise ValueError(f"Anomaly channels are absent from channel list: {unknown}.")
    train_years = tuple(sorted({int(year) for year in train_years}))
    if not train_years:
        raise ValueError("Training-year set cannot be empty.")
    train_year_set = set(train_years)
    feature_count = 1 + 2 * int(n_harmonics)
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    rhs = None
    used_dates = []
    for valid_date, fields in iterator_factory():
        parsed = label_to_date(valid_date)
        if parsed.year not in train_year_set:
            continue
        feature = _feature_for_label(parsed, n_harmonics)
        if rhs is None:
            shape = np.asarray(fields[channels[0]]).shape
            rhs = {
                channel: np.zeros((feature_count,) + shape, dtype=np.float64)
                for channel in anomaly_channels
            }
        gram += np.outer(feature, feature)
        for channel in anomaly_channels:
            values = np.asarray(fields[channel], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite training value in {channel} on {parsed}.")
            rhs[channel] += feature.reshape((feature_count,) + (1,) * values.ndim) * values
        used_dates.append(parsed)
    if rhs is None or not used_dates:
        raise RuntimeError(f"No slices matched training years {train_years}.")
    observed_years = {value.year for value in used_dates}
    if not observed_years.issubset(train_year_set):
        raise AssertionError("Climatology fit consumed a non-training year.")
    regularized = gram + np.eye(feature_count, dtype=np.float64) * 1e-10
    coefficients = {
        channel: np.linalg.solve(regularized, values.reshape(feature_count, -1)).reshape(values.shape).astype(np.float32)
        for channel, values in rhs.items()
    }

    sums = {}
    sums2 = {}
    count = 0
    for valid_date, fields in iterator_factory():
        parsed = label_to_date(valid_date)
        if parsed.year not in train_year_set:
            continue
        feature = _feature_for_label(parsed, n_harmonics)
        for channel in channels:
            values = np.asarray(fields[channel], dtype=np.float64)
            if channel in anomaly_channels:
                climatology = np.tensordot(feature, coefficients[channel], axes=(0, 0))
                values = values - climatology
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite normalization value in {channel} on {parsed}.")
            if channel not in sums:
                sums[channel] = np.zeros_like(values, dtype=np.float64)
                sums2[channel] = np.zeros_like(values, dtype=np.float64)
            sums[channel] += values
            sums2[channel] += values * values
        count += 1
    means = {channel: (sums[channel] / count).astype(np.float32) for channel in channels}
    stds = {}
    for channel in channels:
        variance = np.maximum(sums2[channel] / count - np.square(means[channel], dtype=np.float64), 0.0)
        std = np.sqrt(variance)
        stds[channel] = np.where(std >= float(std_floor), std, 1.0).astype(np.float32)
    return FoldFieldPreprocessor(
        channels=channels,
        anomaly_channels=anomaly_channels,
        coefficients=coefficients,
        means=means,
        stds=stds,
        train_years=train_years,
        n_harmonics=int(n_harmonics),
    )


def fit_fold_preprocessor_from_arrays(
    dates: Sequence,
    data: np.ndarray,
    channels: Sequence[str],
    train_years: Sequence[int],
    *,
    anomaly_channels: Sequence[str],
    n_harmonics: int = DEFAULT_HARMONICS,
) -> FoldFieldPreprocessor:
    """Array-fixture adapter around the streaming fitter."""
    values = np.asarray(data)
    if values.shape[0] != len(dates) or values.shape[-1] != len(channels):
        raise ValueError("Data must have shape (time, ..., channel) matching dates/channels.")

    def iterator():
        for index, valid_date in enumerate(dates):
            yield valid_date, {
                str(channel): values[index, ..., channel_index]
                for channel_index, channel in enumerate(channels)
            }

    return fit_fold_preprocessor(
        iterator,
        channels,
        train_years,
        anomaly_channels=anomaly_channels,
        n_harmonics=n_harmonics,
    )


def fit_fold_preprocessor_from_zarr(
    store_path: Path,
    train_years: Sequence[int],
    *,
    anomaly_channels: Sequence[str],
    n_harmonics: int = DEFAULT_HARMONICS,
) -> FoldFieldPreprocessor:
    """Fit from a time-chunked cache while reading exactly one day at a time."""
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr<3 is required for global fold preprocessing.") from exc
    root = zarr.open_group(str(store_path), mode="r")
    channels = tuple(str(value) for value in root.attrs["channels"])
    date_labels = tuple(int(value) for value in np.asarray(root["time"][:]).tolist())
    del root

    def iterator():
        worker_root = zarr.open_group(str(store_path), mode="r")
        data = worker_root["data"]
        for index, valid_date in enumerate(date_labels):
            daily = np.asarray(data[index, :, :, :], dtype=np.float32)
            yield valid_date, {
                channel: daily[..., channel_index]
                for channel_index, channel in enumerate(channels)
            }

    return fit_fold_preprocessor(
        iterator,
        channels,
        train_years,
        anomaly_channels=anomaly_channels,
        n_harmonics=n_harmonics,
    )
