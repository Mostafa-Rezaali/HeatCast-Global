"""Configured target-grid, lazy truth, mask, and area-weight helpers for ENS.

The ECMWF pipeline imports this module at its I/O boundary.  Cycle-specific
quantile mapping, fold-safe calibration, and matched-initialization logic stay
unchanged while CONUS and global modes obtain their coordinates and masks from
one config-driven interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data_pipeline.build_cache import CACHE_CHANNELS, _require_zarr
from data_pipeline.regrid import grid_for_resolution
from init_calendar import W34_LEADS, window_falls_in_months
from spatial_weights import area_weights


BASE_DATE = datetime(1981, 5, 1)


def label_to_date(label: int) -> date:
    value = int(label)
    return date(value // 10000, (value // 100) % 100, value % 100)


@dataclass(frozen=True)
class ENSTargetGrid:
    """Rectilinear target grid plus all-land and headline masks."""

    domain: str
    resolution: str
    lat: np.ndarray
    lon: np.ndarray
    land_mask: np.ndarray

    def __post_init__(self):
        latitude = np.asarray(self.lat, dtype=np.float32)
        longitude = np.asarray(self.lon, dtype=np.float32)
        land = np.asarray(self.land_mask, dtype=bool)
        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError("ENS target latitude and longitude must be one-dimensional.")
        if land.shape != (latitude.size, longitude.size):
            raise ValueError("ENS land mask shape does not match target coordinates.")
        object.__setattr__(self, "lat", latitude)
        object.__setattr__(self, "lon", longitude)
        object.__setattr__(self, "land_mask", land)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.land_mask.shape

    @property
    def lat2d(self) -> np.ndarray:
        return np.broadcast_to(self.lat[:, None], self.shape)

    @property
    def lon2d(self) -> np.ndarray:
        return np.broadcast_to(self.lon[None, :], self.shape)

    def headline_mask(self, initialization=None, leads=W34_LEADS) -> np.ndarray:
        """Return CONUS land or NH land for a full-MJJAS-valid global window."""
        if self.domain != "global":
            return self.land_mask.copy()
        if initialization is not None and not window_falls_in_months(initialization, leads):
            return np.zeros(self.shape, dtype=bool)
        return self.land_mask & (self.lat[:, None] >= 0.0)

    def flattened_area_weights(self, mask=None) -> np.ndarray:
        """Return cosine-latitude weights in the selected flattened cell order."""
        selected = self.land_mask if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != self.shape:
            raise ValueError("Selected ENS mask does not match target grid.")
        weights = np.broadcast_to(area_weights(self.lat)[:, None], self.shape)
        values = np.asarray(weights[selected], dtype=np.float64)
        total = float(np.sum(values))
        return values / total if total > 0.0 else values


def target_grid_for_config(config) -> ENSTargetGrid:
    """Load the configured global cache grid or preserved CONUS target grid."""
    domain = str(getattr(config, "DOMAIN", "conus"))
    if domain == "global":
        expected = grid_for_resolution(str(config.RESOLUTION))
        zarr = _require_zarr()
        root = zarr.open_group(str(config.TRAINING_DATA_PATH), mode="r")
        lat = np.asarray(root["lat"][:], dtype=np.float32)
        lon = np.asarray(root["lon"][:], dtype=np.float32)
        land = np.asarray(
            root["data"][0, :, :, CACHE_CHANNELS.index("land_mask")] >= 0.5,
            dtype=bool,
        )
        if land.shape != expected.shape or not np.allclose(lat, expected.lat) or not np.allclose(lon, expected.lon):
            raise RuntimeError(
                f"Global cache grid does not match configured {config.RESOLUTION} grid {expected.shape}."
            )
        return ENSTargetGrid(domain="global", resolution=str(config.RESOLUTION), lat=lat, lon=lon, land_mask=land)

    import cfm_mesh_train as cfm
    from publication_analysis_utils import conus_lat_lon

    land = np.asarray(cfm.load_conus_mask(config).cpu().numpy() > 0.5, dtype=bool)
    lat, lon, _, _ = conus_lat_lon(land.shape)
    return ENSTargetGrid(domain="conus", resolution="prism_4km", lat=lat, lon=lon, land_mask=land)


def global_cache_time_axis(store_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Return date labels, legacy day offsets, and lookup from a global zarr cache."""
    zarr = _require_zarr()
    root = zarr.open_group(str(store_path), mode="r")
    labels = np.asarray(root["time"][:], dtype=np.int32)
    dates = np.asarray([np.datetime64(label_to_date(value), "D") for value in labels])
    offsets = np.asarray([(label_to_date(value) - BASE_DATE.date()).days for value in labels], dtype=np.float64)
    lookup = {str(int(value)): index for index, value in enumerate(labels)}
    return dates, offsets, lookup


class LazyGlobalChannel:
    """Expose one cache channel as ``(lat, lon, time)`` without eager loading."""

    def __init__(self, store_path: Path, channel: str):
        if channel not in CACHE_CHANNELS:
            raise ValueError(f"Unknown global cache channel {channel!r}.")
        self.store_path = str(store_path)
        self.channel = str(channel)
        self.channel_index = CACHE_CHANNELS.index(self.channel)
        self._root = None
        self._pid = None
        zarr = _require_zarr()
        root = zarr.open_group(self.store_path, mode="r")
        self.shape = (int(root["lat"].shape[0]), int(root["lon"].shape[0]), int(root["time"].shape[0]))
        self.ndim = 3

    def _open(self):
        import os

        if self._root is None or self._pid != os.getpid():
            self._root = _require_zarr().open_group(self.store_path, mode="r")
            self._pid = os.getpid()
        return self._root

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise IndexError("LazyGlobalChannel expects (lat, lon, time) indexing.")
        lat_index, lon_index, time_index = index
        if not isinstance(time_index, (int, np.integer)):
            if isinstance(time_index, slice):
                times = range(*time_index.indices(self.shape[2]))
            else:
                times = tuple(int(value) for value in np.asarray(time_index).reshape(-1))
            slices = [self[lat_index, lon_index, int(value)] for value in times]
            return np.stack(slices, axis=-1) if slices else np.empty((0,), dtype=np.float32)
        root = self._open()
        field = np.asarray(
            root["data"][int(time_index), :, :, self.channel_index],
            dtype=np.float32,
        )
        return field[lat_index, lon_index]

    def read_pixels_times(self, flat_pixels, time_indices) -> np.ndarray:
        """Read selected cells for selected dates one time chunk at a time."""
        pixels = np.asarray(flat_pixels, dtype=np.int64).reshape(-1)
        times = np.asarray(time_indices, dtype=np.int64).reshape(-1)
        if np.any(pixels < 0) or np.any(pixels >= self.shape[0] * self.shape[1]):
            raise IndexError("Global cache pixel index is outside the configured grid.")
        rows, columns = np.unravel_index(pixels, self.shape[:2])
        values = np.empty((pixels.size, times.size), dtype=np.float32)
        for output_index, time_index in enumerate(times):
            if time_index < 0 or time_index >= self.shape[2]:
                raise IndexError("Global cache time index is outside the configured axis.")
            values[:, output_index] = self[rows, columns, int(time_index)]
        return values


class LazyGlobalTruth(LazyGlobalChannel):
    """Expose zarr Tmax through the worker/process-lazy channel interface."""

    def __init__(self, store_path: Path):
        super().__init__(store_path, "tmax")


class LazyNormalizedGlobalTruth(LazyGlobalTruth):
    """Apply a fold preprocessor to each lazily read Tmax day."""

    def __init__(self, store_path: Path, preprocessor, date_labels):
        super().__init__(store_path)
        self.preprocessor = preprocessor
        self.date_labels = np.asarray(date_labels, dtype=np.int32)
        if self.date_labels.size != self.shape[2]:
            raise ValueError("Global normalized truth date axis does not match the zarr cache.")

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise IndexError("LazyNormalizedGlobalTruth expects (lat, lon, time) indexing.")
        lat_index, lon_index, time_index = index
        if not isinstance(time_index, (int, np.integer)):
            return super().__getitem__(index)
        full_field = LazyGlobalChannel.__getitem__(
            self,
            (slice(None), slice(None), int(time_index)),
        )
        normalized = self.preprocessor.transform(
            "tmax",
            int(self.date_labels[int(time_index)]),
            full_field,
        )
        return normalized[lat_index, lon_index]
