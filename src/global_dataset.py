"""Lazy global ERA5 model-input assembly for HeatCast-Global.

This module converts the 17 cached physical fields into the authoritative
26-channel fine-grid stack, keeping zarr access worker-local and applying only
fold-specific preprocessors fit from training years.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from build_driver_tables import parse_rmm_components_file
from cfm_mesh_train import compute_toa_insolation
from data_pipeline.build_cache import CACHE_CHANNELS, LazyGlobalZarrDataset
from global_targets import FoldFieldPreprocessor, label_to_date


ANOMALY_CHANNELS: Tuple[str, ...] = (
    "tmax",
    "t2m_mean",
    "swvl1",
    "swvl1_trailing20",
    "swvl2_trailing20",
    "sst",
    "z500",
    "z500_low20",
    "mslp",
    "t850",
)

GLOBAL_INPUT_CHANNELS: Tuple[str, ...] = (
    "tmax_t_anom", "tmax_tm1_anom", "tmax_tm2_anom",
    "t2m_mean_t_anom",
    "swvl1_t_anom", "swvl1_trailing20_anom", "swvl2_trailing20_anom",
    "sst_t_anom", "sst_valid",
    "z500_t_anom", "z500_low20_anom",
    "mslp_t_anom", "t850_t_anom", "q850_t", "u850_t", "v850_t", "z300_t",
    "orography", "land_mask",
    "sin_lat", "cos_lat", "sin_lon", "cos_lon",
    "doy_sin", "doy_cos", "toa_insolation_scaled",
)

VECTOR_INPUT_CHANNELS: Tuple[str, ...] = (
    "teleconnection_1", "teleconnection_2", "teleconnection_3",
    "teleconnection_4", "teleconnection_5", "rmm1", "rmm2", "mjo_amplitude",
)


def build_model_condition_vectors(
    date_labels: Sequence[int],
    base_teleconnections: np.ndarray,
    rmm_path,
) -> np.ndarray:
    """Append RMM1/RMM2/amplitude using the existing driver-table parser."""
    base = np.asarray(base_teleconnections, dtype=np.float32)
    if base.shape != (len(date_labels), 5):
        raise ValueError(f"Base teleconnections must be (time,5), got {base.shape}.")
    rmm = parse_rmm_components_file(rmm_path)
    output = np.empty((len(date_labels), len(VECTOR_INPUT_CHANNELS)), dtype=np.float32)
    output[:, :5] = base
    for index, label in enumerate(date_labels):
        key = int(label)
        if key not in rmm:
            raise RuntimeError(f"RMM table does not cover global cache date {key}.")
        output[index, 5:] = rmm[key]
    return output


def normalize_condition_vectors(vectors, train_indices: Sequence[int]):
    """Fold-safely normalize the eight vector channels from training indices."""
    values = np.asarray(vectors, dtype=np.float32)
    indices = np.asarray(tuple(int(value) for value in train_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Training indices cannot be empty for vector normalization.")
    mean = np.mean(values[indices], axis=0)
    std = np.std(values[indices], axis=0)
    std = np.where(std >= 1e-6, std, 1.0)
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def identity_preprocessor(shape: Sequence[int]) -> FoldFieldPreprocessor:
    """Return a zero-climatology/unit-scale fixture preprocessor for smoke tests."""
    spatial_shape = tuple(int(value) for value in shape)
    coefficient_count = 1 + 2 * 4
    return FoldFieldPreprocessor(
        channels=CACHE_CHANNELS,
        anomaly_channels=ANOMALY_CHANNELS,
        coefficients={
            channel: np.zeros((coefficient_count,) + spatial_shape, dtype=np.float32)
            for channel in ANOMALY_CHANNELS
        },
        means={channel: np.zeros(spatial_shape, dtype=np.float32) for channel in CACHE_CHANNELS},
        stds={channel: np.ones(spatial_shape, dtype=np.float32) for channel in CACHE_CHANNELS},
        train_years=(1979,),
    )


def _raw_map(context: np.ndarray, history_position: int) -> Mapping[str, np.ndarray]:
    return {
        channel: np.asarray(context[history_position, :, :, index], dtype=np.float32)
        for index, channel in enumerate(CACHE_CHANNELS)
    }


def assemble_global_tensors(
    context,
    target_tmax,
    history_dates: Sequence[int],
    target_dates: Sequence[int],
    lat,
    lon,
    condition_vector,
    preprocessor: FoldFieldPreprocessor,
):
    """Assemble one 26-channel sample without any disk access."""
    context = np.asarray(context, dtype=np.float32)
    target_tmax = np.asarray(target_tmax, dtype=np.float32)
    if context.shape[0] != 3 or context.shape[-1] != len(CACHE_CHANNELS):
        raise ValueError(f"Context must be (3,H,W,{len(CACHE_CHANNELS)}), got {context.shape}.")
    current = _raw_map(context, 0)
    history = [_raw_map(context, position) for position in range(3)]
    current_date = label_to_date(history_dates[0])

    tmax = [
        preprocessor.transform("tmax", history_dates[position], history[position]["tmax"])
        for position in range(3)
    ]
    spatial = []
    for channel in (
        "t2m_mean", "swvl1", "swvl1_trailing20", "swvl2_trailing20",
        "sst",
    ):
        spatial.append(preprocessor.transform(channel, history_dates[0], current[channel]))
    spatial.append(current["sst_valid"])
    for channel in ("z500", "z500_low20", "mslp", "t850", "q850", "u850", "v850", "z300"):
        spatial.append(preprocessor.transform(channel, history_dates[0], current[channel]))

    orography = current["orography"]
    finite_orography = orography[np.isfinite(orography)]
    orography_scale = float(np.std(finite_orography)) if finite_orography.size else 1.0
    orography_scale = orography_scale if orography_scale >= 1e-6 else 1.0
    spatial.append((orography - float(np.mean(finite_orography))) / orography_scale)
    spatial.append(current["land_mask"])

    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    spatial.extend((
        np.sin(np.deg2rad(lat_grid)), np.cos(np.deg2rad(lat_grid)),
        np.sin(np.deg2rad(lon_grid)), np.cos(np.deg2rad(lon_grid)),
    ))
    doy = current_date.timetuple().tm_yday
    spatial.extend((
        np.full(lat_grid.shape, np.sin(2.0 * np.pi * doy / 365.25), dtype=np.float32),
        np.full(lat_grid.shape, np.cos(2.0 * np.pi * doy / 365.25), dtype=np.float32),
        np.broadcast_to(compute_toa_insolation(lat, doy)[:, None] / 1361.0, lat_grid.shape),
    ))
    if len(spatial) != 23:
        raise AssertionError(f"Global spatial stack has {len(spatial)} channels, expected 23.")
    target = np.stack([
        preprocessor.transform("tmax", target_dates[index], target_tmax[index])
        for index in range(len(target_dates))
    ])
    return {
        "x_t": torch.from_numpy(tmax[0][None].astype(np.float32)),
        "x_tm1": torch.from_numpy(tmax[1][None].astype(np.float32)),
        "x_tm2": torch.from_numpy(tmax[2][None].astype(np.float32)),
        "spatial_c": torch.from_numpy(np.stack(spatial).astype(np.float32)),
        "target": torch.from_numpy(target.astype(np.float32)),
        "vector": torch.from_numpy(np.asarray(condition_vector, dtype=np.float32)),
        "global_fields": torch.empty((0,) + lat_grid.shape, dtype=torch.float32),
        "mask": torch.from_numpy(current["land_mask"][None].astype(np.float32)),
    }


class GlobalHeatCastDataset(LazyGlobalZarrDataset):
    """Worker-local zarr Dataset returning the legacy training tuple contract."""

    def __init__(self, *args, condition_vectors, preprocessor, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_vectors = np.asarray(condition_vectors, dtype=np.float32)
        if self.condition_vectors.shape != (int(self.metadata["shape"][0]), 8):
            raise ValueError("condition_vectors must have shape (cache_time, 8).")
        self.preprocessor = preprocessor
        self.indices = self.init_indices
        self._coordinate_cache = None
        self._coordinate_pid = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_coordinate_cache"] = None
        state["_coordinate_pid"] = None
        return state

    def _coordinates(self):
        pid = os.getpid()
        if self._coordinate_cache is None or self._coordinate_pid != pid:
            root = self._ensure_open()
            self._coordinate_cache = (
                np.asarray(root["lat"][:], dtype=np.float32),
                np.asarray(root["lon"][:], dtype=np.float32),
            )
            self._coordinate_pid = pid
        return self._coordinate_cache

    def __getitem__(self, item):
        raw = super().__getitem__(item)
        lat, lon = self._coordinates()
        assembled = assemble_global_tensors(
            raw["context"].numpy(),
            raw["target"].numpy(),
            raw["history_dates"],
            raw["target_dates"],
            lat,
            lon,
            self.condition_vectors[raw["init_index"]],
            self.preprocessor,
        )
        return (
            assembled["target"], assembled["x_t"], assembled["x_tm1"], assembled["x_tm2"],
            assembled["spatial_c"], assembled["vector"], assembled["global_fields"],
            raw["init_index"], assembled["mask"],
        )
