#!/usr/bin/env python3
"""
Shared utilities for Met-JEPA spatial-map pretraining, export, and probing.

The forecast model consumes only the exported JEPA maps as regional grid input.
The JEPA model itself may use initialization-time local and global physical
fields to construct those maps.
"""

import gc
import ast
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from netCDF4 import Dataset as NetCDFDataset
from torch.utils.data import Dataset


WORK_DIR = "/blue/nessie/mostafarezaali/Teleconnection"
TRAINING_DATA_PATH = os.path.join(WORK_DIR, "VDM_Training_Data_Extended_v2.nc")
GLOBAL_DATA_PATH = os.path.join(WORK_DIR, "Global_Coarse_Conditions.nc")
TOPO_PATH = os.path.join(WORK_DIR, "CONUS_topography_ETOPO2022_60s_on_model_grid.nc")
OUTPUT_DIR = WORK_DIR
DATA_CACHE = os.path.join(WORK_DIR, "data_cache")
SUB_TAG = "_sub30_40_-105_-90"
SUB_CACHE = os.path.join(WORK_DIR, "data_cache" + SUB_TAG)
CLIMATOLOGY_PATH = os.path.join(DATA_CACHE, "climatology_daily_1981_2015_sub30_40_-105_-90.npz")
JEPA_CHECKPOINT_PATH = os.path.join(DATA_CACHE, "met_jepa_spatial_sub30_40_-105_-90.pth")
JEPA_MAPS_PATH = os.path.join(DATA_CACHE, "jepa_spatial_maps_1981_2023_sub30_40_-105_-90.npy")
JEPA_META_PATH = os.path.join(DATA_CACHE, "jepa_spatial_maps_1981_2023_sub30_40_-105_-90_meta.npz")
JEPA_PROBE_REPORT = os.path.join(WORK_DIR, "test_prediction_plots", "jepa_probe_report.txt")
SUPERVISED_CHECKPOINT_PATH = os.path.join(DATA_CACHE, "supervised_spatial_encoder_sub30_40_-105_-90.pth")
SUPERVISED_MAPS_PATH = os.path.join(DATA_CACHE, "supervised_spatial_maps_1981_2023_sub30_40_-105_-90.npy")
SUPERVISED_META_PATH = os.path.join(DATA_CACHE, "supervised_spatial_maps_1981_2023_sub30_40_-105_-90_meta.npz")
SUPERVISED_PROBE_REPORT = os.path.join(WORK_DIR, "test_prediction_plots", "supervised_spatial_probe_report.txt")
LATENT_ROLLOUT_CHECKPOINT_PATH = os.path.join(DATA_CACHE, "latent_rollout_encoder_sub30_40_-105_-90.pth")
LATENT_ROLLOUT_MAPS_PATH = os.path.join(DATA_CACHE, "latent_rollout_maps_1981_2023_sub30_40_-105_-90.npy")
LATENT_ROLLOUT_META_PATH = os.path.join(DATA_CACHE, "latent_rollout_maps_1981_2023_sub30_40_-105_-90_meta.npz")
LATENT_ROLLOUT_PROBE_REPORT = os.path.join(WORK_DIR, "test_prediction_plots", "latent_rollout_probe_report.txt")

LAT_SLICE = slice(124, 373)
LON_SLICE = slice(502, 803)
IMAGE_SIZE = (249, 301)
GLOBAL_SIZE = (181, 360)
TRAIN_YEARS = list(range(1981, 2016))
VAL_YEARS = list(range(2016, 2020))
TEST_YEARS = list(range(2020, 2024))
LEAD_TIME = 15
ROLLOUT_LEADS = (3, 6, 9, 12, 15)
BASE_DATE = datetime(1981, 5, 1)

LOCAL_PHYSICS_KEYS = [
    "geopotential",
    "soil_moisture",
    "slp",
    "temperature_2m",
    "specific_humidity_850",
    "temperature_850",
    "u_wind_850",
    "v_wind_850",
    "geopotential_300",
]

GLOBAL_VARIABLES = [
    "sst",
    "olr",
    "geopotential_200",
    "u_wind_200",
    "total_column_water_vapour",
    "v_wind_200",
    "geopotential_500",
    "temperature_850",
    "temperature_2m_global",
]

EXTENDED_GLOBAL_VARIABLES_PATH = os.path.join(DATA_CACHE, "extended_global_variables.txt")


def _read_extended_global_variable_report(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    global_path = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("GLOBAL_DATA_PATH"):
            global_path = ast.literal_eval(stripped.split("=", 1)[1].strip())
            break

    marker = "NEW_GLOBAL_VARIABLES"
    if marker not in text:
        raise ValueError(f"{path} does not define NEW_GLOBAL_VARIABLES")
    start = text.index(marker)
    bracket_start = text.index("[", start)
    bracket_end = text.index("]", bracket_start) + 1
    new_vars = ast.literal_eval(text[bracket_start:bracket_end])
    if not isinstance(new_vars, list) or not all(isinstance(v, str) for v in new_vars):
        raise ValueError(f"Invalid NEW_GLOBAL_VARIABLES in {path}")
    return global_path, new_vars


def apply_extended_global_fields():
    global GLOBAL_DATA_PATH, GLOBAL_VARIABLES
    if not os.path.exists(EXTENDED_GLOBAL_VARIABLES_PATH):
        return

    global_path, new_vars = _read_extended_global_variable_report(EXTENDED_GLOBAL_VARIABLES_PATH)
    if global_path:
        GLOBAL_DATA_PATH = global_path

    seen = set(GLOBAL_VARIABLES)
    added = []
    for name in new_vars:
        if name not in seen:
            GLOBAL_VARIABLES.append(name)
            seen.add(name)
            added.append(name)

    print(f"Loaded extended global-field report: {EXTENDED_GLOBAL_VARIABLES_PATH}")
    print(f"  GLOBAL_DATA_PATH: {GLOBAL_DATA_PATH}")
    print(f"  Added global variables: {len(added)}")
    print(f"  Total global channels: {len(GLOBAL_VARIABLES)}")


apply_extended_global_fields()


def time_to_date(tv):
    return BASE_DATE + timedelta(days=float(tv))


def compute_doy_array(time_values):
    doys = np.empty(len(time_values), dtype=np.int32)
    for i, tv in enumerate(time_values):
        doys[i] = time_to_date(tv).timetuple().tm_yday
    return doys


def compute_toa_insolation(lat_deg, doy):
    s0 = 1361.0
    lat_rad = np.radians(lat_deg)
    decl = np.radians(23.4393 * np.sin(np.radians(360.0 / 365.0 * (float(doy) + 284.0))))
    cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    omega_s = np.arccos(cos_omega)
    q = (s0 / np.pi) * (
        omega_s * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(omega_s)
    )
    return q.astype(np.float32)


def detect_continuous_runs(time_values):
    runs, start = [], 0
    for i in range(1, len(time_values)):
        if time_values[i] - time_values[i - 1] > 1.5:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(time_values) - 1))
    return runs


def build_valid_indices(runs, lead_time, min_history=2):
    indices = []
    for s, e in runs:
        for t in range(s + min_history, e - lead_time + 1):
            indices.append(t)
    return indices


def split_indices(time_values):
    runs = detect_continuous_runs(time_values)
    valid = build_valid_indices(runs, LEAD_TIME, min_history=2)
    years = np.array([time_to_date(tv).year for tv in time_values])
    train_set, val_set, test_set = set(TRAIN_YEARS), set(VAL_YEARS), set(TEST_YEARS)
    train = [i for i in valid if years[i] in train_set]
    val = [i for i in valid if years[i] in val_set]
    test = [i for i in valid if years[i] in test_set]
    return train, val, test, valid


def subdomain_tag():
    return SUB_TAG


def ensure_data_cache():
    """Create the same subdomain cache layout used by the forecast training script."""
    os.makedirs(SUB_CACHE, exist_ok=True)
    os.makedirs(DATA_CACHE, exist_ok=True)
    paths = {
        "heat_index": os.path.join(SUB_CACHE, "heat_index.npy"),
        "geopotential": os.path.join(SUB_CACHE, "geopotential.npy"),
        "soil_moisture": os.path.join(SUB_CACHE, "soil_moisture.npy"),
        "slp": os.path.join(SUB_CACHE, "slp.npy"),
        "cond_train": os.path.join(SUB_CACHE, "cond_train.npy"),
        "topography": os.path.join(SUB_CACHE, "topography.npy"),
        "temperature_2m": os.path.join(SUB_CACHE, "temperature_2m.npy"),
        "specific_humidity_850": os.path.join(SUB_CACHE, "specific_humidity_850.npy"),
        "temperature_850": os.path.join(SUB_CACHE, "temperature_850.npy"),
        "u_wind_850": os.path.join(SUB_CACHE, "u_wind_850.npy"),
        "v_wind_850": os.path.join(SUB_CACHE, "v_wind_850.npy"),
        "geopotential_300": os.path.join(SUB_CACHE, "geopotential_300.npy"),
        "time_values": os.path.join(SUB_CACHE, "time_values.npy"),
    }
    global_dir = os.path.join(SUB_CACHE, "global")
    os.makedirs(global_dir, exist_ok=True)
    global_paths = {name: os.path.join(global_dir, f"{name}.npy") for name in GLOBAL_VARIABLES}

    if not all(os.path.exists(p) for p in paths.values()):
        print("Creating subdomain data cache for Met-JEPA...")
        with NetCDFDataset(TRAINING_DATA_PATH, "r") as nc:
            hi_raw = nc.variables["t2m_prism"][:]
            if hi_raw.ndim == 4:
                hi = np.array(hi_raw[:, :, 0, :], dtype=np.float32)[LAT_SLICE, LON_SLICE, :]
            else:
                hi = np.array(hi_raw, dtype=np.float32)[LAT_SLICE, LON_SLICE, :]
            np.save(paths["heat_index"], hi)
            del hi, hi_raw
            gc.collect()

            def load3(name):
                raw = np.array(nc.variables[name][:], dtype=np.float32)
                out = raw[LAT_SLICE, LON_SLICE, :]
                del raw
                return out

            def load4(name):
                raw = np.array(nc.variables[name][:], dtype=np.float32)
                out = raw[LAT_SLICE, LON_SLICE, :, :]
                del raw
                return out

            np.save(paths["geopotential"], load4("geopotential"))
            np.save(paths["soil_moisture"], load3("soil_moisture"))
            np.save(paths["slp"], load3("sea_level_pressure"))
            np.save(paths["cond_train"], np.array(nc.variables["CondTrain"][:], dtype=np.float32))
            np.save(paths["temperature_2m"], load3("temperature_2m"))
            np.save(paths["specific_humidity_850"], load3("specific_humidity_850"))
            np.save(paths["temperature_850"], load3("temperature_850"))
            np.save(paths["u_wind_850"], load3("u_wind_850"))
            np.save(paths["v_wind_850"], load3("v_wind_850"))
            np.save(paths["geopotential_300"], load3("geopotential_300"))
            np.save(paths["time_values"], np.array(nc.variables["time"][:], dtype=np.float64))

        with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
            topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
            topo = np.flipud(topo)
        if topo.shape == (1405, 621):
            topo = topo.T
        if topo.shape != (621, 1405):
            raise ValueError(f"Unexpected topography shape: {topo.shape}")
        np.save(paths["topography"], np.ascontiguousarray(topo[LAT_SLICE, LON_SLICE]))
        print("Subdomain data cache written.")

    if not all(os.path.exists(p) for p in global_paths.values()):
        print("Creating global teleconnection cache for Met-JEPA...")
        with NetCDFDataset(GLOBAL_DATA_PATH, "r") as nc_g:
            for name in GLOBAL_VARIABLES:
                data = np.array(nc_g.variables[name][:], dtype=np.float32)
                if data.ndim == 4:
                    data = data[:, 0, :, :]
                data = np.transpose(data, (1, 2, 0))
                data = np.nan_to_num(data, nan=0.0)
                np.save(global_paths[name], data)
                del data
        print("Global teleconnection cache written.")


def load_shared_arrays(mmap=True):
    ensure_data_cache()
    mode = "r" if mmap else None
    shared = {
        "heat_index": np.load(os.path.join(SUB_CACHE, "heat_index.npy"), mmap_mode=mode),
        "geopotential": np.load(os.path.join(SUB_CACHE, "geopotential.npy"), mmap_mode=mode),
        "soil_moisture": np.load(os.path.join(SUB_CACHE, "soil_moisture.npy"), mmap_mode=mode),
        "slp": np.load(os.path.join(SUB_CACHE, "slp.npy"), mmap_mode=mode),
        "cond_train": np.load(os.path.join(SUB_CACHE, "cond_train.npy"), mmap_mode=mode),
        "topography": np.load(os.path.join(SUB_CACHE, "topography.npy"), mmap_mode=mode),
        "temperature_2m": np.load(os.path.join(SUB_CACHE, "temperature_2m.npy"), mmap_mode=mode),
        "specific_humidity_850": np.load(os.path.join(SUB_CACHE, "specific_humidity_850.npy"), mmap_mode=mode),
        "temperature_850": np.load(os.path.join(SUB_CACHE, "temperature_850.npy"), mmap_mode=mode),
        "u_wind_850": np.load(os.path.join(SUB_CACHE, "u_wind_850.npy"), mmap_mode=mode),
        "v_wind_850": np.load(os.path.join(SUB_CACHE, "v_wind_850.npy"), mmap_mode=mode),
        "geopotential_300": np.load(os.path.join(SUB_CACHE, "geopotential_300.npy"), mmap_mode=mode),
        "time_values": np.load(os.path.join(SUB_CACHE, "time_values.npy"), mmap_mode=mode),
    }
    global_data = OrderedDict()
    for name in GLOBAL_VARIABLES:
        global_data[name] = np.load(os.path.join(SUB_CACHE, "global", f"{name}.npy"), mmap_mode=mode)
    shared["global_data"] = global_data
    return shared


def load_climatology():
    if not os.path.exists(CLIMATOLOGY_PATH):
        raise FileNotFoundError(f"Missing climatology at {CLIMATOLOGY_PATH}. Run compute_climatology.py first.")
    clim = np.load(CLIMATOLOGY_PATH)
    return {
        "clim_mean": clim["clim_mean"],
        "clim_std": clim["clim_std"],
        "valid_doys": clim["valid_doys"].astype(bool),
    }


def _physics_array(shared, key, t):
    if key == "geopotential":
        return np.array(shared[key][:, :, 0, t], dtype=np.float32)
    return np.array(shared[key][:, :, t], dtype=np.float32)


def _compute_mean_std(arr):
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return mean, max(std, 1e-6)


def compute_norm_stats(shared, clim, train_indices):
    time_values = np.array(shared["time_values"])
    doys = compute_doy_array(time_values)
    doy_idx = np.clip(doys - 1, 0, 365)
    train_arr = np.array(train_indices, dtype=np.int64)

    physics_mean, physics_std = [], []
    for key in LOCAL_PHYSICS_KEYS:
        if key == "geopotential":
            data = shared[key][:, :, 0, train_arr]
        else:
            data = shared[key][:, :, train_arr]
        m, s = _compute_mean_std(data)
        physics_mean.append(m)
        physics_std.append(s)

    topo = np.array(shared["topography"], dtype=np.float32)
    topo_land = topo[np.isfinite(topo)]
    topo_mean, topo_std = _compute_mean_std(topo_land)

    lat_1d = np.linspace(30.0, 40.0, IMAGE_SIZE[0], dtype=np.float32)
    toa_samples = np.stack([compute_toa_insolation(lat_1d, doys[i]) for i in train_arr[:500]])
    toa_mean, toa_std = _compute_mean_std(toa_samples)

    sample_t = int(train_arr[0])
    land = np.isfinite(shared["heat_index"][:, :, sample_t]) & (shared["heat_index"][:, :, sample_t] != 0.0)
    clim_train = clim["clim_mean"][doy_idx[train_arr]]
    cstd_train = clim["clim_std"][doy_idx[train_arr]]
    clim_ch_mean, clim_ch_std = _compute_mean_std(clim_train[:, land])
    clim_std_mean, clim_std_std = _compute_mean_std(cstd_train[:, land])

    global_mean, global_std = [], []
    for name, data in shared["global_data"].items():
        g = data[:, :, train_arr]
        m, s = _compute_mean_std(g)
        global_mean.append(m)
        global_std.append(s)

    stats = {
        "physics_mean": np.array(physics_mean, dtype=np.float32),
        "physics_std": np.array(physics_std, dtype=np.float32),
        "topo_mean": np.float32(topo_mean),
        "topo_std": np.float32(topo_std),
        "toa_mean": np.float32(toa_mean),
        "toa_std": np.float32(toa_std),
        "clim_ch_mean": np.float32(clim_ch_mean),
        "clim_ch_std": np.float32(clim_ch_std),
        "clim_std_ch_mean": np.float32(clim_std_mean),
        "clim_std_ch_std": np.float32(clim_std_std),
        "global_mean": np.array(global_mean, dtype=np.float32),
        "global_std": np.array(global_std, dtype=np.float32),
    }
    return stats


def anomaly_field(shared, clim, time_idx, doy_idx):
    valid_doys = clim["valid_doys"]
    if doy_idx < 0 or doy_idx >= 366 or not bool(valid_doys[doy_idx]):
        raise ValueError(f"DOY index {doy_idx} is outside populated MJJAS climatology.")
    raw = np.array(shared["heat_index"][:, :, time_idx], dtype=np.float32)
    cmean = np.array(clim["clim_mean"][doy_idx], dtype=np.float32)
    cstd = np.array(clim["clim_std"][doy_idx], dtype=np.float32)
    valid = np.isfinite(raw) & (raw != 0.0) & np.isfinite(cmean) & np.isfinite(cstd)
    out = np.zeros_like(raw, dtype=np.float32)
    out[valid] = (raw[valid] - cmean[valid]) / (cstd[valid] + 1e-6)
    return out


def build_local_stack(shared, clim, stats, anchor_t, doys, doy_idx):
    h, w = IMAGE_SIZE
    anchor_doy = int(doy_idx[anchor_t])
    x0 = anomaly_field(shared, clim, anchor_t, anchor_doy)
    x1 = anomaly_field(shared, clim, anchor_t - 1, int(doy_idx[anchor_t - 1]))
    x2 = anomaly_field(shared, clim, anchor_t - 2, int(doy_idx[anchor_t - 2]))

    physics = []
    for i, key in enumerate(LOCAL_PHYSICS_KEYS):
        field = _physics_array(shared, key, anchor_t)
        field = np.nan_to_num(field, nan=0.0)
        field = (field - stats["physics_mean"][i]) / (stats["physics_std"][i] + 1e-8)
        physics.append(field.astype(np.float32))

    topo = np.array(shared["topography"], dtype=np.float32)
    topo = (topo - stats["topo_mean"]) / (stats["topo_std"] + 1e-8)

    lat_1d = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    lon_1d = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    doy = float(doys[anchor_t])
    doy_sin = np.full((h, w), np.sin(2.0 * np.pi * doy / 365.25), dtype=np.float32)
    doy_cos = np.full((h, w), np.cos(2.0 * np.pi * doy / 365.25), dtype=np.float32)

    lat_deg = np.linspace(30.0, 40.0, h, dtype=np.float32)
    toa_1d = compute_toa_insolation(lat_deg, doy)
    toa = np.broadcast_to(toa_1d[:, None], (h, w)).astype(np.float32)
    toa = (toa - stats["toa_mean"]) / (stats["toa_std"] + 1e-8)

    raw = np.array(shared["heat_index"][:, :, anchor_t], dtype=np.float32)
    land = (np.isfinite(raw) & (raw != 0.0)).astype(np.float32)

    cmean = np.array(clim["clim_mean"][anchor_doy], dtype=np.float32)
    cstd = np.array(clim["clim_std"][anchor_doy], dtype=np.float32)
    cmean = (cmean - stats["clim_ch_mean"]) / (stats["clim_ch_std"] + 1e-8)
    cstd = (cstd - stats["clim_std_ch_mean"]) / (stats["clim_std_ch_std"] + 1e-8)

    stack = [x0, x1, x2] + physics + [
        topo.astype(np.float32),
        lat_grid.astype(np.float32),
        lon_grid.astype(np.float32),
        doy_sin,
        doy_cos,
        toa.astype(np.float32),
        land.astype(np.float32),
        cmean.astype(np.float32),
        cstd.astype(np.float32),
    ]
    arr = np.stack(stack, axis=0).astype(np.float32)
    arr[:, land < 0.5] = 0.0
    return arr


def build_global_stack(shared, stats, time_idx):
    chans = []
    for i, (_, data) in enumerate(shared["global_data"].items()):
        g = np.array(data[:, :, time_idx], dtype=np.float32)
        g = (g - stats["global_mean"][i]) / (stats["global_std"][i] + 1e-8)
        chans.append(g)
    return np.stack(chans, axis=0).astype(np.float32)


@dataclass
class JEPANormBundle:
    stats: dict
    doys: np.ndarray
    doy_idx: np.ndarray


class MetJEPADataset(Dataset):
    def __init__(self, shared, clim, stats, indices, return_target=True):
        self.shared = shared
        self.clim = clim
        self.stats = stats
        self.indices = list(indices)
        self.return_target = return_target
        self.time_values = np.array(shared["time_values"])
        self.doys = compute_doy_array(self.time_values)
        self.doy_idx = np.clip(self.doys - 1, 0, 365)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        context = build_local_stack(self.shared, self.clim, self.stats, t, self.doys, self.doy_idx)
        global_fields = build_global_stack(self.shared, self.stats, t)
        if not self.return_target:
            return torch.from_numpy(context), torch.from_numpy(global_fields), t
        target_t = t + LEAD_TIME
        target = build_local_stack(self.shared, self.clim, self.stats, target_t, self.doys, self.doy_idx)
        y = anomaly_field(self.shared, self.clim, target_t, int(self.doy_idx[target_t]))
        raw = np.array(self.shared["heat_index"][:, :, t], dtype=np.float32)
        mask = (np.isfinite(raw) & (raw != 0.0)).astype(np.float32)
        return (
            torch.from_numpy(context),
            torch.from_numpy(target),
            torch.from_numpy(global_fields),
            torch.from_numpy(y).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            t,
        )


class LatentRolloutDataset(Dataset):
    """Multi-lead dataset for teacher-forced latent rollout pretraining.

    Export-time use sets ``return_targets=False``; then only initialization-time
    context/global fields at ``t`` are loaded. Future local states are used only
    during pretraining as EMA target-encoder inputs.
    """

    def __init__(self, shared, clim, stats, indices, leads=ROLLOUT_LEADS, return_targets=True):
        self.shared = shared
        self.clim = clim
        self.stats = stats
        self.indices = list(indices)
        self.leads = tuple(int(x) for x in leads)
        self.return_targets = return_targets
        self.time_values = np.array(shared["time_values"])
        self.doys = compute_doy_array(self.time_values)
        self.doy_idx = np.clip(self.doys - 1, 0, 365)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        context = build_local_stack(self.shared, self.clim, self.stats, t, self.doys, self.doy_idx)
        global_fields = build_global_stack(self.shared, self.stats, t)
        if not self.return_targets:
            return torch.from_numpy(context), torch.from_numpy(global_fields), t

        targets, anomalies, masks = [], [], []
        for lead in self.leads:
            target_t = t + int(lead)
            target_doy = int(self.doy_idx[target_t])
            targets.append(build_local_stack(self.shared, self.clim, self.stats, target_t, self.doys, self.doy_idx))
            anomalies.append(anomaly_field(self.shared, self.clim, target_t, target_doy))
            raw = np.array(self.shared["heat_index"][:, :, target_t], dtype=np.float32)
            masks.append((np.isfinite(raw) & (raw != 0.0)).astype(np.float32))

        return (
            torch.from_numpy(context),
            torch.from_numpy(np.stack(targets, axis=0).astype(np.float32)),
            torch.from_numpy(global_fields),
            torch.from_numpy(np.stack(anomalies, axis=0).astype(np.float32)).unsqueeze(1),
            torch.from_numpy(np.stack(masks, axis=0).astype(np.float32)).unsqueeze(1),
            torch.tensor(self.leads, dtype=torch.float32),
            t,
        )


class ConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, dilation=1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class PeriodicLonConv2d(nn.Conv2d):
    """Conv2d for global fields: circular in longitude, non-periodic in latitude."""

    def __init__(self, *args, **kwargs):
        padding = kwargs.pop("padding", 0)
        if isinstance(padding, int):
            self.pad_lat = padding
            self.pad_lon = padding
        else:
            self.pad_lat = int(padding[0])
            self.pad_lon = int(padding[1])
        super().__init__(*args, padding=0, **kwargs)

    def forward(self, x):
        if self.pad_lon > 0:
            x = F.pad(x, (self.pad_lon, self.pad_lon, 0, 0), mode="circular")
        if self.pad_lat > 0:
            x = F.pad(x, (0, 0, self.pad_lat, self.pad_lat), mode="replicate")
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, map_channels=32, hidden=96, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ConvBlock(hidden, dropout=dropout, dilation=1),
            ConvBlock(hidden, dropout=dropout, dilation=2),
            ConvBlock(hidden, dropout=dropout, dilation=4),
            ConvBlock(hidden, dropout=dropout, dilation=1),
        )
        self.proj = nn.Conv2d(hidden, map_channels, 1)

    def forward(self, x):
        return self.proj(self.blocks(self.stem(x)))


class GlobalContextEncoder(nn.Module):
    def __init__(self, in_channels=9, map_channels=32, hidden=64):
        super().__init__()
        self.map_channels = map_channels
        self.feature_net = nn.Sequential(
            PeriodicLonConv2d(in_channels, hidden, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            PeriodicLonConv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            PeriodicLonConv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            PeriodicLonConv2d(hidden * 2, hidden * 2, 3, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
        )
        self.spatial_proj = nn.Conv2d(hidden * 2, map_channels, 1)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.film_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, map_channels * 2),
        )

    def forward(self, x, out_hw):
        feat = self.feature_net(x)
        spatial = self.spatial_proj(feat)
        spatial = F.interpolate(spatial, size=out_hw, mode="bilinear", align_corners=False)
        film = self.film_proj(self.global_pool(feat))
        scale, shift = film.chunk(2, dim=1)
        return spatial, scale.unsqueeze(-1).unsqueeze(-1), shift.unsqueeze(-1).unsqueeze(-1)


class MapPredictor(nn.Module):
    def __init__(self, map_channels=32, hidden=96, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(map_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            ConvBlock(hidden, dropout=dropout, dilation=2),
            ConvBlock(hidden, dropout=dropout, dilation=4),
            nn.Conv2d(hidden, map_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class MetJEPAModel(nn.Module):
    def __init__(self, local_channels=21, global_channels=9, map_channels=32, hidden=96, dropout=0.05):
        super().__init__()
        self.local_channels = local_channels
        self.global_channels = global_channels
        self.map_channels = map_channels
        self.online_encoder = SpatialEncoder(local_channels, map_channels, hidden=hidden, dropout=dropout)
        self.target_encoder = SpatialEncoder(local_channels, map_channels, hidden=hidden, dropout=dropout)
        self.global_encoder = GlobalContextEncoder(global_channels, map_channels)
        self.predictor = MapPredictor(map_channels, hidden=hidden, dropout=dropout)
        self._init_target_encoder()

    @torch.no_grad()
    def _init_target_encoder(self):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(decay).add_(p_o.data, alpha=1.0 - decay)

    def predict_maps(self, context, global_fields):
        z = self.online_encoder(context)
        g_spatial, g_scale, g_shift = self.global_encoder(global_fields, z.shape[-2:])
        z = z + g_spatial
        z = z * (1.0 + torch.tanh(g_scale)) + g_shift
        return self.predictor(z)

    def forward(self, context, target, global_fields):
        z_pred = self.predict_maps(context, global_fields)
        with torch.no_grad():
            z_target = self.target_encoder(target)
        return z_pred, z_target


class SupervisedSpatialFeatureModel(nn.Module):
    """Context/global encoder trained directly against target anomaly.

    The exported artifact is the 32-channel map `z`, not the 1-channel decoder
    prediction. The decoder is deliberately shallow so the maps must remain
    easy to probe and useful as MeshFlowNet inputs.
    """

    def __init__(self, local_channels=21, global_channels=9, map_channels=32, hidden=96, dropout=0.15):
        super().__init__()
        self.local_channels = local_channels
        self.global_channels = global_channels
        self.map_channels = map_channels
        self.online_encoder = SpatialEncoder(local_channels, map_channels, hidden=hidden, dropout=dropout)
        self.global_encoder = GlobalContextEncoder(global_channels, map_channels)
        self.refiner = MapPredictor(map_channels, hidden=hidden, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(map_channels, 1, kernel_size=1),
        )

    def encode_maps(self, context, global_fields):
        z = self.online_encoder(context)
        g_spatial, g_scale, g_shift = self.global_encoder(global_fields, z.shape[-2:])
        z = z + g_spatial
        z = z * (1.0 + torch.tanh(g_scale)) + g_shift
        return self.refiner(z)

    def forward(self, context, global_fields):
        z = self.encode_maps(context, global_fields)
        pred = self.decoder(z).clamp(-4.0, 4.0)
        return pred, z


class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h):
        gate_in = torch.cat([x, h], dim=1)
        update, reset = self.gates(gate_in).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * cand


class LatentRolloutFeatureModel(nn.Module):
    """Teacher-forced latent trajectory model for forecast-ready spatial maps.

    The online context/global encoders see only initialization-time fields.
    During pretraining, the EMA target encoder embeds true future regional
    states at the configured leads. During export, only ``predict_lead15_maps``
    is used, so no future target fields are touched.
    """

    def __init__(
        self,
        local_channels=21,
        global_channels=9,
        map_channels=32,
        hidden=96,
        dropout=0.10,
        leads=ROLLOUT_LEADS,
    ):
        super().__init__()
        self.local_channels = local_channels
        self.global_channels = global_channels
        self.map_channels = map_channels
        self.leads = tuple(int(x) for x in leads)
        self.online_encoder = SpatialEncoder(local_channels, map_channels, hidden=hidden, dropout=dropout)
        self.target_encoder = SpatialEncoder(local_channels, map_channels, hidden=hidden, dropout=dropout)
        self.global_encoder = GlobalContextEncoder(global_channels, map_channels)
        self.init_refiner = MapPredictor(map_channels, hidden=hidden, dropout=dropout)
        self.step_cell = ConvGRUCell(map_channels)
        self.step_refiner = MapPredictor(map_channels, hidden=hidden, dropout=dropout)
        self.lead_embedding = nn.Embedding(max(self.leads) + 1, map_channels)
        self.anomaly_heads = nn.ModuleDict({
            str(lead): nn.Sequential(
                nn.Dropout2d(dropout),
                nn.Conv2d(map_channels, 1, kernel_size=1),
            )
            for lead in self.leads
        })
        self._init_target_encoder()

    @torch.no_grad()
    def _init_target_encoder(self):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(decay).add_(p_o.data, alpha=1.0 - decay)

    def encode_initial(self, context, global_fields):
        z = self.online_encoder(context)
        g_spatial, g_scale, g_shift = self.global_encoder(global_fields, z.shape[-2:])
        z = z + g_spatial
        z = z * (1.0 + torch.tanh(g_scale)) + g_shift
        return self.init_refiner(z)

    def encode_targets(self, target_contexts):
        b, n, c, h, w = target_contexts.shape
        flat = target_contexts.reshape(b * n, c, h, w)
        z = self.target_encoder(flat)
        return z.reshape(b, n, self.map_channels, h, w)

    def _step(self, h, lead):
        lead_ids = torch.full((h.shape[0],), int(lead), dtype=torch.long, device=h.device)
        lead_bias = self.lead_embedding(lead_ids).view(h.shape[0], self.map_channels, 1, 1)
        stepped = self.step_cell(h + lead_bias, h)
        return stepped + 0.5 * self.step_refiner(stepped)

    def _decode(self, z, lead):
        return self.anomaly_heads[str(int(lead))](z).clamp(-4.0, 4.0)

    def rollout(self, context, global_fields, target_contexts=None, teacher_forcing_prob=0.0, leads=None):
        leads = tuple(int(x) for x in (self.leads if leads is None else leads))
        h = self.encode_initial(context, global_fields)
        with torch.no_grad():
            z_targets = self.encode_targets(target_contexts) if target_contexts is not None else None

        z_preds, y_preds = [], []
        for i, lead in enumerate(leads):
            h = self._step(h, lead)
            z_preds.append(h)
            y_preds.append(self._decode(h, lead))
            if z_targets is not None and i < len(leads) - 1 and teacher_forcing_prob > 0.0:
                use_teacher = (torch.rand((h.shape[0], 1, 1, 1), device=h.device) < teacher_forcing_prob).float()
                h = use_teacher * z_targets[:, i] + (1.0 - use_teacher) * h

        z_preds = torch.stack(z_preds, dim=1)
        y_preds = torch.stack(y_preds, dim=1)
        return z_preds, z_targets, y_preds

    def predict_lead15_maps(self, context, global_fields):
        z_preds, _, _ = self.rollout(context, global_fields, target_contexts=None, teacher_forcing_prob=0.0, leads=self.leads)
        return z_preds[:, -1]

    def forward(self, context, global_fields, target_contexts=None, teacher_forcing_prob=0.0):
        return self.rollout(context, global_fields, target_contexts=target_contexts, teacher_forcing_prob=teacher_forcing_prob)


def random_spatial_jitter(x, max_shift=3):
    if max_shift <= 0:
        return x
    shifts_y = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    shifts_x = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    out = []
    for b in range(x.shape[0]):
        out.append(torch.roll(x[b], shifts=(int(shifts_y[b]), int(shifts_x[b])), dims=(-2, -1)))
    return torch.stack(out, dim=0)


def random_blur_mix(x, p=0.15):
    if p <= 0 or torch.rand((), device=x.device).item() > p:
        return x
    blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return 0.65 * x + 0.35 * blurred


def aggressive_context_augment(x, noise_std=0.05):
    """Aggressive masking so the context encoder cannot pass x_t through unchanged."""
    b, c, h, w = x.shape
    device = x.device
    x = random_spatial_jitter(x, max_shift=3)

    # Per-channel dropout. Anomaly history receives stronger dropout.
    probs = torch.full((c,), 0.35, device=device)
    probs[:3] = 0.60
    channel_keep = (torch.rand((b, c, 1, 1), device=device) > probs.view(1, c, 1, 1)).float()
    x = x * channel_keep

    # Whole group dropout: anomaly history, local physics, static/seasonal.
    groups = [(0, 3, 0.35), (3, 12, 0.25), (12, c, 0.10)]
    for start, end, p in groups:
        drop = (torch.rand((b, 1, 1, 1), device=device) < p).float()
        x[:, start:end] = x[:, start:end] * (1.0 - drop)

    # Large block masks across all local channels.
    for _ in range(4):
        bh = torch.randint(max(8, h // 8), max(9, h // 3), (b,), device=device)
        bw = torch.randint(max(8, w // 8), max(9, w // 3), (b,), device=device)
        y0 = torch.randint(0, max(1, h - int(bh.max().item())), (b,), device=device)
        x0 = torch.randint(0, max(1, w - int(bw.max().item())), (b,), device=device)
        for i in range(b):
            y1 = min(h, int(y0[i] + bh[i]))
            x1 = min(w, int(x0[i] + bw[i]))
            x[i, :, int(y0[i]):y1, int(x0[i]):x1] = 0.0

    x = random_blur_mix(x, p=0.15)
    x = x + noise_std * torch.randn_like(x)
    return x


def weak_target_augment(x, noise_std=0.005):
    return x + noise_std * torch.randn_like(x)


def vicreg_map_loss(z, variance_threshold=0.01, var_weight=1.0, cov_weight=0.02):
    b, c, h, w = z.shape
    spatial_var = z.float().var(dim=(-2, -1), unbiased=False)
    var_loss = F.relu(variance_threshold - spatial_var).mean()

    flat = z.permute(0, 2, 3, 1).reshape(-1, c).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / c
    return var_weight * var_loss + cov_weight * cov_loss, spatial_var.detach()


def channel_variance_diagnostics(z, threshold=0.01):
    var = z.float().var(dim=(-2, -1), unbiased=False)
    mean_var = var.mean(dim=0)
    collapsed = mean_var < threshold
    return {
        "min_channel_variance": float(mean_var.min().item()),
        "mean_channel_variance": float(mean_var.mean().item()),
        "collapsed_channels": int(collapsed.sum().item()),
        "active_channels": int((~collapsed).sum().item()),
        "per_channel_variance": mean_var.detach().cpu().numpy().astype(np.float32),
    }


def save_checkpoint(path, model, optimizer, epoch, stats, train_indices, val_indices, args_dict, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "stats": stats,
        "train_indices": np.array(train_indices, dtype=np.int64),
        "val_indices": np.array(val_indices, dtype=np.int64),
        "args": args_dict,
        "best_val_loss": float(best_val_loss),
        "local_channels": int(model.local_channels),
        "global_channels": int(model.global_channels),
        "map_channels": int(model.map_channels),
    }
    torch.save(payload, path)


def save_supervised_checkpoint(path, model, optimizer, epoch, stats, train_indices, val_indices, args_dict, best_metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "stats": stats,
        "train_indices": np.array(train_indices, dtype=np.int64),
        "val_indices": np.array(val_indices, dtype=np.int64),
        "args": args_dict,
        "best_metric": float(best_metric),
        "local_channels": int(model.local_channels),
        "global_channels": int(model.global_channels),
        "map_channels": int(model.map_channels),
        "model_type": "supervised_spatial_feature_encoder",
    }
    torch.save(payload, path)


def save_latent_rollout_checkpoint(path, model, optimizer, epoch, stats, train_indices, val_indices, args_dict, best_metric, best_val_r2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "stats": stats,
        "train_indices": np.array(train_indices, dtype=np.int64),
        "val_indices": np.array(val_indices, dtype=np.int64),
        "args": args_dict,
        "best_metric": float(best_metric),
        "best_val_r2": float(best_val_r2),
        "local_channels": int(model.local_channels),
        "global_channels": int(model.global_channels),
        "map_channels": int(model.map_channels),
        "rollout_leads": np.array(model.leads, dtype=np.int32),
        "model_type": "teacher_forced_latent_rollout_encoder",
    }
    torch.save(payload, path)


def load_jepa_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    model = MetJEPAModel(
        local_channels=int(ckpt.get("local_channels", 21)),
        global_channels=int(ckpt.get("global_channels", 9)),
        map_channels=int(ckpt.get("map_channels", 32)),
        hidden=int(args.get("hidden", 96)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_supervised_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    model = SupervisedSpatialFeatureModel(
        local_channels=int(ckpt.get("local_channels", 21)),
        global_channels=int(ckpt.get("global_channels", 9)),
        map_channels=int(ckpt.get("map_channels", 32)),
        hidden=int(args.get("hidden", 96)),
        dropout=float(args.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_latent_rollout_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    raw_leads = ckpt.get("rollout_leads", args.get("leads", ROLLOUT_LEADS))
    if isinstance(raw_leads, str):
        leads = tuple(int(x.strip()) for x in raw_leads.split(",") if x.strip())
    else:
        leads = tuple(int(x) for x in raw_leads)
    model = LatentRolloutFeatureModel(
        local_channels=int(ckpt.get("local_channels", 21)),
        global_channels=int(ckpt.get("global_channels", 9)),
        map_channels=int(ckpt.get("map_channels", 32)),
        hidden=int(args.get("hidden", 96)),
        dropout=float(args.get("dropout", 0.10)),
        leads=leads,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
