#!/usr/bin/env python3
"""
================================================================================
Met-JEPA: Joint Embedding Predictive Architecture for Subseasonal Forecasting
================================================================================

Learns 32-channel spatial latent maps that encode predictable components of
the regional atmospheric state at 15-day lead time.

Architecture:
  - Context encoder:  CNN on initialization-time local + global fields -> (32, H, W)
  - Target encoder:   EMA copy, sees future t+15 anomaly -> (32, H, W)
  - Predictor:        lightweight CNN: context embedding -> target embedding space
  - Loss:             L2 prediction + VICReg anti-collapse (variance + covariance)

Usage:
    python3 -u Subdomain/pretrain_met_jepa.py

After training, run export_jepa_maps.py then probe_jepa_maps.py.
================================================================================
"""

import os
import gc
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from copy import deepcopy
from datetime import datetime, timedelta
from netCDF4 import Dataset as NetCDFDataset
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# CONFIG
# ============================================================================

class JEPAConfig:
    # --- Paths ---
    WORK_DIR = "/blue/nessie/mostafarezaali/Teleconnection"
    TRAINING_DATA_PATH = os.path.join(WORK_DIR, "VDM_Training_Data_Extended_v2.nc")
    GLOBAL_DATA_PATH = os.path.join(WORK_DIR, "Global_Coarse_Conditions.nc")
    TOPO_PATH = os.path.join(
        WORK_DIR, "CONUS_topography_ETOPO2022_60s_on_model_grid.nc"
    )
    OUTPUT_DIR = os.path.join(WORK_DIR, "jepa_pretrain")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "jepa_pretrain", "checkpoints")
    PLOTS_DIR = os.path.join(WORK_DIR, "jepa_pretrain", "plots")

    # --- Subdomain ---
    SUBDOMAIN_ENABLED = True
    SUBDOMAIN_LAT_RANGE = (30.0, 40.0)
    SUBDOMAIN_LON_RANGE = (-105.0, -90.0)
    FULL_IMAGE_SIZE = (621, 1405)
    FULL_LAT_RANGE = (25.0, 50.0)
    FULL_LON_RANGE = (-130.0, -60.0)
    IMAGE_SIZE = None       # set by apply_subdomain_config()
    LAT_SLICE = slice(None)
    LON_SLICE = slice(None)
    CONUS_LAT_RANGE = None  # set by apply_subdomain_config()
    CONUS_LON_RANGE = None

    # --- Global teleconnection ---
    GLOBAL_VARIABLES = [
        "sst", "olr", "geopotential_200", "u_wind_200",
        "total_column_water_vapour", "v_wind_200",
        "geopotential_500", "temperature_850", "temperature_2m_global",
    ]
    NUM_GLOBAL_CHANNELS = 9
    GLOBAL_SIZE = (181, 360)

    # --- Year split ---
    TRAIN_YEARS = list(range(1981, 2016))
    VAL_YEARS = list(range(2016, 2020))
    TEST_YEARS = list(range(2020, 2024))
    LEAD_TIME = 15

    # --- JEPA architecture ---
    JEPA_MAP_CHANNELS = 32
    ENCODER_BASE_DIM = 64
    ENCODER_NUM_BLOCKS = 4
    PREDICTOR_DIM = 64

    # Context input: 3 anomaly + 9 physics + 9 static/seasonal + 9 global = 30
    CONTEXT_CHANNELS = 30
    # Target input: just the future anomaly (start simple)
    TARGET_CHANNELS = 1

    # --- Training ---
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.05
    MAX_EPOCHS = 300
    WARMUP_EPOCHS = 20
    CHECKPOINT_FREQ = 25
    NUM_VAL_SAMPLES = 100
    EMA_DECAY = 0.996
    GRAD_CLIP_NORM = 1.0

    # --- VICReg weights ---
    LAMBDA_PRED = 1.0
    LAMBDA_VAR = 1.0
    LAMBDA_COV = 0.04
    VAR_EPSILON = 0.01

    # --- Masking / augmentation ---
    SPATIAL_MASK_RATIO = 0.3
    SPATIAL_MASK_BLOCK_SIZE = 16
    CHANNEL_MASK_PROB = 0.15
    HISTORY_MASK_PROB = 0.3
    NOISE_STD = 0.05


CFG = JEPAConfig()

BASE_DATE = datetime(1981, 5, 1)


def time_to_date(tv):
    return BASE_DATE + timedelta(days=float(tv))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def compute_doy_array(time_values):
    doys = np.empty(len(time_values), dtype=np.float32)
    for i, tv in enumerate(time_values):
        dt = time_to_date(tv)
        doys[i] = dt.timetuple().tm_yday
    return doys


def compute_toa_insolation(lat_deg, doy):
    S0 = 1361.0
    lat_rad = np.radians(lat_deg)
    decl = np.radians(23.4393 * np.sin(np.radians(360.0 / 365.0 * (doy + 284))))
    cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    omega_s = np.arccos(cos_omega)
    Q = (S0 / np.pi) * (
        omega_s * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(omega_s)
    )
    return Q.astype(np.float32)


def apply_subdomain_config(cfg):
    if not cfg.SUBDOMAIN_ENABLED:
        cfg.IMAGE_SIZE = cfg.FULL_IMAGE_SIZE
        cfg.CONUS_LAT_RANGE = cfg.FULL_LAT_RANGE
        cfg.CONUS_LON_RANGE = cfg.FULL_LON_RANGE
        cfg.LAT_SLICE = slice(None)
        cfg.LON_SLICE = slice(None)
        return

    H_full, W_full = cfg.FULL_IMAGE_SIZE
    lat_full = np.linspace(cfg.FULL_LAT_RANGE[0], cfg.FULL_LAT_RANGE[1], H_full)
    lon_full = np.linspace(cfg.FULL_LON_RANGE[0], cfg.FULL_LON_RANGE[1], W_full)

    lat_idx = np.where(
        (lat_full >= cfg.SUBDOMAIN_LAT_RANGE[0])
        & (lat_full <= cfg.SUBDOMAIN_LAT_RANGE[1])
    )[0]
    lon_idx = np.where(
        (lon_full >= cfg.SUBDOMAIN_LON_RANGE[0])
        & (lon_full <= cfg.SUBDOMAIN_LON_RANGE[1])
    )[0]

    cfg.LAT_SLICE = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    cfg.LON_SLICE = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    H_sub = cfg.LAT_SLICE.stop - cfg.LAT_SLICE.start
    W_sub = cfg.LON_SLICE.stop - cfg.LON_SLICE.start
    cfg.IMAGE_SIZE = (H_sub, W_sub)
    cfg.CONUS_LAT_RANGE = (
        float(lat_full[cfg.LAT_SLICE.start]),
        float(lat_full[cfg.LAT_SLICE.stop - 1]),
    )
    cfg.CONUS_LON_RANGE = (
        float(lon_full[cfg.LON_SLICE.start]),
        float(lon_full[cfg.LON_SLICE.stop - 1]),
    )
    print(f"Subdomain: {H_sub}x{W_sub}, lat {cfg.CONUS_LAT_RANGE}, lon {cfg.CONUS_LON_RANGE}")


# ============================================================================
# DATA LOADING (reuses patterns from main training script)
# ============================================================================

def get_subdomain_tag(cfg):
    if not cfg.SUBDOMAIN_ENABLED:
        return ""
    la0, la1 = cfg.SUBDOMAIN_LAT_RANGE
    lo0, lo1 = cfg.SUBDOMAIN_LON_RANGE
    return f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"


def load_all_data(cfg):
    """Load and cache all fields for the subdomain. Returns a dict of arrays."""
    sub_tag = get_subdomain_tag(cfg)
    cache_dir = os.path.join(cfg.WORK_DIR, "data_cache" + sub_tag)
    os.makedirs(cache_dir, exist_ok=True)

    lat_s = cfg.LAT_SLICE if cfg.SUBDOMAIN_ENABLED else slice(None)
    lon_s = cfg.LON_SLICE if cfg.SUBDOMAIN_ENABLED else slice(None)

    paths = {
        "heat_index": os.path.join(cache_dir, "heat_index.npy"),
        "geopotential": os.path.join(cache_dir, "geopotential.npy"),
        "soil_moisture": os.path.join(cache_dir, "soil_moisture.npy"),
        "slp": os.path.join(cache_dir, "slp.npy"),
        "cond_train": os.path.join(cache_dir, "cond_train.npy"),
        "topography": os.path.join(cache_dir, "topography.npy"),
        "temperature_2m": os.path.join(cache_dir, "temperature_2m.npy"),
        "specific_humidity_850": os.path.join(cache_dir, "specific_humidity_850.npy"),
        "temperature_850": os.path.join(cache_dir, "temperature_850.npy"),
        "u_wind_850": os.path.join(cache_dir, "u_wind_850.npy"),
        "v_wind_850": os.path.join(cache_dir, "v_wind_850.npy"),
        "geopotential_300": os.path.join(cache_dir, "geopotential_300.npy"),
        "time_values": os.path.join(cache_dir, "time_values.npy"),
    }

    cache_ok = all(os.path.exists(p) for p in paths.values())
    if not cache_ok:
        raise FileNotFoundError(
            f"Data cache not found at {cache_dir}. "
            f"Run the main training script once to build the cache, "
            f"or run: python3 -u cfm_mesh_train_small_domain.py --mode train --deterministic"
        )

    # Load via mmap for efficiency
    data = {}
    for key, path in paths.items():
        data[key] = np.load(path, mmap_mode="r")
    print(f"Loaded local data cache from {cache_dir}")
    print(f"  heat_index shape: {data['heat_index'].shape}")

    # Global fields
    global_cache_dir = os.path.join(cache_dir, "global")
    global_data = OrderedDict()
    for var_name in cfg.GLOBAL_VARIABLES:
        gpath = os.path.join(global_cache_dir, f"{var_name}.npy")
        if not os.path.exists(gpath):
            raise FileNotFoundError(
                f"Global cache not found: {gpath}. Run main training script first."
            )
        global_data[var_name] = np.load(gpath, mmap_mode="r")
    data["global_data"] = global_data
    print(f"  Loaded {len(global_data)} global variables")

    # Climatology
    clim_path = os.path.join(
        cfg.WORK_DIR, "data_cache",
        f"climatology_daily_{cfg.TRAIN_YEARS[0]}_{cfg.TRAIN_YEARS[-1]}{sub_tag}.npz",
    )
    if not os.path.exists(clim_path):
        raise FileNotFoundError(
            f"Climatology not found: {clim_path}. Run compute_climatology.py first."
        )
    clim = np.load(clim_path)
    data["clim_mean"] = clim["clim_mean"]
    data["clim_std"] = clim["clim_std"]
    data["valid_doys"] = clim["valid_doys"].astype(bool)
    print(f"  Loaded climatology: {int(data['valid_doys'].sum())} valid DOYs")

    return data


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Dropout2d(dropout),
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class JEPAEncoder(nn.Module):
    """CNN encoder: (C_in, H, W) -> (out_channels, H, W) at full resolution."""

    def __init__(self, in_channels, out_channels=32, base_dim=64,
                 num_blocks=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.GroupNorm(min(8, base_dim), base_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(base_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, base_dim), base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h)


class JEPAPredictor(nn.Module):
    """Lightweight predictor: context embedding -> target embedding space."""

    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, embed_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class EMA:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.target = deepcopy(model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_ema, p_model in zip(self.target.parameters(), model.parameters()):
            p_ema.mul_(self.decay).add_(p_model.data, alpha=1.0 - self.decay)

    def get_target(self):
        return self.target


# ============================================================================
# MASKING / AUGMENTATION
# ============================================================================

def generate_spatial_block_mask(H, W, mask_ratio, block_size, device):
    """
    Generate a binary mask (1 = keep, 0 = masked) with rectangular blocks.
    Returns: (1, 1, H, W) tensor.
    """
    mask = torch.ones(1, 1, H, W, device=device)
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    total_blocks = num_blocks_h * num_blocks_w
    n_mask = max(1, int(total_blocks * mask_ratio))

    indices = torch.randperm(total_blocks, device=device)[:n_mask]
    for idx in indices:
        bh = (idx // num_blocks_w) * block_size
        bw = (idx % num_blocks_w) * block_size
        h_end = min(bh + block_size, H)
        w_end = min(bw + block_size, W)
        mask[:, :, bh:h_end, bw:w_end] = 0.0
    return mask


def apply_context_masking(context, land_mask, cfg, device):
    """
    Apply aggressive masking to context inputs to prevent trivial reconstruction.

    context: (B, C, H, W)
    land_mask: (1, 1, H, W) or broadcastable

    Masking strategy:
      1. Spatial block masking on all channels
      2. Channel group masking (randomly zero entire channel groups)
      3. History masking (randomly zero x_tm1 or x_tm2)
      4. Gaussian noise on surviving values
    """
    B, C, H, W = context.shape
    masked = context.clone()

    # 1. Spatial block masking
    for b in range(B):
        spatial_mask = generate_spatial_block_mask(
            H, W, cfg.SPATIAL_MASK_RATIO, cfg.SPATIAL_MASK_BLOCK_SIZE, device
        )
        masked[b] = masked[b] * spatial_mask.squeeze(0)

    # 2. Channel group masking
    # Groups: anomaly[0:3], physics[3:12], static[12:21], global[21:30]
    groups = [(0, 3), (3, 12), (12, 21), (21, min(30, C))]
    for b in range(B):
        for g_start, g_end in groups:
            if g_end > C:
                continue
            if torch.rand(1).item() < cfg.CHANNEL_MASK_PROB:
                masked[b, g_start:g_end] = 0.0

    # 3. History masking (channels 1 and 2 are x_tm1, x_tm2)
    for b in range(B):
        if torch.rand(1).item() < cfg.HISTORY_MASK_PROB:
            # Mask x_tm1
            masked[b, 1] = 0.0
        if torch.rand(1).item() < cfg.HISTORY_MASK_PROB:
            # Mask x_tm2
            masked[b, 2] = 0.0

    # 4. Gaussian noise
    if cfg.NOISE_STD > 0:
        noise = torch.randn_like(masked) * cfg.NOISE_STD
        masked = masked + noise * land_mask

    return masked


# ============================================================================
# VICREG LOSS
# ============================================================================

def vicreg_variance_loss(z, epsilon=0.01):
    """
    Variance regularization: push per-channel spatial variance above epsilon.
    z: (B, C, H, W)
    Returns scalar loss.
    """
    # Per-sample, per-channel spatial variance
    z_flat = z.flatten(2)  # (B, C, N)
    var = z_flat.var(dim=2)  # (B, C)
    # Hinge: penalize only when variance falls below epsilon
    return F.relu(epsilon - var).mean()


def vicreg_covariance_loss(z):
    """
    Covariance regularization: decorrelate channels.
    z: (B, C, H, W)
    Returns scalar loss.
    """
    B, C, H, W = z.shape
    z_flat = z.flatten(2)  # (B, C, N)
    z_centered = z_flat - z_flat.mean(dim=2, keepdim=True)

    # Average covariance matrix across batch
    # cov: (C, C)
    total_cov = torch.zeros(C, C, device=z.device, dtype=z.dtype)
    N = H * W
    for b in range(B):
        cov_b = (z_centered[b] @ z_centered[b].T) / (N - 1 + 1e-8)
        total_cov = total_cov + cov_b
    total_cov = total_cov / B

    # Penalize off-diagonal elements
    off_diag = total_cov - torch.diag(torch.diag(total_cov))
    return (off_diag ** 2).sum() / C


def jepa_loss(pred_embedding, target_embedding, land_mask, cfg):
    """
    Full JEPA loss: prediction L2 + VICReg anti-collapse.

    pred_embedding:   (B, C, H, W) from predictor(context_encoder(x_context))
    target_embedding: (B, C, H, W) from target_encoder(x_target), detached
    land_mask:        (B, 1, H, W) binary
    """
    # Prediction loss: L2 over land pixels
    mask = land_mask.expand_as(pred_embedding)
    diff = (pred_embedding - target_embedding) ** 2 * mask
    n_valid = mask.sum().clamp(min=1.0)
    pred_loss = diff.sum() / n_valid

    # VICReg on predicted embeddings (encourage diverse, non-collapsed maps)
    var_loss = vicreg_variance_loss(pred_embedding * mask, epsilon=cfg.VAR_EPSILON)
    cov_loss = vicreg_covariance_loss(pred_embedding * mask)

    total = (
        cfg.LAMBDA_PRED * pred_loss
        + cfg.LAMBDA_VAR * var_loss
        + cfg.LAMBDA_COV * cov_loss
    )

    return total, {
        "pred_loss": pred_loss.detach(),
        "var_loss": var_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "total_loss": total.detach(),
    }


# ============================================================================
# ANTI-COLLAPSE DIAGNOSTICS
# ============================================================================

@torch.no_grad()
def check_collapse(embedding, land_mask):
    """
    Check per-channel spatial variance of JEPA maps.
    Returns dict with diagnostics.
    """
    mask = land_mask.expand_as(embedding)
    B, C, H, W = embedding.shape

    channel_vars = []
    for c in range(C):
        vals = embedding[:, c:c+1][mask[:, c:c+1] > 0.5]
        if vals.numel() > 10:
            channel_vars.append(vals.var().item())
        else:
            channel_vars.append(0.0)

    channel_vars = np.array(channel_vars)
    collapsed = (channel_vars < 0.01).sum()

    return {
        "min_channel_var": float(channel_vars.min()),
        "mean_channel_var": float(channel_vars.mean()),
        "max_channel_var": float(channel_vars.max()),
        "collapsed_channels": int(collapsed),
        "active_channels": int(C - collapsed),
        "channel_vars": channel_vars,
    }


# ============================================================================
# DATASET
# ============================================================================

class JEPADataset(Dataset):
    """
    Dataset for JEPA pretraining.

    Returns (context_input, target_input, land_mask) where:
      context_input: (30, H, W) initialization-time fields
      target_input:  (1, H, W)  future anomaly at t+15
      land_mask:     (1, H, W)  binary
    """

    def __init__(self, cfg, data, indices, norm_stats=None):
        self.cfg = cfg
        self.data = data
        self.indices = indices
        self.h, self.w = cfg.IMAGE_SIZE

        # Precompute DOY arrays
        time_values = np.array(data["time_values"])
        self.doy_values = compute_doy_array(time_values)
        self.doy_indices = np.clip(self.doy_values.astype(np.int32) - 1, 0, 365)
        self.doy_sin = np.sin(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)
        self.doy_cos = np.cos(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)

        # Lat/lon grids
        self.lat_1d_deg = np.linspace(
            cfg.CONUS_LAT_RANGE[0], cfg.CONUS_LAT_RANGE[1], self.h
        )
        lat_1d = np.linspace(-1.0, 1.0, self.h)
        lon_1d = np.linspace(-1.0, 1.0, self.w)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        self.lat_grid = torch.from_numpy(lat_grid).float().unsqueeze(0)
        self.lon_grid = torch.from_numpy(lon_grid).float().unsqueeze(0)

        # Climatology
        self.clim_mean = data["clim_mean"]
        self.clim_std = data["clim_std"]
        self.valid_doys = data["valid_doys"]

        # Compute or load normalization stats
        if norm_stats is None:
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

    def _compute_norm_stats(self):
        print("  Computing JEPA normalization statistics from training data...")
        ti = np.array(self.indices)
        hi = self.data["heat_index"]

        hi_train = np.array(hi[:, :, ti], dtype=np.float32)
        hi_mask = np.isfinite(hi_train) & (hi_train != 0.0)
        valid_hi = hi_train[hi_mask]

        stats = {}
        stats["hi_mean"] = float(np.mean(valid_hi))
        stats["hi_std"] = float(np.std(valid_hi))

        # Physics channels
        physics_keys = [
            "geopotential", "soil_moisture", "slp", "temperature_2m",
            "specific_humidity_850", "temperature_850",
            "u_wind_850", "v_wind_850", "geopotential_300",
        ]
        p_means, p_stds = [], []
        for key in physics_keys:
            arr = self.data[key]
            if key == "geopotential":
                subset = np.array(arr[:, :, 0, ti], dtype=np.float32)
            else:
                subset = np.array(arr[:, :, ti], dtype=np.float32)
            p_means.append(float(np.nanmean(subset)))
            p_stds.append(float(np.nanstd(subset)))
            del subset
        stats["physics_mean"] = torch.tensor(p_means, dtype=torch.float32).view(9, 1, 1)
        stats["physics_std"] = torch.tensor(p_stds, dtype=torch.float32).view(9, 1, 1)

        # Topography
        topo = np.array(self.data["topography"], dtype=np.float32)
        topo_land = topo[topo != 0.0]
        stats["topo_mean"] = float(np.mean(topo_land)) if len(topo_land) > 0 else 0.0
        stats["topo_std"] = float(np.std(topo_land)) if len(topo_land) > 0 else 1.0

        # TOA
        sample_doys = self.doy_values[ti[:500]]
        toa_samples = np.stack([
            compute_toa_insolation(self.lat_1d_deg, d) for d in sample_doys
        ])
        stats["toa_mean"] = float(np.mean(toa_samples))
        stats["toa_std"] = float(np.std(toa_samples))

        # Climatology channel stats
        train_doy_idx = self.doy_indices[ti]
        sample_raw = np.array(hi[:, :, ti[0]], dtype=np.float32)
        land_2d = np.isfinite(sample_raw) & (sample_raw != 0.0)
        clim_at_train = self.clim_mean[train_doy_idx]
        valid_clim = clim_at_train[:, land_2d]
        stats["clim_ch_mean"] = float(np.nanmean(valid_clim))
        stats["clim_ch_std"] = float(np.nanstd(valid_clim))

        cstd_at_train = self.clim_std[train_doy_idx]
        valid_cstd = cstd_at_train[:, land_2d]
        stats["clim_std_ch_mean"] = float(np.nanmean(valid_cstd))
        stats["clim_std_ch_std"] = float(np.nanstd(valid_cstd))

        # Global field stats
        g_means, g_stds = [], []
        for var_name, var_data in self.data["global_data"].items():
            g_train = np.array(var_data[:, :, ti], dtype=np.float32)
            g_means.append(float(np.nanmean(g_train)))
            g_stds.append(float(np.nanstd(g_train)))
            del g_train
        stats["global_mean"] = torch.tensor(g_means, dtype=torch.float32).view(9, 1, 1)
        stats["global_std"] = torch.tensor(g_stds, dtype=torch.float32).view(9, 1, 1)

        # CondTrain stats
        cond = np.array(self.data["cond_train"][:, ti], dtype=np.float32)
        stats["cond_mean"] = torch.tensor(np.mean(cond, axis=1), dtype=torch.float32)
        stats["cond_std"] = torch.tensor(np.std(cond, axis=1), dtype=torch.float32)

        print(f"    hi: mean={stats['hi_mean']:.2f}, std={stats['hi_std']:.2f}")
        print(f"    toa: mean={stats['toa_mean']:.1f}, std={stats['toa_std']:.1f}")
        gc.collect()
        return stats

    def _assert_valid_doy(self, doy_idx, context=""):
        if doy_idx < 0 or doy_idx >= 366 or not bool(self.valid_doys[doy_idx]):
            raise ValueError(f"{context}: DOY index {doy_idx} outside MJJAS climatology.")

    def _to_anomaly(self, time_idx):
        doy_idx = int(self.doy_indices[time_idx])
        self._assert_valid_doy(doy_idx, f"t={time_idx}")
        raw = torch.from_numpy(
            np.array(self.data["heat_index"][:, :, time_idx], dtype=np.float32)
        )
        clim = torch.from_numpy(self.clim_mean[doy_idx].copy().astype(np.float32))
        cstd = torch.from_numpy(self.clim_std[doy_idx].copy().astype(np.float32))
        valid = torch.isfinite(raw) & (raw != 0.0) & torch.isfinite(clim) & torch.isfinite(cstd)
        anom = torch.zeros_like(raw)
        anom[valid] = (raw[valid] - clim[valid]) / (cstd[valid] + 1e-6)
        return anom.unsqueeze(0)  # (1, H, W)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        h, w = self.h, self.w
        ns = self.norm_stats

        doy_tgt = int(self.doy_indices[t + self.cfg.LEAD_TIME])
        self._assert_valid_doy(doy_tgt, "target DOY")

        # --- Land mask ---
        raw_t = torch.from_numpy(
            np.array(self.data["heat_index"][:, :, t], dtype=np.float32)
        )
        land_mask = (torch.isfinite(raw_t) & (raw_t != 0.0)).float().unsqueeze(0)

        # --- Anomaly history ---
        x_t = self._to_anomaly(t)
        x_tm1 = self._to_anomaly(t - 1)
        x_tm2 = self._to_anomaly(t - 2)

        # --- Local physics at t (9 channels) ---
        gp = torch.from_numpy(np.array(self.data["geopotential"][:, :, 0, t], dtype=np.float32))
        sm = torch.from_numpy(np.array(self.data["soil_moisture"][:, :, t], dtype=np.float32))
        slp = torch.from_numpy(np.array(self.data["slp"][:, :, t], dtype=np.float32))
        t2m = torch.from_numpy(np.array(self.data["temperature_2m"][:, :, t], dtype=np.float32))
        q850 = torch.from_numpy(np.array(self.data["specific_humidity_850"][:, :, t], dtype=np.float32))
        t850 = torch.from_numpy(np.array(self.data["temperature_850"][:, :, t], dtype=np.float32))
        u850 = torch.from_numpy(np.array(self.data["u_wind_850"][:, :, t], dtype=np.float32))
        v850 = torch.from_numpy(np.array(self.data["v_wind_850"][:, :, t], dtype=np.float32))
        z300 = torch.from_numpy(np.array(self.data["geopotential_300"][:, :, t], dtype=np.float32))

        physics = torch.stack([gp, sm, slp, t2m, q850, t850, u850, v850, z300], dim=0)
        physics = torch.nan_to_num(physics, nan=0.0)
        physics = (physics - ns["physics_mean"]) / (ns["physics_std"] + 1e-8)
        physics = physics * land_mask

        # --- Static / seasonal fields (9 channels) ---
        topo = torch.from_numpy(np.array(self.data["topography"], dtype=np.float32))
        topo = ((topo - ns["topo_mean"]) / (ns["topo_std"] + 1e-8)).unsqueeze(0) * land_mask

        lat_c = self.lat_grid.clone() * land_mask
        lon_c = self.lon_grid.clone() * land_mask

        doy_sin_ch = torch.full((1, h, w), self.doy_sin[t], dtype=torch.float32)
        doy_cos_ch = torch.full((1, h, w), self.doy_cos[t], dtype=torch.float32)

        toa_1d = compute_toa_insolation(self.lat_1d_deg, self.doy_values[t])
        toa_2d = torch.from_numpy(
            np.broadcast_to(toa_1d[:, None], (h, w)).copy()
        ).float().unsqueeze(0)
        toa_2d = (toa_2d - ns["toa_mean"]) / (ns["toa_std"] + 1e-8)

        land_mask_ch = land_mask

        clim_tgt = torch.from_numpy(self.clim_mean[doy_tgt].copy().astype(np.float32))
        clim_tgt_ch = ((clim_tgt - ns["clim_ch_mean"]) / (ns["clim_ch_std"] + 1e-6)).unsqueeze(0) * land_mask
        cstd_tgt = torch.from_numpy(self.clim_std[doy_tgt].copy().astype(np.float32))
        clim_std_ch = ((cstd_tgt - ns["clim_std_ch_mean"]) / (ns["clim_std_ch_std"] + 1e-6)).unsqueeze(0) * land_mask

        static_seasonal = torch.cat([
            topo, lat_c, lon_c, doy_sin_ch, doy_cos_ch, toa_2d,
            land_mask_ch, clim_tgt_ch, clim_std_ch,
        ], dim=0)  # (9, H, W)

        # --- Global fields interpolated to local grid (9 channels) ---
        global_channels = []
        for var_name, var_data in self.data["global_data"].items():
            g_slice = torch.from_numpy(
                np.array(var_data[:, :, t], dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 181, 360)
            g_interp = F.interpolate(g_slice, size=(h, w), mode="bilinear", align_corners=False)
            global_channels.append(g_interp.squeeze(0))  # (1, H, W)
        global_local = torch.cat(global_channels, dim=0)  # (9, H, W)
        global_local = (global_local - ns["global_mean"]) / (ns["global_std"] + 1e-8)

        # --- Assemble context: (30, H, W) ---
        context = torch.cat([
            x_t, x_tm1, x_tm2,          # 3
            physics,                     # 9
            static_seasonal,             # 9
            global_local,                # 9
        ], dim=0)

        # --- Target: future anomaly at t+15 (1, H, W) ---
        target = self._to_anomaly(t + self.cfg.LEAD_TIME)

        return context, target, land_mask


# ============================================================================
# TRAINING
# ============================================================================

def train_jepa(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    apply_subdomain_config(cfg)

    # Load data
    data = load_all_data(cfg)

    # Build indices
    time_values = np.array(data["time_values"])
    runs = detect_continuous_runs(time_values)
    valid_indices = build_valid_indices(runs, lead_time=cfg.LEAD_TIME, min_history=2)

    time_years = np.array([time_to_date(tv).year for tv in time_values])
    train_set = set(cfg.TRAIN_YEARS)
    val_set = set(cfg.VAL_YEARS)

    train_indices = [i for i in valid_indices if time_years[i] in train_set]
    val_indices = [i for i in valid_indices if time_years[i] in val_set]

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")

    # Datasets
    train_dataset = JEPADataset(cfg, data, train_indices)
    norm_stats = train_dataset.norm_stats
    val_dataset = JEPADataset(cfg, data, val_indices, norm_stats=norm_stats)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Save norm stats for export script
    stats_path = os.path.join(cfg.OUTPUT_DIR, "jepa_norm_stats.npz")
    save_stats = {}
    for k, v in norm_stats.items():
        if isinstance(v, torch.Tensor):
            save_stats[k] = v.numpy()
        else:
            save_stats[k] = np.array(v)
    np.savez(stats_path, **save_stats)
    print(f"Saved norm stats to {stats_path}")

    # Models
    context_encoder = JEPAEncoder(
        in_channels=cfg.CONTEXT_CHANNELS,
        out_channels=cfg.JEPA_MAP_CHANNELS,
        base_dim=cfg.ENCODER_BASE_DIM,
        num_blocks=cfg.ENCODER_NUM_BLOCKS,
    ).to(device)

    target_encoder = JEPAEncoder(
        in_channels=cfg.TARGET_CHANNELS,
        out_channels=cfg.JEPA_MAP_CHANNELS,
        base_dim=cfg.ENCODER_BASE_DIM,
        num_blocks=cfg.ENCODER_NUM_BLOCKS,
    ).to(device)

    predictor = JEPAPredictor(
        embed_dim=cfg.JEPA_MAP_CHANNELS,
        hidden_dim=cfg.PREDICTOR_DIM,
    ).to(device)

    # EMA for target encoder
    ema = EMA(target_encoder, decay=cfg.EMA_DECAY)
    # Initialize target encoder with its own random weights (not a copy of context)
    # The EMA will slowly move it

    # Parameter count
    n_ctx = sum(p.numel() for p in context_encoder.parameters())
    n_tgt = sum(p.numel() for p in target_encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"\nParameter counts:")
    print(f"  Context encoder: {n_ctx/1e6:.2f}M")
    print(f"  Target encoder:  {n_tgt/1e6:.2f}M")
    print(f"  Predictor:       {n_pred/1e6:.2f}M")
    print(f"  Total trainable: {(n_ctx + n_pred)/1e6:.2f}M")
    print(f"  (Target encoder is EMA, not directly trained)\n")

    # Optimizer (context encoder + predictor only)
    optimizer = AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.WARMUP_EPOCHS],
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_pred": [], "val_pred": [],
        "train_var": [], "val_var": [],
        "active_channels": [],
    }

    best_val_loss = float("inf")

    # First batch check
    first_batch_checked = False

    print(f"{'='*70}")
    print(f"Met-JEPA Pretraining")
    print(f"{'='*70}")
    print(f"  Epochs: {cfg.MAX_EPOCHS}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Map channels: {cfg.JEPA_MAP_CHANNELS}")
    print(f"  Spatial mask ratio: {cfg.SPATIAL_MASK_RATIO}")
    print(f"  Channel mask prob: {cfg.CHANNEL_MASK_PROB}")
    print(f"  History mask prob: {cfg.HISTORY_MASK_PROB}")
    print(f"  VICReg weights: pred={cfg.LAMBDA_PRED}, var={cfg.LAMBDA_VAR}, cov={cfg.LAMBDA_COV}")
    print(f"{'='*70}\n")

    for epoch in range(cfg.MAX_EPOCHS):
        # ---- TRAIN ----
        context_encoder.train()
        predictor.train()

        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        n_batches = 0

        for batch_idx, (context, target, land_mask) in enumerate(train_loader):
            context = context.to(device)
            target = target.to(device)
            land_mask = land_mask.to(device)

            # First-batch diagnostic
            if not first_batch_checked:
                print(f"  First batch shapes:")
                print(f"    context:   {context.shape}")
                print(f"    target:    {target.shape}")
                print(f"    land_mask: {land_mask.shape}")
                with torch.no_grad():
                    valid_ctx = context[:, 0][land_mask[:, 0] > 0.5]
                    valid_tgt = target[:, 0][land_mask[:, 0] > 0.5]
                    print(f"    context ch0 (x_t): mean={valid_ctx.mean():.4f}, std={valid_ctx.std():.4f}")
                    print(f"    target (t+15 anom): mean={valid_tgt.mean():.4f}, std={valid_tgt.std():.4f}")
                first_batch_checked = True

            # Apply masking to context (target is NOT masked)
            context_masked = apply_context_masking(context, land_mask, cfg, device)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Context path: encode masked context, predict target embedding
                ctx_embedding = context_encoder(context_masked)
                pred_embedding = predictor(ctx_embedding)

                # Target path: encode future state with EMA target encoder
                with torch.no_grad():
                    tgt_embedding = ema.get_target()(target).detach()

                # Loss
                loss, components = jepa_loss(
                    pred_embedding, tgt_embedding, land_mask, cfg
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(context_encoder.parameters()) + list(predictor.parameters()),
                cfg.GRAD_CLIP_NORM,
            )
            optimizer.step()

            # EMA update of target encoder
            ema.update(target_encoder)

            epoch_loss += components["total_loss"].item()
            epoch_pred += components["pred_loss"].item()
            epoch_var += components["var_loss"].item()
            epoch_cov += components["cov_loss"].item()
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            print(f"  Epoch {epoch+1}: ALL batches failed!")
            continue

        avg_loss = epoch_loss / n_batches
        avg_pred = epoch_pred / n_batches
        avg_var = epoch_var / n_batches
        avg_cov = epoch_cov / n_batches

        # ---- VALIDATE ----
        if (epoch + 1) % cfg.CHECKPOINT_FREQ == 0 or epoch == 0:
            context_encoder.eval()
            predictor.eval()

            val_loss_sum = 0.0
            val_pred_sum = 0.0
            val_var_sum = 0.0
            val_n = 0
            all_pred_emb = []

            with torch.no_grad():
                for context, target, land_mask in val_loader:
                    if val_n >= cfg.NUM_VAL_SAMPLES:
                        break
                    context = context.to(device)
                    target = target.to(device)
                    land_mask = land_mask.to(device)

                    # No masking on context during validation
                    ctx_emb = context_encoder(context)
                    pred_emb = predictor(ctx_emb)
                    tgt_emb = ema.get_target()(target)

                    loss_v, comp_v = jepa_loss(pred_emb, tgt_emb, land_mask, cfg)

                    val_loss_sum += comp_v["total_loss"].item() * context.shape[0]
                    val_pred_sum += comp_v["pred_loss"].item() * context.shape[0]
                    val_var_sum += comp_v["var_loss"].item() * context.shape[0]
                    val_n += context.shape[0]

                    if len(all_pred_emb) < 3:
                        all_pred_emb.append(pred_emb.cpu())

            val_loss = val_loss_sum / max(val_n, 1)
            val_pred = val_pred_sum / max(val_n, 1)
            val_var = val_var_sum / max(val_n, 1)

            # Collapse check
            if all_pred_emb:
                sample_emb = all_pred_emb[0][:4]
                sample_mask = torch.ones(sample_emb.shape[0], 1,
                                         sample_emb.shape[2], sample_emb.shape[3])
                collapse_info = check_collapse(sample_emb, sample_mask)
            else:
                collapse_info = {"active_channels": 0, "collapsed_channels": 32,
                                 "min_channel_var": 0.0, "mean_channel_var": 0.0}

            # Log
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:4d} | "
                f"Train: loss={avg_loss:.4f} pred={avg_pred:.4f} var={avg_var:.4f} cov={avg_cov:.4f} | "
                f"Val: loss={val_loss:.4f} pred={val_pred:.4f} | "
                f"Active ch: {collapse_info['active_channels']}/32 | "
                f"LR: {lr:.2e}"
            )

            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["train_pred"].append(avg_pred)
            history["val_pred"].append(val_pred)
            history["train_var"].append(avg_var)
            history["val_var"].append(val_var)
            history["active_channels"].append(collapse_info["active_channels"])

            # Collapse warning
            if collapse_info["collapsed_channels"] > 8:
                print(
                    f"  WARNING: {collapse_info['collapsed_channels']}/32 channels collapsed! "
                    f"min_var={collapse_info['min_channel_var']:.6f}"
                )

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "epoch": epoch + 1,
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder": ema.get_target().state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "collapse_info": collapse_info,
                    "norm_stats_path": stats_path,
                    "config": {
                        "JEPA_MAP_CHANNELS": cfg.JEPA_MAP_CHANNELS,
                        "CONTEXT_CHANNELS": cfg.CONTEXT_CHANNELS,
                        "TARGET_CHANNELS": cfg.TARGET_CHANNELS,
                        "ENCODER_BASE_DIM": cfg.ENCODER_BASE_DIM,
                        "ENCODER_NUM_BLOCKS": cfg.ENCODER_NUM_BLOCKS,
                        "PREDICTOR_DIM": cfg.PREDICTOR_DIM,
                        "IMAGE_SIZE": cfg.IMAGE_SIZE,
                    },
                }
                best_path = os.path.join(cfg.OUTPUT_DIR, "best_jepa.pth")
                torch.save(ckpt, best_path)
                print(f"  New best model saved (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder": ema.get_target().state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(cfg.CHECKPOINT_DIR, f"jepa_epoch_{epoch+1:04d}.pth"),
            )

            # Save training curves
            _save_training_plots(history, cfg)

            context_encoder.train()
            predictor.train()

        elif (epoch + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:4d} | "
                f"Train: loss={avg_loss:.4f} pred={avg_pred:.4f} var={avg_var:.4f} | "
                f"LR: {lr:.2e}"
            )

    # Final summary
    print(f"\n{'='*70}")
    print(f"JEPA Pretraining Complete")
    print(f"{'='*70}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {os.path.join(cfg.OUTPUT_DIR, 'best_jepa.pth')}")
    print(f"  Next step: python3 -u export_jepa_maps.py")
    print(f"{'='*70}")

    return context_encoder, predictor, ema.get_target()


def _save_training_plots(history, cfg):
    if len(history["train_loss"]) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total JEPA Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_pred"], "b-", label="Train")
    axes[0, 1].plot(epochs, history["val_pred"], "r-", label="Val")
    axes[0, 1].set_ylabel("Prediction L2")
    axes[0, 1].set_title("Prediction Loss (L2 in latent space)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_var"], "g-", label="Train var loss")
    axes[1, 0].plot(epochs, history["val_var"], "m-", label="Val var loss")
    axes[1, 0].set_ylabel("Variance Loss")
    axes[1, 0].set_title("VICReg Variance Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["active_channels"], "k-o", markersize=3)
    axes[1, 1].axhline(y=32, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_ylabel("Active Channels")
    axes[1, 1].set_title("Non-Collapsed Channels (32 = healthy)")
    axes[1, 1].set_ylim(0, 34)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.set_xlabel("Checkpoint")

    plt.suptitle("Met-JEPA Pretraining", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, "jepa_training.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"Met-JEPA Pretraining")
    print(f"{'='*70}\n")

    train_jepa(CFG)


if __name__ == "__main__":
    main()
