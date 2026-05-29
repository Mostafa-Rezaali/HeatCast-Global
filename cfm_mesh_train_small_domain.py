#!/usr/bin/env python3
"""
================================================================================
Conditional Flow Matching (CFM) - Optimal Transport Path
================================================================================
Physics-Guided Heat Wave Forecasting with DETERMINISTIC ODE SOLVER
+ ICOSAHEDRAL MESH GNN BACKBONE (GraphCast-style Encoder-Processor-Decoder)
+ GLOBAL CONTEXT ENCODER (Per-node bilinear sampling)

Key features:
- Uses Optimal Transport (OT) displacement path: z_t = (1-t)*z_0 + t*z_1
- Model predicts the velocity field v_t pointing towards the target heat index.
- LEAD_TIME=1 with ROLLOUT_STEPS=15 autoregressive rollout at inference.
- Icosahedral mesh GNN replaces U-Net backbone.
- Per-round FiLM conditioning in MeshProcessor and Mesh2GridDecoder.
- DDP padding removed (GNN operates on 1D node arrays).
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import gc
import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from tqdm import tqdm
import os
import math
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import shutil
mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

from datetime import datetime, timedelta

from icosahedral_mesh import IcosahedralMesh
from mesh_backbone import MeshFlowNet, count_parameters
from mode_dispatch import compute_loss, generate_sample, generate_autoregressive_rollout
import pickle

# ======================================================================================
# Logger
# ======================================================================================
import logging

def setup_spike_logger(output_dir):
    logger = logging.getLogger("spike_warnings")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, "spike_warnings.log"), mode="a")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ======================================================================================
# COMPUTED VARIABLE HELPERS
# ======================================================================================
def compute_doy_array(time_values):
    """Convert 'days since 1981-05-01' offsets to day-of-year (1-366)."""
    base = datetime(1981, 5, 1)
    doys = np.empty(len(time_values), dtype=np.float32)
    for i, tv in enumerate(time_values):
        dt = base + timedelta(days=float(tv))
        doys[i] = dt.timetuple().tm_yday
    return doys


def compute_toa_insolation(lat_deg, doy):
    """
    Daily-mean TOA insolation (W/m^2) for given latitudes and day-of-year.
    lat_deg: 1D array of latitudes in degrees (H,)
    doy: scalar day-of-year
    Returns: (H,) array
    """
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


# ======================================================================================
# DDP Setup
# ======================================================================================
def setup_ddp(rank, world_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}: entering init_process_group...", flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
        timeout=timedelta(hours=5),
    )
    print(f"Rank {rank}: DDP initialized", flush=True)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# ======================================================================================
# CFM COMPONENTS
# ======================================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

class FlowMatching(nn.Module):
    def sample_xt(self, x_0, x_1, t):
        t = t.view(-1, 1, 1, 1)
        x_flow = (1 - t) * x_0 + t * x_1
        return x_flow

    def velocity_target(self, x_0, x_1):
        return x_1 - x_0

# ======================================================================================
# CONFIGURATION
# ======================================================================================
class Config:
    # ==================== DATA PATHS ====================
    TRAINING_DATA_PATH = '/blue/nessie/mostafarezaali/Teleconnection/VDM_Training_Data_Extended_v2.nc'
    OUTPUT_DIR = "/blue/nessie/mostafarezaali/Teleconnection/"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "test_prediction_plots")

    OUTPUT_NC_FILE = os.path.join(OUTPUT_DIR, "CFM_Forecasts_Improved.nc")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_improved.pth")

    # ==================== GLOBAL DATA PATHS ====================
    GLOBAL_DATA_PATH = '/blue/nessie/mostafarezaali/Teleconnection/Global_Coarse_Conditions.nc'
    GLOBAL_VARIABLES = [
        'sst',
        'olr',
        'geopotential_200',
        'u_wind_200',
        'total_column_water_vapour',
        'v_wind_200',
        'geopotential_500',
        'temperature_850',
        'temperature_2m_global',
    ]
    
    GLOBAL_SIZE = (181, 360)
    NUM_GLOBAL_CHANNELS = 9

    # ==================== SUBDOMAIN (fast architecture iteration) ====================
    # When True, restrict CONUS to a small region. Global fields stay full.
    # IMAGE_SIZE, CONUS_LAT_RANGE, CONUS_LON_RANGE override at runtime.
    SUBDOMAIN_ENABLED = True
    SUBDOMAIN_LAT_RANGE = (30.0, 40.0)    # South-central US
    SUBDOMAIN_LON_RANGE = (-105.0, -90.0)
    LAT_SLICE = slice(None)               # filled by apply_subdomain_config
    LON_SLICE = slice(None)

    # ==================== MODEL ARCHITECTURE ====================
    IMAGE_SIZE = (621, 1405)              # Overridden at runtime if SUBDOMAIN_ENABLED
    IMAGE_CHANNELS = 1
    # spatial_c = physics(9) + topo(1) + lat(1) + lon(1) + doy_sin(1) + doy_cos(1) + toa(1) + land_mask(1) = 16
    # model input = [x_flow(1), x_t(1), x_tm1(1), x_tm2(1), spatial_c(16)] = 20 (CFM)
    # or            [x_t(1), x_tm1(1), x_tm2(1), spatial_c(16)] = 19 (deterministic)
    NUM_SPATIAL_CONDITIONS = 19
    LEAD_TIME = 1              # Phase 2: 1-day steps
    ROLLOUT_STEPS = 15         # Phase 2: 15-step autoregressive rollout at inference
    CONDITION_DIM = 5

    BASE_DIM = 64
    DIM_MULTS = (1, 2, 4, 8)
    DROPOUT_RATE = 0.2

    GLOBAL_ENCODER_DIM = 64

    BATCH_SIZE = 32
    OCEAN_FILL = 0

    # ==================== CFM SCHEDULE ====================
    CFM_SAMPLING_STEPS = 50

    # ==================== TRAINING HYPERPARAMETERS ====================
    DEVICE = "cuda"
    LEARNING_RATE = 1e-5
    GRAD_CLIP_NORM = 0.5
    WINDOW_SIZE = 64   # kept for reference but no longer used for padding

    # ==================== GENERATION SETTINGS ====================
    ENSEMBLE_SIZE = 20
    ENSEMBLE_MODE = False

    # ==================== TRAINING SCHEDULE ====================
    MAX_EPOCHS = 500             # Subdomain exploration; increase for full CONUS
    TEST_FRACTION = 0.15
    VAL_FRACTION = 0.15
    CHECKPOINT_FREQ = 50         # Frequent validation for fast architecture feedback
    NUM_VALIDATION_SAMPLES = 20
    WARMUP_EPOCHS = 100

    # ==================== MESH CONFIG ====================
    MESH_REFINEMENT_LEVEL = 7
    MESH_PROCESSOR_ROUNDS = 4
    MESH_LATENT_DIM = 128
    MESH_BUFFER_DEG = 5.0
    K_GRID2MESH = 3
    K_MESH2GRID = 3
    CONUS_LAT_RANGE = (25.0, 50.0)
    CONUS_LON_RANGE = (-130.0, -60.0)

    # ==================== MODE ====================
    DETERMINISTIC = False


if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    mode_name = "DETERMINISTIC (GraphCast)" if Config.DETERMINISTIC else "PROBABILISTIC (GenCast/CFM)"
    print(f"\n{'='*80}")
    print(f"CFM TRAINING - {mode_name} + ICOSAHEDRAL MESH GNN")
    print(f"{'='*80}")
    print(f"Using device: {Config.DEVICE}")
    print(f"Mode: {mode_name}")
    print(f"Lead time: {Config.LEAD_TIME} day, Rollout: {Config.ROLLOUT_STEPS} steps")
    print(f"Sampling Steps: {Config.CFM_SAMPLING_STEPS} {'(ignored in deterministic)' if Config.DETERMINISTIC else ''}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Mesh refinement level: {Config.MESH_REFINEMENT_LEVEL}")
    print(f"Mesh processor rounds: {Config.MESH_PROCESSOR_ROUNDS}")
    print(f"Mesh latent dim: {Config.MESH_LATENT_DIM}")
    print(f"Global fields: {Config.NUM_GLOBAL_CHANNELS} channels at {Config.GLOBAL_SIZE}")

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA init failed, falling back to CPU: {e}")
    return torch.device("cpu")


# ======================================================================================
# SEASON BOUNDARY DETECTION
# ======================================================================================
def detect_continuous_runs(time_values):
    """
    Detect contiguous daily runs in the time array.
    Returns list of (start_idx, end_idx) for each continuous run.
    A gap is any jump > 1.5 days between consecutive entries.
    """
    runs = []
    start = 0
    for i in range(1, len(time_values)):
        if time_values[i] - time_values[i - 1] > 1.5:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(time_values) - 1))
    return runs


def build_valid_indices(runs, lead_time, min_history=2):
    """
    Build valid sample indices ensuring every (t-2, t-1, t, t+lead_time)
    window falls within a single continuous run.

    For training: lead_time = LEAD_TIME (1)
    For val/test rollout: lead_time = ROLLOUT_STEPS (15)
    """
    indices = []
    for start, end in runs:
        # Need t-2 >= start and t+lead_time <= end
        first_valid = start + min_history
        last_valid = end - lead_time
        for t in range(first_valid, last_valid + 1):
            indices.append(t)
    return indices


# ======================================================================================
# BUILD MESH
# ======================================================================================
def build_mesh_once(config, conus_mask, device, ddp=False):
    sub_tag = ""
    if config.SUBDOMAIN_ENABLED:
        la0, la1 = config.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = config.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"
    cache_path = os.path.join(config.OUTPUT_DIR, "data_cache",
                               f"mesh_level{config.MESH_REFINEMENT_LEVEL}{sub_tag}.pkl")

    if not ddp or dist.get_rank() == 0:
        if os.path.exists(cache_path):
            print(f"Loading cached mesh from {cache_path}")
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            H, W = config.IMAGE_SIZE
            grid_lat = np.linspace(config.CONUS_LAT_RANGE[0], config.CONUS_LAT_RANGE[1], H)
            grid_lon = np.linspace(config.CONUS_LON_RANGE[0], config.CONUS_LON_RANGE[1], W)
            mask_raw = conus_mask.squeeze().cpu().numpy() > 0.5

            mesh = IcosahedralMesh(
                refinement_level=config.MESH_REFINEMENT_LEVEL,
                lat_range=config.CONUS_LAT_RANGE,
                lon_range=config.CONUS_LON_RANGE,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                land_mask=mask_raw,
                buffer_deg=config.MESH_BUFFER_DEG,
                k_grid2mesh=config.K_GRID2MESH,
                k_mesh2grid=config.K_MESH2GRID,
            )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(mesh, f)
            print(f"Cached mesh to {cache_path}")

    if ddp:
        dist.barrier()
        if dist.get_rank() != 0:
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)

    mesh.to_torch(device)

    if is_main_process():
        s = mesh.summary()
        print(f"\nMesh summary:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    return mesh


# ======================================================================================
# IMPROVED METRICS
# ======================================================================================
@torch.inference_mode()
def compute_persistence_baseline(val_dataset, mask, n_samples=200):
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_2d = mask[:h, :w].cpu()
    else:
        mask_2d = torch.from_numpy(np.array(mask[:h, :w]))

    all_pred = []
    all_truth = []
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        (y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, _, batch_mask) = batch
        x_tp1_true = y[0, 0, :h, :w]
        x_tp1_persist = x_t[0, 0, :h, :w]

        all_pred.append(x_tp1_persist)
        all_truth.append(x_tp1_true)
        count += 1

    preds = torch.stack(all_pred).unsqueeze(1)
    truth = torch.stack(all_truth).unsqueeze(1)

    metrics = calculate_improved_metrics(preds, truth, mask=mask_2d)
    return metrics


def calculate_improved_metrics(pred, truth, mask=None):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred

    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif mask is not None:
        mask_np = mask
    else:
        mask_np = None

    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]
        if mask_np is not None and mask_np.ndim == 4:
            mask_np = mask_np[:, 0]

    if mask_np is not None:
        if mask_np.ndim == 2:
            mask_np = np.broadcast_to(mask_np[None, :, :], pred_np.shape)

    if mask_np is not None:
        pred_masked = np.where(mask_np > 0.5, pred_np, np.nan)
        truth_masked = np.where(mask_np > 0.5, truth_np, np.nan)
    else:
        pred_masked = pred_np
        truth_masked = truth_np

    if mask_np is not None:
        pred_spatial_std = np.nanstd(pred_masked, axis=(1, 2)).mean()
        truth_spatial_std = np.nanstd(truth_masked, axis=(1, 2)).mean()
    else:
        pred_spatial_std = np.std(pred_np, axis=(1, 2)).mean()
        truth_spatial_std = np.std(truth_np, axis=(1, 2)).mean()

    variance_ratio = pred_spatial_std / (truth_spatial_std + 1e-8)

    if mask_np is not None:
        pred_for_grad = np.where(mask_np > 0.5, pred_np, 0.0)
        truth_for_grad = np.where(mask_np > 0.5, truth_np, 0.0)
    else:
        pred_for_grad = pred_np
        truth_for_grad = truth_np

    pred_grad_y = np.abs(np.diff(pred_for_grad, axis=1))
    pred_grad_x = np.abs(np.diff(pred_for_grad, axis=2))
    truth_grad_y = np.abs(np.diff(truth_for_grad, axis=1))
    truth_grad_x = np.abs(np.diff(truth_for_grad, axis=2))

    if mask_np is not None:
        mask_grad_y = mask_np[:, :-1, :] * mask_np[:, 1:, :]
        mask_grad_x = mask_np[:, :, :-1] * mask_np[:, :, 1:]
        pred_grad_y = np.where(mask_grad_y > 0.5, pred_grad_y, np.nan)
        pred_grad_x = np.where(mask_grad_x > 0.5, pred_grad_x, np.nan)
        truth_grad_y = np.where(mask_grad_y > 0.5, truth_grad_y, np.nan)
        truth_grad_x = np.where(mask_grad_x > 0.5, truth_grad_x, np.nan)
        pred_grad_mag = np.nanmean(pred_grad_y) + np.nanmean(pred_grad_x)
        truth_grad_mag = np.nanmean(truth_grad_y) + np.nanmean(truth_grad_x)
    else:
        pred_grad_mag = np.mean(pred_grad_y) + np.mean(pred_grad_x)
        truth_grad_mag = np.mean(truth_grad_y) + np.mean(truth_grad_x)

    gradient_ratio = pred_grad_mag / (truth_grad_mag + 1e-8)

    if mask_np is not None:
        valid_pred = pred_masked[~np.isnan(pred_masked)]
        valid_truth = truth_masked[~np.isnan(truth_masked)]
        truth_p95 = np.percentile(valid_truth, 95)
        pred_p95 = np.percentile(valid_pred, 95)
    else:
        truth_p95 = np.percentile(truth_np, 95)
        pred_p95 = np.percentile(pred_np, 95)

    extreme_bias = pred_p95 - truth_p95

    if mask_np is not None:
        corrs = []
        for b in range(pred_masked.shape[0]):
            vp = pred_masked[b][~np.isnan(pred_masked[b])]
            vt = truth_masked[b][~np.isnan(truth_masked[b])]
            if vt.std() > 1e-6 and vp.std() > 1e-6:
                corrs.append(np.corrcoef(vt, vp)[0, 1])
        correlation = float(np.mean(corrs)) if len(corrs) else 0.0
    else:
        corrs = []
        for b in range(pred_np.shape[0]):
            vp = pred_np[b].ravel()
            vt = truth_np[b].ravel()
            if vt.std() > 1e-6 and vp.std() > 1e-6:
                corrs.append(np.corrcoef(vt, vp)[0, 1])
        correlation = float(np.mean(corrs)) if len(corrs) else 0.0

    if mask_np is not None:
        valid_pred = pred_masked[~np.isnan(pred_masked)]
        valid_truth = truth_masked[~np.isnan(truth_masked)]
        ss_res = np.sum((valid_truth - valid_pred) ** 2)
        ss_tot = np.sum((valid_truth - valid_truth.mean()) ** 2) + 1e-8
    else:
        ss_res = np.sum((truth_np - pred_np) ** 2)
        ss_tot = np.sum((truth_np - truth_np.mean()) ** 2) + 1e-8

    r2 = 1 - (ss_res / ss_tot)

    return {
        'variance_ratio': float(variance_ratio),
        'gradient_ratio': float(gradient_ratio),
        'extreme_bias': float(extreme_bias),
        'correlation': float(correlation),
        'r2': float(r2),
        'pred_spatial_std': float(pred_spatial_std),
        'truth_spatial_std': float(truth_spatial_std),
    }


# ======================================================================================
# DATASET
# ======================================================================================
class ClimateDataset(Dataset):
    def __init__(self, config, mode='train', train_indices=None, val_indices=None, test_indices=None,
                 normalization_stats=None, shared_data=None):
        if is_main_process():
            print(f"Initializing ClimateDataset ({mode} mode)...")
        self.config = config
        self.mode = mode

        if shared_data is None:
            print(f"  Loading data to RAM...")
            with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
                raw_shape = nc.variables['t2m_prism'].shape
                print(f"  Raw dataset shape: {raw_shape}")
                hi_raw = nc.variables['t2m_prism'][:]
                if hi_raw.ndim == 4:
                    self.heat_index = np.array(hi_raw[:, :, 0, :], dtype=np.float32)
                elif hi_raw.ndim == 3:
                    self.heat_index = np.array(hi_raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unexpected t2m_prism shape: {raw_shape}")
                self.n_timesteps = self.heat_index.shape[-1]
                self.geopotential = np.array(nc.variables['geopotential'][:], dtype=np.float32)
                self.soil_moisture = np.array(nc.variables['soil_moisture'][:], dtype=np.float32)
                self.slp = np.array(nc.variables['sea_level_pressure'][:], dtype=np.float32)
                self.cond_train = np.array(nc.variables['CondTrain'][:], dtype=np.float32)
                self.time_values = np.array(nc.variables['time'][:], dtype=np.float64)
            self.topography = None
            self.global_data = None
            print(f"  Data loaded to RAM")
        else:
            if is_main_process():
                print(f"  Using SHARED data arrays (zero-copy)")
            self.heat_index = shared_data['heat_index']
            self.geopotential = shared_data['geopotential']
            self.soil_moisture = shared_data['soil_moisture']
            self.slp = shared_data['slp']
            self.cond_train = shared_data['cond_train']
            self.topography = shared_data['topography']
            self.temperature_2m = shared_data['temperature_2m']
            self.specific_humidity_850 = shared_data['specific_humidity_850']
            self.temperature_850 = shared_data['temperature_850']
            self.u_wind_850 = shared_data['u_wind_850']
            self.v_wind_850 = shared_data['v_wind_850']
            self.geopotential_300 = shared_data['geopotential_300']
            self.global_data = shared_data.get('global_data', None)
            self.time_values = shared_data['time_values']
            self.n_timesteps = self.heat_index.shape[-1]

        # Precompute day-of-year arrays and latitude for TOA
        self.doy_values = compute_doy_array(self.time_values)
        self.doy_sin_arr = np.sin(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)
        self.doy_cos_arr = np.cos(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)
        self.lat_1d_deg = np.linspace(config.CONUS_LAT_RANGE[0], config.CONUS_LAT_RANGE[1],
                                       config.IMAGE_SIZE[0])

        if normalization_stats is None and mode == 'train':
            print("  Calculating Z-Score statistics (TRAINING SET ONLY)...")
            if train_indices is None:
                train_indices = list(range(1, self.n_timesteps - self.config.LEAD_TIME))

            train_t_indices = np.array(train_indices)
            hi_train = self.heat_index[:, :, train_t_indices]
            hi_mask = np.isfinite(hi_train) & (hi_train != 0.0)
            
            if not np.any(hi_mask):
                raise ValueError("No valid t2m_prism values found in training set after excluding NaNs and zeros.")
            
            valid_hi = hi_train[hi_mask]
            self.hi_mean = torch.tensor(np.mean(valid_hi), dtype=torch.float32)
            self.hi_std  = torch.tensor(np.std(valid_hi), dtype=torch.float32)
            
            print(f"    t2m_prism -> Mean: {self.hi_mean:.4f}, Std: {self.hi_std:.4f}")

            gp_train = self.geopotential[:, :, 0, train_t_indices]
            sm_train = self.soil_moisture[:, :, train_t_indices]
            slp_train = self.slp[:, :, train_t_indices]
            t2m_train = self.temperature_2m[:, :, train_t_indices]
            q850_train = self.specific_humidity_850[:, :, train_t_indices]
            t850_train = self.temperature_850[:, :, train_t_indices]
            u850_train = self.u_wind_850[:, :, train_t_indices]
            v850_train = self.v_wind_850[:, :, train_t_indices]
            z300_train = self.geopotential_300[:, :, train_t_indices]

            self.stats_mean = torch.tensor(
                [np.nanmean(gp_train), np.nanmean(sm_train), np.nanmean(slp_train),
                 np.nanmean(t2m_train), np.nanmean(q850_train), np.nanmean(t850_train),
                 np.nanmean(u850_train), np.nanmean(v850_train), np.nanmean(z300_train)],
                dtype=torch.float32
            ).view(9, 1, 1)
            self.stats_std = torch.tensor(
                [np.nanstd(gp_train), np.nanstd(sm_train), np.nanstd(slp_train),
                 np.nanstd(t2m_train), np.nanstd(q850_train), np.nanstd(t850_train),
                 np.nanstd(u850_train), np.nanstd(v850_train), np.nanstd(z300_train)],
                dtype=torch.float32
            ).view(9, 1, 1)

            cond_train_subset = self.cond_train[:, train_t_indices]
            self.cond_mean = torch.tensor(np.mean(cond_train_subset, axis=1), dtype=torch.float32)
            self.cond_std = torch.tensor(np.std(cond_train_subset, axis=1), dtype=torch.float32)

            if self.topography is not None:
                topo_land = self.topography[self.topography != 0.0]
                self.topo_mean = torch.tensor(float(np.mean(topo_land)), dtype=torch.float32)
                self.topo_std  = torch.tensor(float(np.std(topo_land)),  dtype=torch.float32)

            if self.global_data is not None:
                global_means = []
                global_stds = []
                for var_name, var_data in self.global_data.items():
                    var_train = var_data[:, :, train_t_indices]
                    global_means.append(float(np.nanmean(var_train)))
                    global_stds.append(float(np.nanstd(var_train)))
                self.global_mean = torch.tensor(global_means, dtype=torch.float32).view(-1, 1, 1)
                self.global_std = torch.tensor(global_stds, dtype=torch.float32).view(-1, 1, 1)
                del var_train
            else:
                self.global_mean = None
                self.global_std = None

            # TOA insolation normalization stats
            toa_subset_doys = self.doy_values[train_t_indices[:500]]
            toa_samples = np.stack([compute_toa_insolation(self.lat_1d_deg, doy)
                                    for doy in toa_subset_doys])
            self.toa_mean = torch.tensor(float(np.mean(toa_samples)), dtype=torch.float32)
            self.toa_std  = torch.tensor(float(np.std(toa_samples)),  dtype=torch.float32)
            print(f"    TOA insolation -> Mean: {self.toa_mean:.2f}, Std: {self.toa_std:.2f}")

            del hi_train, gp_train, sm_train, slp_train, t2m_train, q850_train, t850_train, u850_train, v850_train, z300_train, cond_train_subset
            gc.collect()

        elif normalization_stats is not None:
            self.hi_mean = normalization_stats['hi_mean']
            self.hi_std = normalization_stats['hi_std']
            self.stats_mean = normalization_stats['stats_mean']
            self.stats_std = normalization_stats['stats_std']
            self.cond_mean = normalization_stats['cond_mean']
            self.cond_std = normalization_stats['cond_std']
            self.topo_mean = normalization_stats['topo_mean']
            self.topo_std  = normalization_stats['topo_std']
            self.global_mean = normalization_stats.get('global_mean', None)
            self.global_std = normalization_stats.get('global_std', None)
            self.toa_mean = normalization_stats['toa_mean']
            self.toa_std  = normalization_stats['toa_std']
        else:
            raise ValueError(f"normalization_stats required for mode='{mode}'")

        h, w = self.config.IMAGE_SIZE
        lat_1d = np.linspace(-1.0, 1.0, h)
        lon_1d = np.linspace(-1.0, 1.0, w)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        self.lat_grid = torch.from_numpy(lat_grid).float().unsqueeze(0)
        self.lon_grid = torch.from_numpy(lon_grid).float().unsqueeze(0)

        if mode == 'train':
            self.indices = train_indices if train_indices is not None else list(range(1, self.n_timesteps - self.config.LEAD_TIME))
        elif mode == 'val':
            self.indices = val_indices if val_indices is not None else list(range(1, self.n_timesteps - self.config.LEAD_TIME))
        elif mode == 'test':
            self.indices = test_indices if test_indices is not None else list(range(1, self.n_timesteps - self.config.LEAD_TIME))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        h, w = self.config.IMAGE_SIZE

        x_t_slice      = self.heat_index[:, :, t]
        x_tm1_slice    = self.heat_index[:, :, t - 1]
        x_tm2_slice    = self.heat_index[:, :, t - 2]
        x_target_slice = self.heat_index[:, :, t + self.config.LEAD_TIME]

        raw_x_t = torch.from_numpy(self.heat_index[:, :, t].copy())
        land_mask = (torch.isfinite(raw_x_t) & (raw_x_t != 0.0)).float().unsqueeze(0)

        def normalize_hi(data_slice):
            tensor = torch.from_numpy(data_slice.copy()).float()
            valid = torch.isfinite(tensor) & (tensor != 0.0)
        
            safe = tensor.clone()
            safe[~valid] = self.hi_mean
        
            normed = (safe - self.hi_mean) / (self.hi_std + 1e-8)
            normed = normed.unsqueeze(0)
        
            valid_mask = valid.float().unsqueeze(0)
            return normed * valid_mask + Config.OCEAN_FILL * (1 - valid_mask)

        x_t   = normalize_hi(x_t_slice)
        x_tm1 = normalize_hi(x_tm1_slice)
        x_tm2 = normalize_hi(x_tm2_slice)
        y     = normalize_hi(x_target_slice)

        cond_slice = self.cond_train[:, t]
        vec_c = (torch.from_numpy(cond_slice.copy()) - self.cond_mean) / (self.cond_std + 1e-8)

        gp_t   = torch.from_numpy(self.geopotential[:, :, 0, t].copy())
        sm_t   = torch.from_numpy(self.soil_moisture[:, :, t].copy())
        slp_t  = torch.from_numpy(self.slp[:, :, t].copy())
        t2m_t  = torch.from_numpy(self.temperature_2m[:, :, t].copy())
        q850_t = torch.from_numpy(self.specific_humidity_850[:, :, t].copy())
        t850_t = torch.from_numpy(self.temperature_850[:, :, t].copy())
        u850_t = torch.from_numpy(self.u_wind_850[:, :, t].copy())
        v850_t = torch.from_numpy(self.v_wind_850[:, :, t].copy())
        z300_t = torch.from_numpy(self.geopotential_300[:, :, t].copy())

        physics = torch.stack([gp_t, sm_t, slp_t, t2m_t, q850_t, t850_t, u850_t, v850_t, z300_t], dim=0)
        physics = torch.nan_to_num(physics, nan=0.0)
        physics = (physics - self.stats_mean) / (self.stats_std + 1e-8)
        physics = physics * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        topo = torch.from_numpy(self.topography.copy())
        topo = ((topo - self.topo_mean) / (self.topo_std + 1e-8)).unsqueeze(0)
        topo = topo * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        lat_c = self.lat_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        lon_c = self.lon_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        # Computed variables: doy sin/cos, TOA insolation, land-sea mask
        doy_sin_ch = torch.full((1, h, w), self.doy_sin_arr[t], dtype=torch.float32)
        doy_cos_ch = torch.full((1, h, w), self.doy_cos_arr[t], dtype=torch.float32)

        toa_1d = compute_toa_insolation(self.lat_1d_deg, self.doy_values[t])
        toa_2d = torch.from_numpy(
            np.broadcast_to(toa_1d[:, None], (h, w)).copy()
        ).float().unsqueeze(0)
        toa_2d = (toa_2d - self.toa_mean) / (self.toa_std + 1e-8)

        land_mask_ch = land_mask  # (1, H, W), values 0/1

        physics = torch.cat([physics, topo, lat_c, lon_c,
                             doy_sin_ch, doy_cos_ch, toa_2d, land_mask_ch], dim=0)

        if self.global_data is not None and self.global_mean is not None:
            global_channels = []
            for var_name, var_data in self.global_data.items():
                g_slice = torch.from_numpy(np.array(var_data[:, :, t], dtype=np.float32))
                global_channels.append(g_slice)
            global_fields = torch.stack(global_channels, dim=0)
            global_fields = (global_fields - self.global_mean) / (self.global_std + 1e-8)
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        # Phase 3: No padding. Return raw (H, W) tensors.
        mask = land_mask

        return y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, t, mask

    def get_conditions_at(self, t):
        """
        Return (spatial_extra, vec_c, global_fields, mask) for timestep t.
        spatial_extra: (16, H, W) = physics(9) + topo(1) + lat(1) + lon(1) + doy_sin(1) + doy_cos(1) + toa(1) + land_mask(1)
        Used by the autoregressive rollout to get conditioning at each step.
        """
        h, w = self.config.IMAGE_SIZE

        raw_x_t = torch.from_numpy(self.heat_index[:, :, t].copy())
        land_mask = (torch.isfinite(raw_x_t) & (raw_x_t != 0.0)).float().unsqueeze(0)

        gp_t   = torch.from_numpy(self.geopotential[:, :, 0, t].copy())
        sm_t   = torch.from_numpy(self.soil_moisture[:, :, t].copy())
        slp_t  = torch.from_numpy(self.slp[:, :, t].copy())
        t2m_t  = torch.from_numpy(self.temperature_2m[:, :, t].copy())
        q850_t = torch.from_numpy(self.specific_humidity_850[:, :, t].copy())
        t850_t = torch.from_numpy(self.temperature_850[:, :, t].copy())
        u850_t = torch.from_numpy(self.u_wind_850[:, :, t].copy())
        v850_t = torch.from_numpy(self.v_wind_850[:, :, t].copy())
        z300_t = torch.from_numpy(self.geopotential_300[:, :, t].copy())

        physics = torch.stack([gp_t, sm_t, slp_t, t2m_t, q850_t, t850_t, u850_t, v850_t, z300_t], dim=0)
        physics = torch.nan_to_num(physics, nan=0.0)
        physics = (physics - self.stats_mean) / (self.stats_std + 1e-8)
        physics = physics * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        topo = torch.from_numpy(self.topography.copy())
        topo = ((topo - self.topo_mean) / (self.topo_std + 1e-8)).unsqueeze(0)
        topo = topo * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        lat_c = self.lat_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        lon_c = self.lon_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        # Computed variables
        doy_sin_ch = torch.full((1, h, w), self.doy_sin_arr[t], dtype=torch.float32)
        doy_cos_ch = torch.full((1, h, w), self.doy_cos_arr[t], dtype=torch.float32)

        toa_1d = compute_toa_insolation(self.lat_1d_deg, self.doy_values[t])
        toa_2d = torch.from_numpy(
            np.broadcast_to(toa_1d[:, None], (h, w)).copy()
        ).float().unsqueeze(0)
        toa_2d = (toa_2d - self.toa_mean) / (self.toa_std + 1e-8)

        land_mask_ch = land_mask

        spatial_extra = torch.cat([physics, topo, lat_c, lon_c,
                                   doy_sin_ch, doy_cos_ch, toa_2d, land_mask_ch], dim=0)  # (16, H, W)

        cond_slice = self.cond_train[:, t]
        vec_c = (torch.from_numpy(cond_slice.copy()) - self.cond_mean) / (self.cond_std + 1e-8)

        if self.global_data is not None and self.global_mean is not None:
            global_channels = []
            for var_name, var_data in self.global_data.items():
                g_slice = torch.from_numpy(np.array(var_data[:, :, t], dtype=np.float32))
                global_channels.append(g_slice)
            global_fields = torch.stack(global_channels, dim=0)
            global_fields = (global_fields - self.global_mean) / (self.global_std + 1e-8)
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        return spatial_extra, vec_c, global_fields, land_mask


def get_normalization_stats(dataset):
    stats = {
        'hi_mean':    dataset.hi_mean,
        'hi_std':     dataset.hi_std,
        'stats_mean': dataset.stats_mean,
        'stats_std':  dataset.stats_std,
        'cond_mean':  dataset.cond_mean,
        'cond_std':   dataset.cond_std,
        'topo_mean':  dataset.topo_mean,
        'topo_std':   dataset.topo_std,
        'toa_mean':   dataset.toa_mean,
        'toa_std':    dataset.toa_std,
        'shared_data': {
            'heat_index':            dataset.heat_index,
            'geopotential':          dataset.geopotential,
            'soil_moisture':         dataset.soil_moisture,
            'slp':                   dataset.slp,
            'cond_train':            dataset.cond_train,
            'topography':            dataset.topography,
            'temperature_2m':        dataset.temperature_2m,
            'specific_humidity_850': dataset.specific_humidity_850,
            'temperature_850':       dataset.temperature_850,
            'u_wind_850':            dataset.u_wind_850,
            'v_wind_850':            dataset.v_wind_850,
            'geopotential_300':      dataset.geopotential_300,
            'time_values':           dataset.time_values,
        },
    }
    if dataset.global_data is not None:
        stats['shared_data']['global_data'] = dataset.global_data
    stats['global_mean'] = dataset.global_mean
    stats['global_std'] = dataset.global_std
    return stats

# ======================================================================================
# CFM LOSS
# ======================================================================================
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

def masked_ssim_01(pred, target, mask, fill=0.5):
    mask = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    pred_m = pred * mask + fill * (1 - mask)
    targ_m = target * mask + fill * (1 - mask)
    return structural_similarity_index_measure(pred_m, targ_m, data_range=1.0)

# ======================================================================================
# OCEAN MASKING UTILITIES
# ======================================================================================
def load_conus_mask(config):
    with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
        hi = nc.variables['t2m_prism']
        if hi.ndim == 4:
            hi_sample = hi[:, :, 0, 0]
        elif hi.ndim == 3:
            hi_sample = hi[:, :, 0]
        else:
            raise ValueError(f"Unexpected t2m_prism ndim: {hi.ndim}")
        mask = (np.abs(hi_sample) > 0.01).astype(np.float32)

    if config.SUBDOMAIN_ENABLED:
        mask = mask[config.LAT_SLICE, config.LON_SLICE]

    land_fraction = mask.sum() / mask.size
    print(f"  Mask loaded: {land_fraction*100:.1f}% land, {(1-land_fraction)*100:.1f}% ocean  "
          f"(shape={mask.shape})")
    return torch.from_numpy(np.ascontiguousarray(mask))

def apply_mask_to_predictions(predictions, mask, fill_value=0.0):
    if isinstance(predictions, torch.Tensor):
        device = predictions.device
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device)
        mask = mask.to(device)
        if predictions.ndim == 4:
            mask_broadcast = mask[None, None, :, :].expand_as(predictions)
        elif predictions.ndim == 3:
            mask_broadcast = mask[None, :, :].expand_as(predictions)
        else:
            mask_broadcast = mask
        return predictions * mask_broadcast + fill_value * (1 - mask_broadcast)
    else:
        predictions = np.array(predictions)
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if predictions.ndim == 4:
            mask_broadcast = np.broadcast_to(mask[None, None, :, :], predictions.shape)
        elif predictions.ndim == 3:
            mask_broadcast = np.broadcast_to(mask[None, :, :], predictions.shape)
        else:
            mask_broadcast = mask
        return predictions * mask_broadcast + fill_value * (1 - mask_broadcast)

# ======================================================================================
# VALIDATION (Phase 2: Autoregressive rollout)
# ======================================================================================
@torch.inference_mode()
def calculate_validation_metrics_cfm(model, val_dataset, device, mask,
                                      n_samples=20, rank=0, world_size=1, ddp=False):
    """
    Validation using autoregressive rollout (ROLLOUT_STEPS 1-day predictions).
    Each validation sample starts at a val index and rolls forward 15 days.
    The final prediction is compared to ground truth at t + ROLLOUT_STEPS.
    """
    import time
    validation_start = time.time()

    model.eval()
    if len(val_dataset) == 0:
        return 0.0, 0.0, 0.0, {}

    per_rank = max(1, n_samples // world_size)
    actual_n_samples = per_rank * world_size

    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, len(val_dataset))
    rank_sample_count = end_idx - start_idx

    if is_main_process():
        print(f"\n  Generating {actual_n_samples} rollout samples across {world_size} GPUs "
              f"({per_rank} per GPU, {Config.ROLLOUT_STEPS} steps each)...")

    h, w = Config.IMAGE_SIZE
    # Mask as (1, 1, H, W) for the rollout function
    if isinstance(mask, torch.Tensor):
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)

    all_preds = []
    all_truth = []

    for sample_i in range(rank_sample_count):
        dataset_idx = start_idx + sample_i
        t_start = val_dataset.indices[dataset_idx]

        # Run autoregressive rollout
        pred = generate_autoregressive_rollout(
            model, val_dataset, t_start, device, h, w, mask_4d,
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS,
            rollout_steps=Config.ROLLOUT_STEPS,
        )
        all_preds.append(torch.from_numpy(pred))

        # Ground truth at t_start + ROLLOUT_STEPS
        gt_idx = t_start + Config.ROLLOUT_STEPS
        gt_raw = val_dataset.heat_index[:, :, gt_idx]
        gt_tensor = torch.from_numpy(gt_raw.copy())
        gt_land = (torch.isfinite(gt_tensor) & (gt_tensor != 0.0)).float()
        gt_clean = torch.nan_to_num(gt_tensor, nan=0.0)
        gt_normed = (gt_clean - val_dataset.hi_mean) / (val_dataset.hi_std + 1e-8)
        gt_normed = gt_normed * gt_land + Config.OCEAN_FILL * (1 - gt_land)
        all_truth.append(gt_normed[:h, :w])

    local_preds = torch.stack(all_preds)
    local_truth = torch.stack(all_truth)

    if ddp:
        local_preds = local_preds.to(device)
        local_truth = local_truth.to(device)

        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_truth = [torch.zeros_like(local_truth) for _ in range(world_size)]

        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_truth, local_truth)

        all_preds = torch.cat(gathered_preds, dim=0).cpu()
        all_truth = torch.cat(gathered_truth, dim=0).cpu()
    else:
        all_preds = local_preds.cpu()
        all_truth = local_truth.cpu()

    if not is_main_process():
        return 0.0, 0.0, 0.0, {}

    all_preds = all_preds.unsqueeze(1)
    all_truth = all_truth.unsqueeze(1)
    all_preds = torch.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
    all_truth = torch.nan_to_num(all_truth, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute mask for metrics
    mask_2d = mask_4d[0, 0, :h, :w].cpu()

    preds_np = all_preds.numpy()
    truth_np = all_truth.numpy()
    m = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds).numpy()

    err2 = (preds_np - truth_np) ** 2
    mse_per_sample = (err2 * m).sum(axis=(1,2,3)) / (m.sum(axis=(1,2,3)) + 1e-8)
    avg_mse  = float(mse_per_sample.mean())
    avg_rmse = float(np.sqrt(avg_mse))

    def to_ssim_space(x):
        return ((x + 3.0) / 6.0).clamp(0.0, 1.0)

    preds_ssim = to_ssim_space(all_preds).to(device)
    truth_ssim = to_ssim_space(all_truth).to(device)
    mask_ssim = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds).to(device)
    avg_ssim = masked_ssim_01(preds_ssim, truth_ssim, mask_ssim, fill=0.5).item()

    improved = calculate_improved_metrics(all_preds, all_truth, mask=mask_2d)

    validation_total = time.time() - validation_start
    print(f"\n  Validation time: {validation_total:.2f}s "
          f"({actual_n_samples/validation_total:.1f} samples/sec across {world_size} GPUs)")
    print(f"  Metrics (15-day rollout): MSE={avg_mse:.6f}, SSIM={avg_ssim:.4f}, R²={improved['r2']:.4f}")

    return avg_mse, avg_rmse, avg_ssim, {
        'variance_ratio':    improved['variance_ratio'],
        'gradient_ratio':    improved['gradient_ratio'],
        'extreme_bias':      improved['extreme_bias'],
        'correlation':       improved['correlation'],
        'r2':                improved['r2'],
        'pred_spatial_std':  improved['pred_spatial_std'],
        'truth_spatial_std': improved['truth_spatial_std'],
    }


def compute_ensemble_statistics(ensemble, mask=None):
    if mask is not None:
        mask2 = mask
        if isinstance(mask2, torch.Tensor):
            mask2 = mask2.cpu().numpy()
        mask2 = np.array(mask2)
        ensemble_masked = np.where(mask2[None, :, :] > 0.5, ensemble, np.nan)
        return {
            'mean': np.nanmean(ensemble_masked, axis=0),
            'std': np.nanstd(ensemble_masked, axis=0),
            'p10': np.nanpercentile(ensemble_masked, 10, axis=0),
            'p90': np.nanpercentile(ensemble_masked, 90, axis=0),
        }
    else:
        return {
            'mean': np.mean(ensemble, axis=0),
            'std': np.std(ensemble, axis=0),
            'p10': np.percentile(ensemble, 10, axis=0),
            'p90': np.percentile(ensemble, 90, axis=0),
        }

# ======================================================================================
# VISUALIZATION
# ======================================================================================
def save_validation_plots(model, val_dataset, device, mask, epoch, save_dir, n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_np = np.array(mask)
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()

    collected = 0
    for dataset_idx in range(min(n_samples, len(val_dataset))):
        t_start = val_dataset.indices[dataset_idx]

        pred = generate_autoregressive_rollout(
            model, val_dataset, t_start, device, h, w, mask_4d,
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS,
            rollout_steps=Config.ROLLOUT_STEPS,
        )

        gt_idx = t_start + Config.ROLLOUT_STEPS
        gt_raw = val_dataset.heat_index[:, :, gt_idx]
        gt_tensor = torch.from_numpy(gt_raw.copy())
        gt_land = (torch.isfinite(gt_tensor) & (gt_tensor != 0.0)).float()
        gt_clean = torch.nan_to_num(gt_tensor, nan=0.0)
        truth = (gt_clean - val_dataset.hi_mean) / (val_dataset.hi_std + 1e-8)
        truth = (truth * gt_land)[:h, :w].numpy()

        pred_display = np.where(mask_np[:h, :w] > 0.5, pred, np.nan)
        truth_display = np.where(mask_np[:h, :w] > 0.5, truth, np.nan)

        valid_mask = mask_np[:h, :w] > 0.5
        p_v = pred[valid_mask]
        t_v = truth[valid_mask]
        r2 = 1 - np.sum((t_v - p_v)**2) / (np.sum((t_v - t_v.mean())**2) + 1e-8)
        corr = np.corrcoef(t_v, p_v)[0, 1] if t_v.std() > 1e-6 else 0.0
        mae = np.mean(np.abs(t_v - p_v))

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        vmin, vmax = -3.0, 3.0
        im = axes[0].imshow(truth_display, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title('Ground Truth', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0], label='Z-score')
        im = axes[1].imshow(pred_display, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'Prediction (15-day rollout)\nMAE={mae:.3f}, r={corr:.3f}, R²={r2:.3f}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Z-score')
        diff_display = np.where(mask_np[:h, :w] > 0.5, pred - truth, np.nan)
        im = axes[2].imshow(diff_display, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
        axes[2].set_title('Prediction - Truth', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[2], label='Difference (Z-score)')
        fig.suptitle(f'Epoch {epoch} — Validation Sample {collected + 1} (15-day rollout)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        fname = os.path.join(save_dir, f'val_epoch{epoch:04d}_sample{collected+1}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        collected += 1
    print(f"  Saved {collected} validation plots to: {save_dir}")


def save_mask_plot(mask, out_dir, fname="conus_mask.png"):
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)
    if m.ndim == 3: m = m[0]
    land_frac = float(np.mean(m))
    plt.figure(figsize=(12, 5))
    plt.imshow(m, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    plt.title(f"Land/Ocean Mask (land fraction = {land_frac:.3f})")
    plt.colorbar(label="1=land, 0=ocean")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()


# ======================================================================================
# DATA CACHE
# ======================================================================================
def apply_subdomain_config():
    """Compute subdomain slices and override Config.

    Must be called once at the start of train_model() or _test(),
    before load_conus_mask() and prepare_shared_data().
    """
    if not Config.SUBDOMAIN_ENABLED:
        Config.LAT_SLICE = slice(None)
        Config.LON_SLICE = slice(None)
        return

    H_full, W_full = Config.IMAGE_SIZE
    lat_full = np.linspace(Config.CONUS_LAT_RANGE[0], Config.CONUS_LAT_RANGE[1], H_full)
    lon_full = np.linspace(Config.CONUS_LON_RANGE[0], Config.CONUS_LON_RANGE[1], W_full)

    lat_idx = np.where((lat_full >= Config.SUBDOMAIN_LAT_RANGE[0]) &
                       (lat_full <= Config.SUBDOMAIN_LAT_RANGE[1]))[0]
    lon_idx = np.where((lon_full >= Config.SUBDOMAIN_LON_RANGE[0]) &
                       (lon_full <= Config.SUBDOMAIN_LON_RANGE[1]))[0]

    Config.LAT_SLICE = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    Config.LON_SLICE = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)

    H_sub = Config.LAT_SLICE.stop - Config.LAT_SLICE.start
    W_sub = Config.LON_SLICE.stop - Config.LON_SLICE.start

    Config.IMAGE_SIZE = (H_sub, W_sub)
    Config.CONUS_LAT_RANGE = (float(lat_full[Config.LAT_SLICE.start]),
                              float(lat_full[Config.LAT_SLICE.stop - 1]))
    Config.CONUS_LON_RANGE = (float(lon_full[Config.LON_SLICE.start]),
                              float(lon_full[Config.LON_SLICE.stop - 1]))

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*80}")
        print(f"SUBDOMAIN ACTIVE")
        print(f"{'='*80}")
        print(f"  Lat indices: [{Config.LAT_SLICE.start}:{Config.LAT_SLICE.stop}]  ({H_sub} points)")
        print(f"  Lon indices: [{Config.LON_SLICE.start}:{Config.LON_SLICE.stop}]  ({W_sub} points)")
        print(f"  Lat range:   {Config.CONUS_LAT_RANGE[0]:.3f} -> {Config.CONUS_LAT_RANGE[1]:.3f}")
        print(f"  Lon range:   {Config.CONUS_LON_RANGE[0]:.3f} -> {Config.CONUS_LON_RANGE[1]:.3f}")
        print(f"  Reduction:   {(H_full*W_full)/(H_sub*W_sub):.1f}x")
        print(f"{'='*80}\n")
        
def prepare_shared_data(config, rank, world_size, ddp):
    sub_tag = ""
    if config.SUBDOMAIN_ENABLED:
        la0, la1 = config.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = config.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"
    cache_dir = os.path.join(config.OUTPUT_DIR, "data_cache" + sub_tag)
    os.makedirs(cache_dir, exist_ok=True)

    TOPO_PATH = "/blue/nessie/mostafarezaali/Teleconnection/CONUS_topography_ETOPO2022_60s_on_model_grid.nc"

    paths = {
        "heat_index":   os.path.join(cache_dir, "heat_index.npy"),
        "geopotential": os.path.join(cache_dir, "geopotential.npy"),
        "soil_moisture":os.path.join(cache_dir, "soil_moisture.npy"),
        "slp":          os.path.join(cache_dir, "slp.npy"),
        "cond_train":   os.path.join(cache_dir, "cond_train.npy"),
        "topography":   os.path.join(cache_dir, "topography.npy"),
        "temperature_2m":        os.path.join(cache_dir, "temperature_2m.npy"),
        "specific_humidity_850":os.path.join(cache_dir, "specific_humidity_850.npy"),
        "temperature_850":       os.path.join(cache_dir, "temperature_850.npy"),
        "u_wind_850":            os.path.join(cache_dir, "u_wind_850.npy"),
        "v_wind_850":            os.path.join(cache_dir, "v_wind_850.npy"),
        "geopotential_300":      os.path.join(cache_dir, "geopotential_300.npy"),
        "time_values":           os.path.join(cache_dir, "time_values.npy"),
    }

    global_cache_dir = os.path.join(cache_dir, "global")
    os.makedirs(global_cache_dir, exist_ok=True)
    global_paths = {}
    for var_name in config.GLOBAL_VARIABLES:
        global_paths[var_name] = os.path.join(global_cache_dir, f"{var_name}.npy")

    if is_main_process():
        cache_ok = all(os.path.exists(p) for p in paths.values())
        if cache_ok:
            try:
                nc_mtime = max(os.path.getmtime(config.TRAINING_DATA_PATH), os.path.getmtime(TOPO_PATH))
                cache_mtime = min(os.path.getmtime(p) for p in paths.values())
                if nc_mtime > cache_mtime:
                    cache_ok = False
            except Exception:
                pass

        if not cache_ok:
            lat_s = config.LAT_SLICE if config.SUBDOMAIN_ENABLED else slice(None)
            lon_s = config.LON_SLICE if config.SUBDOMAIN_ENABLED else slice(None)
            tag = "subdomain" if config.SUBDOMAIN_ENABLED else "full CONUS"
            print(f"Rank 0: writing {tag} cache (lat={lat_s}, lon={lon_s})...")

            with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc:
                # Load full arrays then slice in numpy (netCDF4 direct slicing
                # on large variables produces incorrect results for this file)
                print(f"  Loading and slicing t2m_prism...")
                hi_raw = nc.variables["t2m_prism"][:]
                if hi_raw.ndim == 4:
                    hi = np.array(hi_raw[:, :, 0, :], dtype=np.float32)[lat_s, lon_s, :]
                else:
                    hi = np.array(hi_raw, dtype=np.float32)[lat_s, lon_s, :]
                print(f"    t2m_prism cached shape: {hi.shape}")
                np.save(paths["heat_index"], hi)
                del hi, hi_raw
                gc.collect()

                def load_and_slice_3d(var_name):
                    """Load full 3D var (H, W, T), slice to subdomain."""
                    raw = np.array(nc.variables[var_name][:], dtype=np.float32)
                    sliced = raw[lat_s, lon_s, :]
                    del raw
                    return sliced

                def load_and_slice_4d(var_name):
                    """Load full 4D var (H, W, L, T), slice to subdomain."""
                    raw = np.array(nc.variables[var_name][:], dtype=np.float32)
                    sliced = raw[lat_s, lon_s, :, :]
                    del raw
                    return sliced

                print(f"  Loading and slicing remaining variables...")
                np.save(paths["geopotential"], load_and_slice_4d("geopotential"))
                gc.collect()
                np.save(paths["soil_moisture"], load_and_slice_3d("soil_moisture"))
                gc.collect()
                np.save(paths["slp"], load_and_slice_3d("sea_level_pressure"))
                gc.collect()
                np.save(paths["cond_train"], np.array(nc.variables["CondTrain"][:], dtype=np.float32))
                np.save(paths["temperature_2m"], load_and_slice_3d("temperature_2m"))
                gc.collect()
                np.save(paths["specific_humidity_850"], load_and_slice_3d("specific_humidity_850"))
                gc.collect()
                np.save(paths["temperature_850"], load_and_slice_3d("temperature_850"))
                gc.collect()
                np.save(paths["u_wind_850"], load_and_slice_3d("u_wind_850"))
                gc.collect()
                np.save(paths["v_wind_850"], load_and_slice_3d("v_wind_850"))
                gc.collect()
                np.save(paths["geopotential_300"], load_and_slice_3d("geopotential_300"))
                gc.collect()
                np.save(paths["time_values"], np.array(nc.variables["time"][:], dtype=np.float64))
                print(f"  All variables sliced and cached.")

            with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
                topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
                topo = np.flipud(topo)
            
            # Ensure topography matches the FULL CONUS grid orientation before slicing
            H_full, W_full = 621, 1405  # original IMAGE_SIZE before subdomain override
            if topo.shape == (W_full, H_full):
                topo = topo.T
            elif topo.shape != (H_full, W_full):
                raise ValueError(f"Unexpected topography shape: {topo.shape}, expected ({H_full},{W_full}) or ({W_full},{H_full})")
            
            if config.SUBDOMAIN_ENABLED:
                topo = topo[lat_s, lon_s]
            np.save(paths["topography"], np.ascontiguousarray(topo))

        global_cache_ok = all(os.path.exists(p) for p in global_paths.values())
        if not global_cache_ok:
            print(f"Rank 0: caching global teleconnection fields...")
            with NetCDFDataset(config.GLOBAL_DATA_PATH, "r") as nc_g:
                for var_name in config.GLOBAL_VARIABLES:
                    data = np.array(nc_g.variables[var_name][:], dtype=np.float32)
                    if data.ndim == 4:
                        data = data[:, 0, :, :]
                    data = np.transpose(data, (1, 2, 0))
                    nan_count = np.isnan(data).sum()
                    if nan_count > 0:
                        data = np.nan_to_num(data, nan=0.0)
                    np.save(global_paths[var_name], data)
                    del data
            print(f"Rank 0: wrote global data cache")
        else:
            print(f"Rank 0: using existing global data cache")

    if ddp:
        dist.barrier()

    shm_dir = "/dev/shm/cfm_cache" + sub_tag

    if not ddp or dist.get_rank() == 0:
        os.makedirs(shm_dir, exist_ok=True)
        for key, src_path in paths.items():
            dst = os.path.join(shm_dir, os.path.basename(src_path))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        shm_global = os.path.join(shm_dir, "global")
        os.makedirs(shm_global, exist_ok=True)
        for var_name, src_path in global_paths.items():
            dst = os.path.join(shm_global, os.path.basename(src_path))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        print(f"Rank 0: copied data cache to {shm_dir}")

    if ddp:
        dist.barrier()

    shm_paths = {k: os.path.join(shm_dir, os.path.basename(v)) for k, v in paths.items()}

    shared = {
        k: np.load(shm_paths[k], mmap_mode="r")
        for k in paths
    }

    from collections import OrderedDict
    global_data = OrderedDict()
    for var_name in config.GLOBAL_VARIABLES:
        gpath = os.path.join(shm_dir, "global", f"{var_name}.npy")
        global_data[var_name] = np.load(gpath, mmap_mode="r")
    shared['global_data'] = global_data

    return shared


torch.backends.cudnn.benchmark = True


# ======================================================================================
# DIAGNOSTIC FUNCTIONS (added from Implementation guid)
# ======================================================================================

@torch.inference_mode()
def diagnose_rollout_error_growth(model, val_dataset, device, mask, n_samples=10):
    """
    Run rollout with return_all_steps=True and compute error at each day.
    Prints a day-by-day error table and saves a plot.
    """
    # Ensure the model we use (EMA) has the 'generate_autoregressive_rollout' call with return_all_steps
    # The implementation in mode_dispatch must support return_all_steps=True.
    model.eval()
    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
        mask_2d = mask[:h, :w].cpu().numpy()
    else:
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)
        mask_2d = mask[:h, :w]

    valid_mask = mask_2d > 0.5

    # Accumulators per step
    step_mse = np.zeros(Config.ROLLOUT_STEPS)
    step_mae = np.zeros(Config.ROLLOUT_STEPS)
    step_corr = np.zeros(Config.ROLLOUT_STEPS)
    step_bias = np.zeros(Config.ROLLOUT_STEPS)
    step_pred_std = np.zeros(Config.ROLLOUT_STEPS)
    step_truth_std = np.zeros(Config.ROLLOUT_STEPS)
    count = 0

    for sample_i in range(min(n_samples, len(val_dataset))):
        t_start = val_dataset.indices[sample_i]

        all_steps = generate_autoregressive_rollout(
            model, val_dataset, t_start, device, h, w, mask_4d,
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS,
            rollout_steps=Config.ROLLOUT_STEPS,
            return_all_steps=True,
        )

        for k, pred_k in enumerate(all_steps):
            gt_idx = t_start + k + 1
            if gt_idx >= val_dataset.heat_index.shape[-1]:
                continue
            gt_raw = val_dataset.heat_index[:, :, gt_idx]
            gt_tensor = torch.from_numpy(gt_raw.copy())
            gt_land = (torch.isfinite(gt_tensor) & (gt_tensor != 0.0)).float()
            gt_clean = torch.nan_to_num(gt_tensor, nan=0.0)
            gt_normed = ((gt_clean - val_dataset.hi_mean) / (val_dataset.hi_std + 1e-8) * gt_land)[:h, :w].numpy()

            p_v = pred_k[valid_mask]
            t_v = gt_normed[valid_mask]

            step_mse[k] += np.mean((p_v - t_v) ** 2)
            step_mae[k] += np.mean(np.abs(p_v - t_v))
            step_bias[k] += np.mean(p_v) - np.mean(t_v)
            step_pred_std[k] += np.std(p_v)
            step_truth_std[k] += np.std(t_v)
            if np.std(t_v) > 1e-6 and np.std(p_v) > 1e-6:
                step_corr[k] += np.corrcoef(p_v, t_v)[0, 1]
            else:
                step_corr[k] += 0.0

        count += 1

    if count == 0:
        print("  No rollout samples available for diagnostics.")
        return

    step_mse /= count
    step_mae /= count
    step_corr /= count
    step_bias /= count
    step_pred_std /= count
    step_truth_std /= count
    step_r2 = 1.0 - step_mse / (step_truth_std ** 2 + 1e-8)

    print(f"\n  {'='*90}")
    print(f"  ROLLOUT ERROR GROWTH ({count} samples)")
    print(f"  {'='*90}")
    print(f"  {'Day':>4s}  {'MSE':>8s}  {'MAE':>8s}  {'R²':>8s}  {'Corr':>8s}  {'Bias':>8s}  {'PredStd':>8s}  {'TruthStd':>8s}  {'VarRatio':>8s}")
    print(f"  {'-'*90}")
    for k in range(Config.ROLLOUT_STEPS):
        vr = step_pred_std[k] / (step_truth_std[k] + 1e-8)
        print(f"  {k+1:4d}  {step_mse[k]:8.4f}  {step_mae[k]:8.4f}  {step_r2[k]:8.4f}  "
              f"{step_corr[k]:8.4f}  {step_bias[k]:+8.4f}  {step_pred_std[k]:8.4f}  "
              f"{step_truth_std[k]:8.4f}  {vr:8.4f}")

    # Identify failure mode
    if step_pred_std[-1] < 0.3 * step_truth_std[-1]:
        print(f"\n  DIAGNOSIS: Variance collapse. Predictions smoothing to climatology.")
        print(f"  Pred std at day 15 is {step_pred_std[-1]:.3f} vs truth {step_truth_std[-1]:.3f}.")
    elif np.abs(step_bias[-1]) > 0.5:
        print(f"\n  DIAGNOSIS: Systematic bias. Mean prediction drifts {step_bias[-1]:+.3f} from truth.")
    elif step_corr[0] > 0.8 and step_corr[-1] < 0.4:
        day_half = np.argmax(step_corr < 0.5 * step_corr[0]) + 1
        print(f"\n  DIAGNOSIS: Error accumulation. Correlation drops below 50% of day-1 at day {day_half}.")
    elif step_r2[0] < 0.3:
        print(f"\n  DIAGNOSIS: Even 1-day prediction is weak (R²={step_r2[0]:.3f}). Single-step model needs work first.")

    # Save plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(range(1, Config.ROLLOUT_STEPS + 1), step_mse, 'o-', color='red')
    axes[0, 0].set_xlabel('Rollout Day')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE vs Rollout Day')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(range(1, Config.ROLLOUT_STEPS + 1), step_corr, 's-', color='blue')
    axes[0, 1].set_xlabel('Rollout Day')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Spatial Correlation vs Rollout Day')
    axes[0, 1].set_ylim(-0.1, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(1, Config.ROLLOUT_STEPS + 1), step_pred_std, 'o-', color='orange', label='Prediction')
    axes[1, 0].plot(range(1, Config.ROLLOUT_STEPS + 1), step_truth_std, 's-', color='green', label='Truth')
    axes[1, 0].set_xlabel('Rollout Day')
    axes[1, 0].set_ylabel('Spatial Std')
    axes[1, 0].set_title('Spatial Variance: Prediction vs Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(range(1, Config.ROLLOUT_STEPS + 1), step_bias, 'D-', color='purple')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Rollout Day')
    axes[1, 1].set_ylabel('Mean Bias')
    axes[1, 1].set_title('Mean Bias vs Rollout Day')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Rollout Error Growth Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(Config.PLOTS_DIR, "rollout_error_growth.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved rollout error growth plot to: {plot_path}")


def diagnose_data_statistics(dataset, mask, n_samples=50):
    """
    Check whether input and target distributions look healthy.
    Catches: dead channels, extreme values, zero-variance features, NaN leaks.
    """
    h, w = Config.IMAGE_SIZE
    if isinstance(mask, torch.Tensor):
        mask_2d = mask[:h, :w].cpu()
    else:
        mask_2d = torch.from_numpy(mask[:h, :w])

    valid_mask = mask_2d.numpy() > 0.5

    hi_vals, target_vals = [], []
    physics_stats = None

    for i in range(min(n_samples, len(dataset))):
        y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, m = dataset[i]

        hi_vals.append(x_t[0][valid_mask].numpy())
        target_vals.append(y[0][valid_mask].numpy())

        sc = spatial_c.numpy()
        if physics_stats is None:
            physics_stats = {
                'mean': np.zeros(sc.shape[0]),
                'std': np.zeros(sc.shape[0]),
                'min': np.full(sc.shape[0], np.inf),
                'max': np.full(sc.shape[0], -np.inf),
                'nan_count': np.zeros(sc.shape[0]),
            }
        for ch in range(sc.shape[0]):
            ch_land = sc[ch][valid_mask]
            physics_stats['mean'][ch] += np.mean(ch_land)
            physics_stats['std'][ch] += np.std(ch_land)
            physics_stats['min'][ch] = min(physics_stats['min'][ch], np.min(ch_land))
            physics_stats['max'][ch] = max(physics_stats['max'][ch], np.max(ch_land))
            physics_stats['nan_count'][ch] += np.isnan(ch_land).sum()

    n = min(n_samples, len(dataset))
    physics_stats['mean'] /= n
    physics_stats['std'] /= n

    all_hi = np.concatenate(hi_vals)
    all_target = np.concatenate(target_vals)

    print(f"\n  {'='*70}")
    print(f"  DATA STATISTICS DIAGNOSTIC ({n} samples)")
    print(f"  {'='*70}")

    print(f"\n  t2m_prism (x_t) over land:")
    print(f"    Mean: {np.mean(all_hi):.4f}, Std: {np.std(all_hi):.4f}")
    print(f"    Min:  {np.min(all_hi):.4f}, Max: {np.max(all_hi):.4f}")
    print(f"    Zeros: {(np.abs(all_hi) < 1e-6).sum()} / {len(all_hi)}")

    print(f"\n  Target (y) over land:")
    print(f"    Mean: {np.mean(all_target):.4f}, Std: {np.std(all_target):.4f}")
    print(f"    Min:  {np.min(all_target):.4f}, Max: {np.max(all_target):.4f}")

    corr_xt_y = np.corrcoef(all_hi[:10000], all_target[:10000])[0, 1]
    print(f"\n  Autocorrelation x_t vs y (1-day): {corr_xt_y:.4f}")
    if corr_xt_y > 0.95:
        print(f"  WARNING: x_t and y are nearly identical. LEAD_TIME=1 target is too easy.")
        print(f"  This means training loss drops but the model learns persistence, not forecasting.")

    channel_names = ['gp500', 'soil_m', 'slp', 't2m', 'q850', 't850',
                     'u850', 'v850', 'z300', 'topo', 'lat', 'lon',
                     'doy_sin', 'doy_cos', 'toa', 'land_mask']
    if len(channel_names) < physics_stats['mean'].shape[0]:
        channel_names += [f'ch{i}' for i in range(len(channel_names), physics_stats['mean'].shape[0])]

    print(f"\n  Spatial conditioning channels (over land, {n} samples):")
    print(f"  {'Ch':>3s} {'Name':>10s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'NaN':>6s}  {'Status'}")
    print(f"  {'-'*75}")
    for ch in range(min(len(channel_names), physics_stats['mean'].shape[0])):
        name = channel_names[ch] if ch < len(channel_names) else f'ch{ch}'
        status = ""
        if physics_stats['std'][ch] < 1e-6 and ch not in (12, 13):  # doy_sin/cos are spatially uniform by design
            status = "DEAD (zero variance)"
        elif physics_stats['std'][ch] < 1e-6 and ch in (12, 13):
            status = "OK (spatially uniform, varies across time)"
        elif physics_stats['nan_count'][ch] > 0:
            status = f"NaN LEAK ({int(physics_stats['nan_count'][ch])})"
        elif np.abs(physics_stats['mean'][ch]) > 10:
            status = "POORLY NORMALIZED"
        else:
            status = "OK"
        print(f"  {ch:3d} {name:>10s}  {physics_stats['mean'][ch]:8.4f}  {physics_stats['std'][ch]:8.4f}  "
              f"{physics_stats['min'][ch]:8.4f}  {physics_stats['max'][ch]:8.4f}  "
              f"{int(physics_stats['nan_count'][ch]):6d}  {status}")


@torch.inference_mode()
def diagnose_single_step_generation(model, val_dataset, device, mask, n_samples=20):
    """
    Test single-step generation quality (not rollout).
    Compares: reconstruction (teacher-forced) vs generation (from x_t).
    """
    model.eval()
    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_2d = mask[:h, :w].cpu().numpy()
    else:
        mask_2d = mask[:h, :w]
    valid_mask = mask_2d > 0.5

    if isinstance(mask, torch.Tensor):
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)

    gen_mse, gen_corr, gen_bias, gen_pred_std = [], [], [], []

    for i in range(min(n_samples, len(val_dataset))):
        y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, m = val_dataset[i]

        pred = generate_sample(
            model,
            spatial_c.unsqueeze(0).to(device),
            vec_c.unsqueeze(0).to(device),
            x_t.unsqueeze(0).to(device),
            x_tm1.unsqueeze(0).to(device),
            x_tm2.unsqueeze(0).to(device),
            global_fields.unsqueeze(0).to(device),
            device, h, w,
            mask_4d,
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS,
        )

        truth = y[0, :h, :w].numpy()
        p_v = pred[valid_mask]
        t_v = truth[valid_mask]

        gen_mse.append(np.mean((p_v - t_v) ** 2))
        gen_bias.append(np.mean(p_v) - np.mean(t_v))
        gen_pred_std.append(np.std(p_v))
        if np.std(t_v) > 1e-6 and np.std(p_v) > 1e-6:
            gen_corr.append(np.corrcoef(p_v, t_v)[0, 1])
        else:
            gen_corr.append(0.0)

    print(f"\n  {'='*60}")
    print(f"  SINGLE-STEP GENERATION (1-day, {len(gen_mse)} samples)")
    print(f"  {'='*60}")
    print(f"  MSE:       {np.mean(gen_mse):.4f} +/- {np.std(gen_mse):.4f}")
    print(f"  Corr:      {np.mean(gen_corr):.4f} +/- {np.std(gen_corr):.4f}")
    print(f"  Bias:      {np.mean(gen_bias):+.4f} +/- {np.std(gen_bias):.4f}")
    print(f"  Pred Std:  {np.mean(gen_pred_std):.4f}")

    if np.mean(gen_corr) < 0.5:
        print(f"\n  DIAGNOSIS: Single-step generation already poor (corr={np.mean(gen_corr):.3f}).")
        print(f"  The model cannot predict 1 day ahead from generation. Rollout will only make this worse.")
        print(f"  Focus on improving single-step quality before evaluating rollout.")
    elif np.mean(gen_pred_std) < 0.3:
        print(f"\n  DIAGNOSIS: Single-step predictions have collapsed variance ({np.mean(gen_pred_std):.3f}).")
        print(f"  Model predicts near-constant fields. Check CFM time sampling or loss weighting.")


# ======================================================================================
# TRAINING LOOP
# ======================================================================================
def train_model(rank=0, world_size=1, checkpoint_path=None):
    ddp = world_size > 1

    spike_logger = setup_spike_logger(Config.OUTPUT_DIR)

    if ddp:
        setup_ddp(rank, world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    apply_subdomain_config()

    conus_mask = load_conus_mask(Config)

    fm = FlowMatching().to(device)

    if is_main_process():
        print("\n" + "=" * 80)
        print("CONDITIONAL FLOW MATCHING (CFM) - DDP + ICOSAHEDRAL MESH GNN")
        print(f"World size: {world_size}")
        print(f"Mesh level: {Config.MESH_REFINEMENT_LEVEL}, Rounds: {Config.MESH_PROCESSOR_ROUNDS}")
        print(f"Lead time: {Config.LEAD_TIME} day, Rollout: {Config.ROLLOUT_STEPS} steps at validation")
        print("=" * 80 + "\n")

    # Data
    shared_data = prepare_shared_data(Config, rank, world_size, ddp)
    n_timesteps = shared_data['heat_index'].shape[-1]

    # ---- Cache verification ----
    if is_main_process():
        print("\n  Cache verification...")
        hi_cached = shared_data['heat_index']
        print(f"    heat_index shape: {hi_cached.shape}")
        print(f"    heat_index dtype: {hi_cached.dtype}")

        # Check a few timesteps for spatial autocorrelation
        test_t = min(100, hi_cached.shape[-1] - 2)
        field_t0 = np.array(hi_cached[:, :, test_t], dtype=np.float32)
        field_t1 = np.array(hi_cached[:, :, test_t + 1], dtype=np.float32)

        valid = np.isfinite(field_t0) & np.isfinite(field_t1) & (field_t0 != 0) & (field_t1 != 0)
        if valid.sum() > 100:
            r = np.corrcoef(field_t0[valid], field_t1[valid])[0, 1]
            print(f"    Raw spatial autocorr (t={test_t} vs t={test_t+1}): r={r:.4f}")
            if r < -0.5:
                raise ValueError(
                    f"FATAL: Consecutive-day spatial correlation is {r:.4f}. "
                    f"Strongly negative correlation indicates data corruption. "
                    f"Delete data_cache_sub* and /dev/shm/cfm_cache_sub* and retry."
                )
            elif r < 0.3:
                print(f"    NOTE: Low autocorr ({r:.4f}) is expected for anomaly/exceedance variables.")
        else:
            print(f"    WARNING: Only {valid.sum()} valid pixels at t={test_t}")

        # Also verify against the original NetCDF
        with NetCDFDataset(Config.TRAINING_DATA_PATH, 'r') as nc:
            hi_var = nc.variables['t2m_prism']
            lat_s = Config.LAT_SLICE if Config.SUBDOMAIN_ENABLED else slice(None)
            lon_s = Config.LON_SLICE if Config.SUBDOMAIN_ENABLED else slice(None)
            if hi_var.ndim == 4:
                direct = np.array(hi_var[lat_s, lon_s, :, :], dtype=np.float32)[:, :, 0, test_t]
            else:
                direct = np.array(hi_var[lat_s, lon_s, test_t], dtype=np.float32)
            cached = np.array(hi_cached[:, :, test_t], dtype=np.float32)
            max_diff = np.nanmax(np.abs(direct - cached))
            print(f"    Cache vs NetCDF max difference at t={test_t}: {max_diff:.6f}")
            if max_diff > 1e-4:
                raise ValueError(
                    f"FATAL: Cache disagrees with source NetCDF by {max_diff:.4f}. "
                    f"Delete data_cache_sub* and retry."
                )
        print("    Cache verification PASSED\n")

    # Read time values for season boundary detection
    time_values = np.array(shared_data['time_values'])

    runs = detect_continuous_runs(time_values)
    if is_main_process():
        print(f"Detected {len(runs)} continuous runs (expected ~43 MJJAS seasons)")

    # Build valid indices for training (LEAD_TIME=1) and val/test (ROLLOUT_STEPS=15)
    train_valid = build_valid_indices(runs, lead_time=Config.LEAD_TIME, min_history=2)
    rollout_valid = build_valid_indices(runs, lead_time=Config.ROLLOUT_STEPS, min_history=2)

    # Temporal split on training indices
    total_train = len(train_valid)
    test_size = max(1, int(total_train * Config.TEST_FRACTION))
    val_size  = max(1, int(total_train * Config.VAL_FRACTION))

    # For val/test: filter to indices that support full rollout
    rollout_set = set(rollout_valid)

    # Split: train uses early portion, val/test use later portions
    # First do a basic temporal split on all training-eligible indices
    train_end = total_train - test_size - val_size
    train_indices = train_valid[:train_end]

    # Val and test must support full rollout
    val_candidates = train_valid[train_end:train_end + val_size]
    val_indices = [i for i in val_candidates if i in rollout_set]

    test_candidates = train_valid[train_end + val_size:]
    test_indices = [i for i in test_candidates if i in rollout_set]

    if is_main_process():
        print(f"\nStrict Temporal Split (Lead={Config.LEAD_TIME}, Rollout={Config.ROLLOUT_STEPS}):")
        print(f"  Train: {len(train_indices)} samples (1-day target)")
        print(f"  Val:   {len(val_indices)} samples (15-day rollout eligible)")
        print(f"  Test:  {len(test_indices)} samples (15-day rollout eligible)\n")


    # Normalization stats (subdomain-aware path)
    sub_tag = ""
    if Config.SUBDOMAIN_ENABLED:
        la0, la1 = Config.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = Config.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"
    stats_path = os.path.join(Config.OUTPUT_DIR, "data_cache", f"norm_stats_v2{sub_tag}.npz")

    if is_main_process():
        if not os.path.exists(stats_path):
            print("  Rank 0: Calculating Z-Score statistics...")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats = get_normalization_stats(tmp_dataset)
            save_dict = {
                'hi_mean': float(norm_stats['hi_mean']), 'hi_std': float(norm_stats['hi_std']),
                'stats_mean': norm_stats['stats_mean'].numpy(), 'stats_std': norm_stats['stats_std'].numpy(),
                'cond_mean': norm_stats['cond_mean'].numpy(), 'cond_std': norm_stats['cond_std'].numpy(),
                'topo_mean': float(norm_stats['topo_mean']), 'topo_std': float(norm_stats['topo_std']),
                'toa_mean': float(norm_stats['toa_mean']), 'toa_std': float(norm_stats['toa_std']),
            }
            if norm_stats['global_mean'] is not None:
                save_dict['global_mean'] = norm_stats['global_mean'].numpy()
                save_dict['global_std'] = norm_stats['global_std'].numpy()
            np.savez(stats_path, **save_dict)
            del tmp_dataset
            gc.collect()

    if ddp:
        dist.barrier()

    s = np.load(stats_path)
    norm_stats = {
        'hi_mean': torch.tensor(float(s['hi_mean'])), 'hi_std': torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']), 'stats_std': torch.from_numpy(s['stats_std']),
        'cond_mean': torch.from_numpy(s['cond_mean']), 'cond_std': torch.from_numpy(s['cond_std']),
        'topo_mean': torch.tensor(float(s['topo_mean'])), 'topo_std': torch.tensor(float(s['topo_std'])),
        'toa_mean': torch.tensor(float(s['toa_mean'])), 'toa_std': torch.tensor(float(s['toa_std'])),
    }
    if 'global_mean' in s:
        norm_stats['global_mean'] = torch.from_numpy(s['global_mean'])
        norm_stats['global_std'] = torch.from_numpy(s['global_std'])
    else:
        norm_stats['global_mean'] = None
        norm_stats['global_std'] = None

    train_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                   normalization_stats=norm_stats, shared_data=shared_data)
    val_dataset = ClimateDataset(Config, mode="val", val_indices=val_indices,
                                 normalization_stats=norm_stats, shared_data=shared_data)
    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data)

    if is_main_process() and checkpoint_path is None:
        print("Computing persistence baseline...")
        persistence_metrics = compute_persistence_baseline(val_dataset, conus_mask, n_samples=200)
        print(f"  Persistence baseline: R²={persistence_metrics['r2']:.4f}")

    # Build mesh + model
    mesh = build_mesh_once(Config, conus_mask, device, ddp=ddp)

    model = MeshFlowNet(
        img_channels=Config.IMAGE_CHANNELS,
        spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
        condition_dim=Config.CONDITION_DIM,
        latent_dim=Config.MESH_LATENT_DIM,
        hidden_dim=Config.MESH_LATENT_DIM * 2,
        num_processor_rounds=Config.MESH_PROCESSOR_ROUNDS,
        mesh=mesh,
        image_size=Config.IMAGE_SIZE,
        num_global_channels=Config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
        deterministic=Config.DETERMINISTIC,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    if is_main_process():
        count_parameters(model)
        print(f"  Mode: {'DETERMINISTIC (GraphCast)' if Config.DETERMINISTIC else 'PROBABILISTIC (GenCast/CFM)'}")

    # Sampler + DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=None,
    )

    effective_batch_size = Config.BATCH_SIZE * world_size
    if is_main_process():
        print(f"Per-GPU batch: {Config.BATCH_SIZE}, Effective batch: {effective_batch_size}")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=Config.WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3000, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[Config.WARMUP_EPOCHS])

    start_epoch = 0
    best_ssim = 0.0
    best_r2 = -999.0

    # Checkpoint loading
    if checkpoint_path is not None:
        if is_main_process():
            print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if is_main_process() and missing:
            print(f"  New parameters (not in checkpoint): {len(missing)} keys")
        start_epoch = checkpoint.get("epoch", 0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        best_r2 = checkpoint.get("best_r2", best_r2)

    # DDP wrapping
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema = EMA(model.module if ddp else model, decay=0.999)
    ema.ema.set_mesh(mesh)

    if checkpoint_path is not None and "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        ema_cleaned = {k.replace("module.", ""): v for k, v in ema_state.items()}
        ema.ema.load_state_dict(ema_cleaned, strict=False)
        ema.ema.set_mesh(mesh)
        if is_main_process():
            print("  Restored EMA state from checkpoint")

    if start_epoch > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()

    # Training loop
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_good_batches = 0
        consecutive_skips = 0
        epoch_components = {}   # Diagnostic 1: accumulate components per epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process(),
                    mininterval=10.0)

        for batch_idx, batch in enumerate(pbar):
            (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, mask) = batch

            y = y.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            x_tm1 = x_tm1.to(device, non_blocking=True)
            x_tm2 = x_tm2.to(device, non_blocking=True)
            spatial_c = spatial_c.to(device, non_blocking=True)
            vec_c = vec_c.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, components = compute_loss(
                    model, fm, y, x_t, x_tm1, x_tm2,
                    spatial_c, vec_c, global_fields, mask,
                    deterministic=Config.DETERMINISTIC)

            is_fatal = torch.isnan(loss) or torch.isinf(loss)
            if ddp:
                fatal_flag = torch.tensor(float(is_fatal), device=device)
                dist.all_reduce(fatal_flag, op=dist.ReduceOp.MAX)
                is_fatal = fatal_flag.item() > 0.5

            if is_fatal:
                consecutive_skips += 1
                if is_main_process() and consecutive_skips <= 3:
                    spike_logger.warning(f"NaN/Inf at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # Gradient diagnostic (first batch of first epoch only)
            if epoch == 0 and batch_idx == 0 and is_main_process():
                diag_path = os.path.join(Config.OUTPUT_DIR, "gradient_diagnostic.txt")
                with open(diag_path, "w") as f:
                    f.write("========================================================\n")
                    f.write("Gradient Diagnostic - Epoch 0, Batch 0\n")
                    f.write("========================================================\n")
                    raw = model.module if ddp else model
                    dead_layers = 0
                    for name, p in raw.named_parameters():
                        gn = p.grad.norm().item() if p.grad is not None else 0.0
                        pn = p.norm().item()
                        f.write(f"{name:60s} grad={gn:10.6f}  param={pn:10.4f}\n")
                        if gn == 0.0:
                            dead_layers += 1
                    f.write(f"\nSummary:\n")
                    f.write(f"Total dead layers (grad=0.0): {dead_layers}\n")
                print(f"  Saved gradient diagnostic to: {diag_path}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            ema.update(model.module if ddp else model)
            consecutive_skips = 0

            epoch_loss += loss.item()
            n_good_batches += 1

            # Diagnostic 1: accumulate component losses
            for comp_key, comp_val in components.items():
                if comp_key not in epoch_components:
                    epoch_components[comp_key] = 0.0
                epoch_components[comp_key] += float(comp_val)

            if is_main_process():
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if n_good_batches == 0 and is_main_process():
            print(f"\n  *** ALERT: Epoch {epoch+1} — ALL batches had NaN/Inf loss! ***")

        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            torch.cuda.empty_cache()
            gc.collect()

            if ddp:
                for param in ema.ema.parameters():
                    dist.broadcast(param.data, src=0)

            # Diagnostic 6: run full diagnostics at first checkpoint and every 5th checkpoint
            if is_main_process() and ((epoch + 1) == Config.CHECKPOINT_FREQ or (epoch + 1) % (Config.CHECKPOINT_FREQ * 5) == 0):
                print(f"\n  Running convergence diagnostics (epoch {epoch+1})...")
                diagnose_single_step_generation(ema.ema, val_dataset, device, conus_mask, n_samples=20)
                diagnose_rollout_error_growth(ema.ema, val_dataset, device, conus_mask, n_samples=10)
                if (epoch + 1) == Config.CHECKPOINT_FREQ:
                    diagnose_data_statistics(train_dataset, conus_mask, n_samples=50)

            val_mse, val_rmse, val_ssim, improved_metrics = calculate_validation_metrics_cfm(
                ema.ema, val_dataset, device, conus_mask,
                n_samples=Config.NUM_VALIDATION_SAMPLES,
                rank=rank, world_size=world_size, ddp=ddp
            )

            current_r2 = improved_metrics.get('r2', -999.0) if improved_metrics else -999.0

            if is_main_process():
                val_plot_dir = os.path.join(Config.PLOTS_DIR, "validation")
                save_validation_plots(ema.ema, val_dataset, device, conus_mask, epoch + 1, val_plot_dir, n_samples=10)
                avg_loss = epoch_loss / max(n_good_batches, 1)
                torch.cuda.empty_cache()
                gc.collect()

                # Diagnostic 2: enhanced epoch summary with component breakdown
                print(f"\n  === EPOCH {epoch + 1} METRICS ===")
                print(f"  Training Loss:     {avg_loss:.6f}")
                for ck, cv in epoch_components.items():
                    print(f"    {ck}: {cv / max(n_good_batches, 1):.6f}")
                print(f"  Validation MSE:    {val_mse:.6f}")
                print(f"  Validation SSIM:   {val_ssim:.4f}")
                print(f"  Variance Ratio:    {improved_metrics['variance_ratio']:.4f}")
                print(f"  Gradient Ratio:    {improved_metrics['gradient_ratio']:.4f}")
                print(f"  Extreme Bias:      {improved_metrics['extreme_bias']:.4f}")
                print(f"  Correlation:       {improved_metrics['correlation']:.4f}")
                print(f"  R²:                {improved_metrics['r2']:.4f}")
                print(f"  LR:                {scheduler.get_last_lr()[0]:.2e}")

                val_r2 = improved_metrics["r2"]
                raw_model = model.module if ddp else model
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_ssim": val_ssim,
                    "val_r2": val_r2,
                    "best_ssim": max(best_ssim, val_ssim),
                    "best_r2": max(best_r2, val_r2),
                }
                torch.save(ckpt, os.path.join(Config.CHECKPOINT_DIR,
                                               f"checkpoint_epoch_{epoch+1:04d}.pth"))
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    torch.save(ckpt, Config.MODEL_SAVE_PATH)
                    print(f"  New best model saved (R²={val_r2:.4f})")

        if ddp:
            dist.barrier()

        scheduler.step()
        model.train()

    raw_model = model.module if ddp else model
    if ddp:
        cleanup_ddp()

    return raw_model, ema.ema, fm, train_dataset, test_dataset


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description='CFM Heat Wave Forecasting + Icosahedral Mesh GNN')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'visualize', 'resume'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_size', type=int, default=20)
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic (GraphCast) mode instead of probabilistic (GenCast/CFM)')

    args = parser.parse_args()

    if args.sampling_steps is not None:
        Config.CFM_SAMPLING_STEPS = args.sampling_steps
    if args.ensemble:
        Config.ENSEMBLE_MODE = True
        Config.ENSEMBLE_SIZE = args.ensemble_size
    if args.deterministic:
        Config.DETERMINISTIC = True

    if args.mode in ('train', 'resume'):
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        _train_worker(rank, world_size, args)
    elif args.mode == 'test':
        _test(args)

    if is_main_process():
        print(f"\n{'='*80}")
        print("ALL TASKS COMPLETE")
        print(f"{'='*80}")


def _train_worker(rank, world_size, args):
    model, ema_model, fm, train_dataset, test_dataset = train_model(
        rank=rank, world_size=world_size,
        checkpoint_path=args.checkpoint if args.mode == 'resume' else None
    )


def _test(args):
    device = torch.device("cuda:0")
    apply_subdomain_config()
    conus_mask = load_conus_mask(Config)

    mesh = build_mesh_once(Config, conus_mask, device, ddp=False)

    model = MeshFlowNet(
        img_channels=Config.IMAGE_CHANNELS,
        spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
        condition_dim=Config.CONDITION_DIM,
        latent_dim=Config.MESH_LATENT_DIM,
        hidden_dim=Config.MESH_LATENT_DIM * 2,
        num_processor_rounds=Config.MESH_PROCESSOR_ROUNDS,
        mesh=mesh,
        image_size=Config.IMAGE_SIZE,
        num_global_channels=Config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
        deterministic=Config.DETERMINISTIC,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint['model_state_dict'])
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    shared_data = prepare_shared_data(Config, rank=0, world_size=1, ddp=False)
    n_timesteps = shared_data['heat_index'].shape[-1]

    time_values = np.array(shared_data['time_values'])

    runs = detect_continuous_runs(time_values)
    train_valid = build_valid_indices(runs, lead_time=Config.LEAD_TIME, min_history=2)
    rollout_valid = build_valid_indices(runs, lead_time=Config.ROLLOUT_STEPS, min_history=2)
    rollout_set = set(rollout_valid)

    total_train = len(train_valid)
    test_size = max(1, int(total_train * Config.TEST_FRACTION))
    val_size  = max(1, int(total_train * Config.VAL_FRACTION))
    train_end = total_train - test_size - val_size

    train_indices = train_valid[:train_end]
    test_candidates = train_valid[train_end + val_size:]
    test_indices = [i for i in test_candidates if i in rollout_set]

    # Normalization stats (subdomain-aware path)
    sub_tag = ""
    if Config.SUBDOMAIN_ENABLED:
        la0, la1 = Config.SUBDOMAIN_LAT_RANGE
        lo0, lo1 = Config.SUBDOMAIN_LON_RANGE
        sub_tag = f"_sub{la0:.0f}_{la1:.0f}_{lo0:.0f}_{lo1:.0f}"
    stats_path = os.path.join(Config.OUTPUT_DIR, "data_cache", f"norm_stats_v2{sub_tag}.npz")
    
    s = np.load(stats_path)
    norm_stats = {
        'hi_mean': torch.tensor(float(s['hi_mean'])), 'hi_std': torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']), 'stats_std': torch.from_numpy(s['stats_std']),
        'cond_mean': torch.from_numpy(s['cond_mean']), 'cond_std': torch.from_numpy(s['cond_std']),
        'topo_mean': torch.tensor(float(s['topo_mean'])), 'topo_std': torch.tensor(float(s['topo_std'])),
        'toa_mean': torch.tensor(float(s['toa_mean'])), 'toa_std': torch.tensor(float(s['toa_std'])),
    }
    if 'global_mean' in s:
        norm_stats['global_mean'] = torch.from_numpy(s['global_mean'])
        norm_stats['global_std'] = torch.from_numpy(s['global_std'])
    else:
        norm_stats['global_mean'] = None
        norm_stats['global_std'] = None

    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data)

    h, w = Config.IMAGE_SIZE
    mask_4d = conus_mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    all_preds, all_truth = [], []

    for dataset_idx in tqdm(range(len(test_dataset)), desc="Test (15-day rollout)"):
        t_start = test_dataset.indices[dataset_idx]

        pred = generate_autoregressive_rollout(
            model, test_dataset, t_start, device, h, w, mask_4d,
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS,
            rollout_steps=Config.ROLLOUT_STEPS,
        )
        all_preds.append(torch.from_numpy(pred))

        gt_idx = t_start + Config.ROLLOUT_STEPS
        gt_raw = test_dataset.heat_index[:, :, gt_idx]
        gt_tensor = torch.from_numpy(gt_raw.copy())
        gt_land = (torch.isfinite(gt_tensor) & (gt_tensor != 0.0)).float()
        gt_clean = torch.nan_to_num(gt_tensor, nan=0.0)
        gt_normed = (gt_clean - test_dataset.hi_mean) / (test_dataset.hi_std + 1e-8)
        gt_normed = gt_normed * gt_land + Config.OCEAN_FILL * (1 - gt_land)
        all_truth.append(gt_normed[:h, :w])

    predictions = torch.stack(all_preds).unsqueeze(1)
    ground_truth = torch.stack(all_truth).unsqueeze(1)

    mask_2d = conus_mask[:h, :w].cpu()
    metrics = calculate_improved_metrics(predictions, ground_truth, mask=mask_2d)

    print(f"\n{'='*80}")
    print("TEST SET METRICS (15-day autoregressive rollout)")
    print(f"{'='*80}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()