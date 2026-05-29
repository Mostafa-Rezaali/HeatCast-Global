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
- Replaces complex VDM gamma schedules with uniform time t in [0, 1].
- Uses deterministic Euler integration for sampling.
- NEW: Icosahedral mesh GNN replaces U-Net backbone.
  Grid -> Mesh -> 8 rounds message passing -> Mesh -> Grid
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

# ===========================================================================
# CHANGE 1: Import mesh modules
# ===========================================================================
from icosahedral_mesh import IcosahedralMesh
from mesh_backbone import MeshFlowNet, count_parameters
from mode_dispatch import compute_loss, generate_sample
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
# PADDING UTILITIES
# ======================================================================================
def calculate_padding(height, width, window_size):
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    return pad_h, pad_w

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
        'u_wind_200',
        'total_column_water_vapour',
        'geopotential_200',
        'olr',
        'sst',
    ]
    GLOBAL_SIZE = (181, 360)
    NUM_GLOBAL_CHANNELS = 5

    # ==================== MODEL ARCHITECTURE ====================
    IMAGE_SIZE = (621, 1405)
    IMAGE_CHANNELS = 1
    # Input: [flow(1), x_t(1), x_tm1(1), x_tm2(1), physics(9), topo(1), lat(1), lon(1)] = 16 total
    # But spatial_cond = x_t(1) + x_tm1(1) + x_tm2(1) + physics(9) + topo(1) + lat(1) + lon(1) = 15
    NUM_SPATIAL_CONDITIONS = 15
    LEAD_TIME = 15
    CONDITION_DIM = 5

    BASE_DIM = 64
    DIM_MULTS = (1, 2, 4, 8)  # kept for reference, not used by mesh
    DROPOUT_RATE = 0.3

    # Global encoder config
    GLOBAL_ENCODER_DIM = 64
    GLOBAL_BOTTLENECK_DIM = 1024  # not used by mesh

    BATCH_SIZE = 4
    OCEAN_FILL = 0

    # ==================== CFM SCHEDULE ====================
    CFM_SAMPLING_STEPS = 50

    # ==================== TRAINING HYPERPARAMETERS ====================
    DEVICE = "cuda"
    LEARNING_RATE = 3e-4
    GRAD_CLIP_NORM = 1.0
    WINDOW_SIZE = 64

    # ==================== GENERATION SETTINGS ====================
    ENSEMBLE_SIZE = 20
    ENSEMBLE_MODE = False

    # ==================== TRAINING SCHEDULE ====================
    MAX_EPOCHS = 20000
    TEST_FRACTION = 0.15
    VAL_FRACTION = 0.15
    CHECKPOINT_FREQ = 50
    NUM_VALIDATION_SAMPLES = 50
    WARMUP_EPOCHS = 50

    # ===========================================================================
    # CHANGE 2: Mesh-specific config
    # ===========================================================================
    MESH_REFINEMENT_LEVEL = 6   # 6 -> ~2,144 regional nodes
    MESH_PROCESSOR_ROUNDS = 8   # message passing rounds
    MESH_LATENT_DIM = 256       # hidden dim for all GNN layers
    MESH_BUFFER_DEG = 5.0       # extra degrees around CONUS for mesh nodes
    K_GRID2MESH = 3             # neighbors per grid point in encoder
    K_MESH2GRID = 3             # neighbors per grid point in decoder
    CONUS_LAT_RANGE = (25.0, 50.0)
    CONUS_LON_RANGE = (-130.0, -60.0)

    # ===========================================================================
    # Mode: deterministic (GraphCast) vs probabilistic (GenCast/CFM)
    # ===========================================================================
    DETERMINISTIC = False  # False = GenCast/CFM, True = GraphCast


if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    mode_name = "DETERMINISTIC (GraphCast)" if Config.DETERMINISTIC else "PROBABILISTIC (GenCast/CFM)"
    print(f"\n{'='*80}")
    print(f"CFM TRAINING - {mode_name} + ICOSAHEDRAL MESH GNN")
    print(f"{'='*80}")
    print(f"Using device: {Config.DEVICE}")
    print(f"Mode: {mode_name}")
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


# ===========================================================================
# CHANGE 3: Build mesh once, cache to disk
# ===========================================================================
def build_mesh_once(config, conus_mask, device, ddp=False):
    """
    Build the icosahedral mesh. Expensive (~5-30 seconds depending on level)
    but only happens once. Cached to disk for subsequent runs.
    """
    cache_path = os.path.join(config.OUTPUT_DIR, "data_cache",
                               f"mesh_level{config.MESH_REFINEMENT_LEVEL}.pkl")

    if not ddp or dist.get_rank() == 0:
        if os.path.exists(cache_path):
            print(f"Loading cached mesh from {cache_path}")
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            # FIX 3: Downsample dimensions and mask for the mesh builder
            downsample_factor = 4
            H_down = config.IMAGE_SIZE[0] // downsample_factor
            W_down = config.IMAGE_SIZE[1] // downsample_factor
            
            grid_lat = np.linspace(
                config.CONUS_LAT_RANGE[0],
                config.CONUS_LAT_RANGE[1],
                H_down  # <-- Use downsampled dimension
            )
            grid_lon = np.linspace(
                config.CONUS_LON_RANGE[0],
                config.CONUS_LON_RANGE[1],
                W_down  # <-- Use downsampled dimension
            )

            # Downsample the mask. Average pooling > 0.5 keeps strictly major landmasses
            mask_tensor = conus_mask.float()
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_down = F.avg_pool2d(mask_tensor, kernel_size=downsample_factor).squeeze().cpu().numpy() > 0.5

            mesh = IcosahedralMesh(
                refinement_level=config.MESH_REFINEMENT_LEVEL,
                lat_range=config.CONUS_LAT_RANGE,
                lon_range=config.CONUS_LON_RANGE,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                land_mask=mask_down, # <-- Pass the downsampled mask
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
# IMPROVED METRICS (UNCHANGED)
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
# DATASET (UNCHANGED)
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
                raw_shape = nc.variables['HeatIndex'].shape
                print(f"  Raw dataset shape: {raw_shape}")
                hi_raw = nc.variables['HeatIndex'][:]
                if hi_raw.ndim == 4:
                    self.heat_index = np.array(hi_raw[:, :, 0, :], dtype=np.float32)
                elif hi_raw.ndim == 3:
                    self.heat_index = np.array(hi_raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unexpected HeatIndex shape: {raw_shape}")
                self.n_timesteps = self.heat_index.shape[-1]
                self.geopotential = np.array(nc.variables['geopotential'][:], dtype=np.float32)
                self.soil_moisture = np.array(nc.variables['soil_moisture'][:], dtype=np.float32)
                self.slp = np.array(nc.variables['sea_level_pressure'][:], dtype=np.float32)
                self.cond_train = np.array(nc.variables['CondTrain'][:], dtype=np.float32)
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
            self.n_timesteps = self.heat_index.shape[-1]

        if normalization_stats is None and mode == 'train':
            print("  Calculating Z-Score statistics (TRAINING SET ONLY)...")
            if train_indices is None:
                train_indices = list(range(1, self.n_timesteps - self.config.LEAD_TIME))

            train_t_indices = np.array(train_indices)
            hi_train = self.heat_index[:, :, train_t_indices]
            hi_mask = hi_train != 0.0
            self.hi_mean = torch.tensor(np.mean(hi_train[hi_mask]), dtype=torch.float32)
            self.hi_std = torch.tensor(np.std(hi_train[hi_mask]), dtype=torch.float32)
            print(f"    HeatIndex -> Mean: {self.hi_mean:.4f}, Std: {self.hi_std:.4f}")

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
                [np.mean(gp_train), np.mean(sm_train), np.mean(slp_train),
                 np.mean(t2m_train), np.mean(q850_train), np.mean(t850_train),
                 np.mean(u850_train), np.mean(v850_train), np.mean(z300_train)],
                dtype=torch.float32
            ).view(9, 1, 1)
            self.stats_std = torch.tensor(
                [np.std(gp_train), np.std(sm_train), np.std(slp_train),
                 np.std(t2m_train), np.std(q850_train), np.std(t850_train),
                 np.std(u850_train), np.std(v850_train), np.std(z300_train)],
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
                    global_means.append(float(np.mean(var_train)))
                    global_stds.append(float(np.std(var_train)))
                self.global_mean = torch.tensor(global_means, dtype=torch.float32).view(-1, 1, 1)
                self.global_std = torch.tensor(global_stds, dtype=torch.float32).view(-1, 1, 1)
                del var_train
            else:
                self.global_mean = None
                self.global_std = None

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
        else:
            raise ValueError(f"normalization_stats required for mode='{mode}'")

        h, w = self.config.IMAGE_SIZE
        lat_1d = np.linspace(-1.0, 1.0, h)
        lon_1d = np.linspace(-1.0, 1.0, w)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        self.lat_grid = torch.from_numpy(lat_grid).float().unsqueeze(0)
        self.lon_grid = torch.from_numpy(lon_grid).float().unsqueeze(0)
        pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)
        self.padding = (0, pad_w, 0, pad_h)

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

        x_t_slice      = self.heat_index[:, :, t]
        x_tm1_slice    = self.heat_index[:, :, t - 1]
        x_tm2_slice    = self.heat_index[:, :, t - 2]
        x_target_slice = self.heat_index[:, :, t + self.config.LEAD_TIME]

        raw_x_t = torch.from_numpy(x_t_slice.copy())
        land_mask = (raw_x_t != 0.0).float().unsqueeze(0)

        def normalize_hi(data_slice):
            tensor = torch.from_numpy(data_slice.copy())
            normed = (tensor - self.hi_mean) / (self.hi_std + 1e-8)
            normed = normed.unsqueeze(0)
            return normed * land_mask + Config.OCEAN_FILL * (1 - land_mask)

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
        physics = (physics - self.stats_mean) / (self.stats_std + 1e-8)
        physics = physics * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        topo = torch.from_numpy(self.topography.copy())
        if topo.shape != (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]):
            topo = topo.T
        topo = ((topo - self.topo_mean) / (self.topo_std + 1e-8)).unsqueeze(0)
        topo = topo * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        lat_c = self.lat_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        lon_c = self.lon_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        physics = torch.cat([physics, topo, lat_c, lon_c], dim=0)

        if self.global_data is not None and self.global_mean is not None:
            global_channels = []
            for var_name, var_data in self.global_data.items():
                g_slice = torch.from_numpy(np.array(var_data[:, :, t], dtype=np.float32))
                global_channels.append(g_slice)
            global_fields = torch.stack(global_channels, dim=0)
            global_fields = (global_fields - self.global_mean) / (self.global_std + 1e-8)
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        y     = F.pad(y,     self.padding, "constant", Config.OCEAN_FILL)
        x_t   = F.pad(x_t,   self.padding, "constant", Config.OCEAN_FILL)
        x_tm1 = F.pad(x_tm1, self.padding, "constant", Config.OCEAN_FILL)
        x_tm2 = F.pad(x_tm2, self.padding, "constant", Config.OCEAN_FILL)
        physics = F.pad(physics, self.padding, "constant", Config.OCEAN_FILL)
        mask    = F.pad(land_mask, self.padding, "constant", 0)

        return y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, t, mask


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
        },
    }
    if dataset.global_data is not None:
        stats['shared_data']['global_data'] = dataset.global_data
    stats['global_mean'] = dataset.global_mean
    stats['global_std'] = dataset.global_std
    return stats

# ======================================================================================
# CFM LOSS (UNCHANGED)
# ======================================================================================
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

def masked_ssim_01(pred, target, mask, fill=0.5):
    mask = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    pred_m = pred * mask + fill * (1 - mask)
    targ_m = target * mask + fill * (1 - mask)
    return structural_similarity_index_measure(pred_m, targ_m, data_range=1.0)

# ======================================================================================
# OCEAN MASKING UTILITIES (UNCHANGED)
# ======================================================================================
def load_conus_mask(config):
    with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
        hi = nc.variables['HeatIndex']
        if hi.ndim == 4:
            hi_sample = hi[:, :, 0, 0]
        elif hi.ndim == 3:
            hi_sample = hi[:, :, 0]
        else:
            raise ValueError(f"Unexpected HeatIndex ndim: {hi.ndim}")
        mask = (np.abs(hi_sample) > 0.01).astype(np.float32)
    land_fraction = mask.sum() / mask.size
    print(f"  Mask loaded: {land_fraction*100:.1f}% land, {(1-land_fraction)*100:.1f}% ocean")
    return torch.from_numpy(mask)

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
# VALIDATION
# ======================================================================================
@torch.inference_mode()
def calculate_validation_metrics_cfm(model, dataset, device, mask,
                                      n_samples=100, rank=0, world_size=1, ddp=False):
    import time
    validation_start = time.time()

    model.eval()
    if len(dataset) == 0:
        return 0.0, 0.0, 0.0, {}

    per_rank = max(1, n_samples // world_size)
    actual_n_samples = per_rank * world_size

    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, len(dataset))
    rank_indices = list(range(start_idx, end_idx))

    rank_subset = torch.utils.data.Subset(dataset, rank_indices)
    val_loader = DataLoader(rank_subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    if is_main_process():
        print(f"\n  Generating {actual_n_samples} samples across {world_size} GPUs "
              f"({per_rank} per GPU...")

    all_preds = []
    all_truth = []
    all_masks = []

    h, w = Config.IMAGE_SIZE
    pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)

    for batch in val_loader:
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, batch_mask) = batch
        y         = y.to(device)
        x_t       = x_t.to(device)
        x_tm1     = x_tm1.to(device)
        x_tm2     = x_tm2.to(device)
        spatial_c = spatial_c.to(device)
        vec_c     = vec_c.to(device)
        global_fields = global_fields.to(device)

        x_tp1_true = y[0, 0, :h, :w].detach().cpu()

        if Config.DETERMINISTIC:
            # Single forward pass, no ensemble needed
            pred = generate_sample(
                model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                global_fields, device, h, w, pad_h, pad_w,
                mask=batch_mask.to(device),
                deterministic=True,
            )
            all_preds.append(torch.from_numpy(pred))
        else:
            # Probabilistic: small ensemble for stable R² estimates
            val_ensemble_size = 4
            member_preds = []
            for _ in range(val_ensemble_size):
                pred = generate_sample(
                    model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                    global_fields, device, h, w, pad_h, pad_w,
                    mask=batch_mask.to(device),
                    deterministic=False,
                    n_steps=Config.CFM_SAMPLING_STEPS,
                )
                member_preds.append(pred)
            ensemble_mean_pred = np.mean(member_preds, axis=0)
            all_preds.append(torch.from_numpy(ensemble_mean_pred))
        all_truth.append(x_tp1_true)
        all_masks.append(batch_mask[0, 0, :h, :w])

    local_preds = torch.stack(all_preds)
    local_truth = torch.stack(all_truth)
    local_masks = torch.stack(all_masks)

    if ddp:
        local_preds = local_preds.to(device)
        local_truth = local_truth.to(device)
        local_masks = local_masks.to(device)

        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_truth = [torch.zeros_like(local_truth) for _ in range(world_size)]
        gathered_masks = [torch.zeros_like(local_masks) for _ in range(world_size)]

        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_truth, local_truth)
        dist.all_gather(gathered_masks, local_masks)

        all_preds = torch.cat(gathered_preds, dim=0).cpu()
        all_truth = torch.cat(gathered_truth, dim=0).cpu()
        all_masks = torch.cat(gathered_masks, dim=0).cpu()
    else:
        all_preds = local_preds.cpu()
        all_truth = local_truth.cpu()
        all_masks = local_masks.cpu()

    if not is_main_process():
        return 0.0, 0.0, 0.0, {}

    all_preds = all_preds.unsqueeze(1)
    all_truth = all_truth.unsqueeze(1)
    all_masks = all_masks.unsqueeze(1)

    preds_np = all_preds.numpy()
    truth_np = all_truth.numpy()
    m = all_masks.numpy()

    err2 = (preds_np - truth_np) ** 2
    mse_per_sample = (err2 * m).sum(axis=(1,2,3)) / (m.sum(axis=(1,2,3)) + 1e-8)
    avg_mse  = float(mse_per_sample.mean())
    avg_rmse = float(np.sqrt(avg_mse))

    def to_ssim_space(x):
        return ((x + 3.0) / 6.0).clamp(0.0, 1.0)

    preds_ssim = to_ssim_space(all_preds).to(device)
    truth_ssim = to_ssim_space(all_truth).to(device)
    avg_ssim = masked_ssim_01(preds_ssim, truth_ssim, all_masks.to(device), fill=0.5).item()

    improved = calculate_improved_metrics(all_preds, all_truth, mask=all_masks)

    validation_total = time.time() - validation_start
    print(f"\n  Validation time: {validation_total:.2f}s "
          f"({actual_n_samples/validation_total:.1f} samples/sec across {world_size} GPUs)")
    print(f"  Metrics: MSE={avg_mse:.6f}, SSIM={avg_ssim:.4f}, R²={improved['r2']:.4f}")

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
# VISUALIZATION (UNCHANGED - abbreviated for brevity, copy from your original)
# ======================================================================================
def save_validation_plots(model, val_dataset, device, mask, epoch, save_dir, n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    h, w = Config.IMAGE_SIZE
    pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    collected = 0
    for batch in loader:
        if collected >= n_samples:
            break
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, batch_mask) = batch
        y = y.to(device)
        x_t = x_t.to(device)
        x_tm1 = x_tm1.to(device)
        x_tm2 = x_tm2.to(device)
        spatial_c = spatial_c.to(device)
        vec_c = vec_c.to(device)
        global_fields = global_fields.to(device)

        pred = generate_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, pad_h, pad_w,
            mask=batch_mask.to(device),
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS
        )

        truth = y[0, 0, :h, :w].cpu().numpy()
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
        axes[1].set_title(f'Prediction\nMAE={mae:.3f}, r={corr:.3f}, R²={r2:.3f}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Z-score')
        diff_display = np.where(mask_np[:h, :w] > 0.5, pred - truth, np.nan)
        im = axes[2].imshow(diff_display, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
        axes[2].set_title('Prediction - Truth', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[2], label='Difference (Z-score)')
        fig.suptitle(f'Epoch {epoch} — Validation Sample {collected + 1}', fontsize=15, fontweight='bold')
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
# DATA CACHE (UNCHANGED)
# ======================================================================================
def prepare_shared_data(config, rank, world_size, ddp):
    cache_dir = os.path.join(config.OUTPUT_DIR, "data_cache")
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
            with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc:
                hi_raw = nc.variables["HeatIndex"][:]
                if hi_raw.ndim == 4:
                    hi = np.array(hi_raw[:, :, 0, :], dtype=np.float32)
                else:
                    hi = np.array(hi_raw, dtype=np.float32)
                np.save(paths["heat_index"], hi)
                np.save(paths["geopotential"], np.array(nc.variables["geopotential"][:], dtype=np.float32))
                np.save(paths["soil_moisture"], np.array(nc.variables["soil_moisture"][:], dtype=np.float32))
                np.save(paths["slp"], np.array(nc.variables["sea_level_pressure"][:], dtype=np.float32))
                np.save(paths["cond_train"], np.array(nc.variables["CondTrain"][:], dtype=np.float32))
                np.save(paths["temperature_2m"], np.array(nc.variables["temperature_2m"][:], dtype=np.float32))
                np.save(paths["specific_humidity_850"], np.array(nc.variables["specific_humidity_850"][:], dtype=np.float32))
                np.save(paths["temperature_850"], np.array(nc.variables["temperature_850"][:], dtype=np.float32))
                np.save(paths["u_wind_850"], np.array(nc.variables["u_wind_850"][:], dtype=np.float32))
                np.save(paths["v_wind_850"], np.array(nc.variables["v_wind_850"][:], dtype=np.float32))
                np.save(paths["geopotential_300"], np.array(nc.variables["geopotential_300"][:], dtype=np.float32))

            with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
                topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
                topo = np.flipud(topo)
            np.save(paths["topography"], topo)
            print(f"Rank 0: wrote CONUS data cache to {cache_dir}")
        else:
            print(f"Rank 0: using existing CONUS data cache in {cache_dir}")

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

    # --- NEW: rank 0 copies to /dev/shm, all ranks mmap from there ---
    shm_dir = "/dev/shm/cfm_cache"

    if not ddp or dist.get_rank() == 0:
        os.makedirs(shm_dir, exist_ok=True)
        for key, src_path in paths.items():
            dst = os.path.join(shm_dir, os.path.basename(src_path))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        # Also copy global files
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

    conus_mask = load_conus_mask(Config)
    if is_main_process():
        save_mask_plot(conus_mask, Config.PLOTS_DIR)
    conus_mask = conus_mask.to(device)

    fm = FlowMatching().to(device)

    if is_main_process():
        print("\n" + "=" * 80)
        print("CONDITIONAL FLOW MATCHING (CFM) - DDP + ICOSAHEDRAL MESH GNN")
        print(f"World size: {world_size}")
        print(f"Mesh level: {Config.MESH_REFINEMENT_LEVEL}, Rounds: {Config.MESH_PROCESSOR_ROUNDS}")
        print("=" * 80 + "\n")

    # Data
    shared_data = prepare_shared_data(Config, rank, world_size, ddp)
    n_timesteps = shared_data['heat_index'].shape[-1]

    all_indices = list(range(1, n_timesteps - Config.LEAD_TIME))
    total_len = len(all_indices)
    test_size = max(1, int(total_len * Config.TEST_FRACTION))
    val_size  = max(1, int(total_len * Config.VAL_FRACTION))

    test_indices  = all_indices[-test_size:]
    val_end_idx   = len(all_indices) - test_size - Config.LEAD_TIME
    val_indices   = all_indices[val_end_idx - val_size : val_end_idx]
    train_end     = val_end_idx - val_size - Config.LEAD_TIME
    train_indices = all_indices[:train_end]

    if is_main_process():
        print(f"\nStrict Temporal Split (Lead Time = {Config.LEAD_TIME} days):")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples\n")

    # Normalization stats
    stats_path = os.path.join(Config.OUTPUT_DIR, "data_cache", "norm_stats_v2.npz")

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

    # ===========================================================================
    # CHANGE 4: Build mesh + create MeshFlowNet (replaces FlowUNet)
    # ===========================================================================
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

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.001)

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

    # ===========================================================================
    # CHANGE 5: Checkpoint loading - reattach mesh after load
    # ===========================================================================
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

    # ===========================================================================
    # CHANGE 6: Reattach mesh to EMA model (mesh is not in state_dict)
    # ===========================================================================
    ema.ema.set_mesh(mesh)

    if checkpoint_path is not None and "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        ema_cleaned = {k.replace("module.", ""): v for k, v in ema_state.items()}
        ema.ema.load_state_dict(ema_cleaned, strict=False)
        # Re-attach mesh after loading EMA state (mesh is not serialized)
        ema.ema.set_mesh(mesh)
        if is_main_process():
            print("  Restored EMA state from checkpoint")

    # Sync scheduler on resume
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

            # ============ DIAGNOSTIC: Gradient flow per component ============
            if is_main_process() and batch_idx == 0 and (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
                raw = model.module if ddp else model
                grad_norms = {}
                for comp_name in ['encoder', 'processor', 'decoder', 'global_encoder', 'time_mlp', 'skip_proj']:
                    comp = getattr(raw, comp_name, None)
                    if comp is None:
                        continue
                    total_norm = 0.0
                    n_params = 0
                    zero_count = 0
                    for p in comp.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                            n_params += 1
                            if p.grad.data.abs().max().item() < 1e-10:
                                zero_count += 1
                        else:
                            zero_count += 1
                            n_params += 1
                    grad_norms[comp_name] = (total_norm ** 0.5, n_params, zero_count)
                print(f"\n  === GRADIENT DIAGNOSTICS (Epoch {epoch+1}, Batch 0) ===")
                for comp_name, (gnorm, np_, zc) in grad_norms.items():
                    print(f"    {comp_name:20s} grad_norm={gnorm:.6f}  params={np_}  zero_grad={zc}")
            # ============ END DIAGNOSTIC ============

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            ema.update(model.module if ddp else model)
            consecutive_skips = 0

            epoch_loss += loss.item()
            n_good_batches += 1

            if is_main_process():
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if n_good_batches == 0 and is_main_process():
            print(f"\n  *** ALERT: Epoch {epoch+1} — ALL batches had NaN/Inf loss! ***")

        # ============ DIAGNOSTIC: per-epoch loss summary ============
        if is_main_process() and n_good_batches > 0:
            avg_ep_loss = epoch_loss / n_good_batches
            if epoch < 5 or (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} avg_loss={avg_ep_loss:.6f}  "
                      f"batches={n_good_batches}  LR={scheduler.get_last_lr()[0]:.2e}")
        # ============ END DIAGNOSTIC ============

        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            torch.cuda.empty_cache()
            gc.collect()

            # ============ DIAGNOSTIC: EMA vs training model weight comparison ============
            if is_main_process():
                raw = model.module if ddp else model
                ema_diff = 0.0
                ema_total = 0.0
                n_params = 0
                for (n1, p1), (n2, p2) in zip(raw.named_parameters(), ema.ema.named_parameters()):
                    if p1.dtype.is_floating_point:
                        ema_diff += (p1.data - p2.data).abs().mean().item()
                        ema_total += p1.data.abs().mean().item()
                        n_params += 1
                avg_diff = ema_diff / max(n_params, 1)
                avg_mag = ema_total / max(n_params, 1)
                print(f"\n  [DIAG] EMA vs Training: avg_param_diff={avg_diff:.6f}  "
                      f"avg_param_mag={avg_mag:.6f}  ratio={avg_diff/(avg_mag+1e-8):.6f}")
            # ============ END DIAGNOSTIC ============

            if ddp:
                for param in ema.ema.parameters():
                    dist.broadcast(param.data, src=0)

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

                print(f"\n  === EPOCH {epoch + 1} METRICS ===")
                print(f"  Training Loss:     {avg_loss:.6f}")
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

    if rank == 0:
        conus_mask = load_conus_mask(Config)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        # Test predictions use generate_sample dispatch
        from torch.utils.data import DataLoader
        h, w = Config.IMAGE_SIZE
        pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        all_preds, all_truth = [], []

        for batch in tqdm(test_loader, desc="Test predictions"):
            (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, batch_mask) = batch
            y = y.to(device); x_t = x_t.to(device); x_tm1 = x_tm1.to(device)
            x_tm2 = x_tm2.to(device); spatial_c = spatial_c.to(device)
            vec_c = vec_c.to(device); global_fields = global_fields.to(device)

            pred = generate_sample(
                ema_model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                global_fields, device, h, w, pad_h, pad_w,
                mask=batch_mask.to(device),
                deterministic=Config.DETERMINISTIC,
                n_steps=Config.CFM_SAMPLING_STEPS
            )
            all_preds.append(torch.from_numpy(pred))
            all_truth.append(y[0, 0, :h, :w].cpu())

        predictions = torch.stack(all_preds)
        ground_truth = torch.stack(all_truth)

        metrics = calculate_improved_metrics(predictions, ground_truth, mask=conus_mask.cpu())
        print(f"\nTest R²={metrics['r2']:.4f}, Corr={metrics['correlation']:.4f}")


def _test(args):
    device = torch.device("cuda:0")
    conus_mask = load_conus_mask(Config)

    # Build mesh first
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
    ).to(device)

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint['model_state_dict'])
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    # Mesh is already attached from constructor

    shared_data = prepare_shared_data(Config, rank=0, world_size=1, ddp=False)
    n_timesteps = shared_data['heat_index'].shape[-1]
    all_indices = list(range(1, n_timesteps - Config.LEAD_TIME))
    total_len = len(all_indices)
    test_size = max(1, int(total_len * Config.TEST_FRACTION))
    val_size  = max(1, int(total_len * Config.VAL_FRACTION))
    test_indices  = all_indices[-test_size:]
    val_end_idx   = len(all_indices) - test_size - Config.LEAD_TIME
    val_indices   = all_indices[val_end_idx - val_size : val_end_idx]
    train_end     = val_end_idx - val_size - Config.LEAD_TIME
    train_indices = all_indices[:train_end]

    train_dataset = ClimateDataset(Config, mode='train', train_indices=train_indices, shared_data=shared_data)
    norm_stats = get_normalization_stats(train_dataset)
    test_dataset = ClimateDataset(Config, mode='test', test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=norm_stats['shared_data'])

    h, w = Config.IMAGE_SIZE
    pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    all_preds, all_truth = [], []

    model.eval()
    for batch in tqdm(test_loader, desc="Test"):
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, batch_mask) = batch
        y = y.to(device); x_t = x_t.to(device); x_tm1 = x_tm1.to(device)
        x_tm2 = x_tm2.to(device); spatial_c = spatial_c.to(device)
        vec_c = vec_c.to(device); global_fields = global_fields.to(device)

        pred = generate_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, pad_h, pad_w,
            mask=batch_mask.to(device),
            deterministic=Config.DETERMINISTIC,
            n_steps=Config.CFM_SAMPLING_STEPS
        )
        all_preds.append(torch.from_numpy(pred))
        all_truth.append(y[0, 0, :h, :w].cpu())

    predictions = torch.stack(all_preds)
    ground_truth = torch.stack(all_truth)
    metrics = calculate_improved_metrics(predictions, ground_truth, mask=conus_mask.cpu())

    print(f"\n{'='*80}")
    print("TEST SET METRICS")
    print(f"{'='*80}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()