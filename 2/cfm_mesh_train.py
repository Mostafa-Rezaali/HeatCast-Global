#!/usr/bin/env python3
"""
================================================================================
MeshFlowNet - DIRECT 15-DAY PREDICTION (No Autoregressive Rollout)
================================================================================
Physics-Guided Heat Wave Forecasting with ICOSAHEDRAL MESH GNN BACKBONE
(GraphCast-style Encoder-Processor-Decoder) + GLOBAL CONTEXT ENCODER

Key changes from rollout version:
- LEAD_TIME=15: model directly predicts day-15 field in one forward pass
- ROLLOUT_STEPS=1: no autoregressive rollout at inference
- Leave-k-years-out cross-validation instead of forward temporal split
- Exponential extreme loss option for transfer learning fine-tuning
- Validation uses single forward pass (15x faster than rollout)

Architecture unchanged: MeshFlowNet GNN backbone with per-round FiLM.
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
import ast
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


def _safe_name(value):
    text = str(value).strip().replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def apply_run_name(run_name):
    run_name = _safe_name(run_name) if run_name else ""
    Config.RUN_NAME = run_name
    if run_name:
        Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints", run_name)
        Config.PLOTS_DIR = os.path.join(Config.OUTPUT_DIR, "test_prediction_plots", run_name)
        Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, f"trained_cfm_direct15_{run_name}.pth")
        Config.TAC_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, f"trained_cfm_direct15_{run_name}_best_tac.pth")
    else:
        Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints")
        Config.PLOTS_DIR = os.path.join(Config.OUTPUT_DIR, "test_prediction_plots")
        Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "trained_cfm_direct15.pth")
        Config.TAC_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "trained_cfm_direct15_best_tac.pth")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.HINDCAST_STATS_DIR, exist_ok=True)


def _format_offsets(offsets):
    return "-".join(str(int(x)) for x in offsets) if offsets else "none"


def parse_cv_offsets(value, stride):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        parts = value
    else:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
    offsets = sorted({int(p) % int(stride) for p in parts})
    return tuple(offsets)


def cv_split_tag(config=None):
    if config is None:
        config = Config
    return (
        f"cv{config.CV_STRIDE}"
        f"_val{_format_offsets(config.CV_VAL_OFFSETS)}"
        f"_test{_format_offsets(config.CV_TEST_OFFSETS)}"
    )


def get_norm_stats_path(config=None):
    if config is None:
        config = Config
    return os.path.join(
        config.OUTPUT_DIR,
        "data_cache",
        f"norm_stats_direct15_{cv_split_tag(config)}.npz",
    )


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
    TARGET_VARIABLE_CANDIDATES = ("t2m_prism", "HeatIndex")
    OUTPUT_DIR = "/blue/nessie/mostafarezaali/Teleconnection/"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "test_prediction_plots")

    OUTPUT_NC_FILE = os.path.join(OUTPUT_DIR, "CFM_Forecasts_Improved.nc")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_direct15.pth")
    TAC_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_direct15_best_tac.pth")

    # ==================== GLOBAL DATA PATHS ====================
    GLOBAL_DATA_PATH = '/blue/nessie/mostafarezaali/Teleconnection/Global_Coarse_Conditions_Extended.nc'
    EXTENDED_GLOBAL_VARIABLES_PATH = os.path.join(
        OUTPUT_DIR, "data_cache", "extended_global_variables.txt"
    )
    USE_EXTENDED_GLOBAL_FIELDS = True
    REQUIRE_EXTENDED_GLOBAL_FIELDS = True
    EXPECTED_NUM_GLOBAL_CHANNELS = 59
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
    GLOBAL_LAG_DAYS = (0, 3, 7, 14)
    NUM_GLOBAL_CHANNELS = EXPECTED_NUM_GLOBAL_CHANNELS * len(GLOBAL_LAG_DAYS)

    # Train on local, train-year-only daily climatological anomalies.
    # The full field is restored for R2/plots/export by adding this climatology back.
    TRAIN_ON_CLIMATOLOGY_ANOMALIES = True
    CLIMATOLOGY_WINDOW_DAYS = 30

    # ==================== MODEL ARCHITECTURE ====================
    IMAGE_SIZE = (621, 1405)
    IMAGE_CHANNELS = 1
    # spatial_c = physics(9) + topo(1) + lat(1) + lon(1) + doy_sin(1) + doy_cos(1) + toa(1) + land_mask(1) = 16
    # model input = [x_t(1), x_tm1(1), x_tm2(1), spatial_c(16)] = 19 (deterministic)
    NUM_SPATIAL_CONDITIONS = 19
    LEAD_TIME = 15             # Direct 15-day prediction (no rollout)
    ROLLOUT_STEPS = 1          # Single forward pass at inference
    CONDITION_DIM = 5

    BASE_DIM = 64
    DIM_MULTS = (1, 2, 4, 8)
    DROPOUT_RATE = 0.25

    GLOBAL_ENCODER_DIM = 64

    BATCH_SIZE = 4
    OCEAN_FILL = 0

    # ==================== CFM SCHEDULE ====================
    CFM_SAMPLING_STEPS = 50

    # ==================== TRAINING HYPERPARAMETERS ====================
    DEVICE = "cuda"
    LEARNING_RATE = 1e-4
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
    NUM_VALIDATION_SAMPLES = 100  # Direct prediction is fast, use more samples
    WARMUP_EPOCHS = 50
    MAX_TRAIN_BATCHES = None
    USE_VALIDATION_PATIENCE = True
    EARLY_STOP_METRIC = "tac"
    EARLY_STOP_PATIENCE = 3
    EARLY_STOP_MIN_EPOCH = 10
    EARLY_STOP_MIN_DELTA = 1e-4

    # ==================== CROSS-VALIDATED HINDCASTS ====================
    CV_STRIDE = 5
    CV_VAL_OFFSETS = (3,)
    CV_TEST_OFFSETS = (4,)
    RUN_NAME = ""
    HINDCAST_STATS_DIR = os.path.join(OUTPUT_DIR, "hindcast_stats")

    # ==================== MESH CONFIG ====================
    MESH_REFINEMENT_LEVEL = 7
    MESH_PROCESSOR_ROUNDS = 8
    MESH_LATENT_DIM = 128
    MESH_BUFFER_DEG = 5.0
    K_GRID2MESH = 3
    K_MESH2GRID = 8
    CONUS_LAT_RANGE = (25.0, 50.0)
    CONUS_LON_RANGE = (-130.0, -60.0)

    # ==================== MODE ====================
    DETERMINISTIC = True  # Default to deterministic for direct prediction

    # ==================== EXTREME LOSS FINE-TUNING ====================
    USE_EXTREME_LOSS = False       # Enable after initial MSE training
    EXTREME_LOSS_WEIGHT = 0.3      # Blend: (1-w)*MSE + w*extreme
    EXTREME_LOSS_HEAT_BIAS = 1.0   # a=1.0: upweight positive extremes (heat)
    EXTREME_LOSS_COLD_BIAS = 0.0   # b=0.0: no cold extreme emphasis


def global_base_channel_count(config=Config):
    return len(config.GLOBAL_VARIABLES)


def global_effective_channel_count(config=Config):
    return global_base_channel_count(config) * len(config.GLOBAL_LAG_DAYS)


def required_input_history(config=Config):
    return max(2, max(int(lag) for lag in config.GLOBAL_LAG_DAYS))


def early_stop_score(metric_name, improved_metrics, val_mse, val_ssim):
    metric_name = str(metric_name).lower()
    if metric_name == "tac":
        return float(improved_metrics.get("tac", float("nan")))
    if metric_name == "r2":
        return float(improved_metrics.get("r2", float("nan")))
    if metric_name == "mse_skill":
        return float(improved_metrics.get("mse_skill_vs_persistence", float("nan")))
    if metric_name == "spatial_anom_r2":
        return float(improved_metrics.get("spatial_anom_r2", float("nan")))
    if metric_name == "ssim":
        return float(val_ssim)
    if metric_name == "val_mse":
        # Larger score is always better for the patience logic.
        return -float(val_mse)
    raise ValueError(f"Unknown EARLY_STOP_METRIC={metric_name!r}")


def early_stop_display_value(metric_name, score):
    return -score if str(metric_name).lower() == "val_mse" else score


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
    if not Config.USE_EXTENDED_GLOBAL_FIELDS:
        Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
        if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS:
            raise RuntimeError(
                "USE_EXTENDED_GLOBAL_FIELDS is False, but this CONUS run requires the "
                f"{Config.EXPECTED_NUM_GLOBAL_CHANNELS}-channel ERA5 global stack."
            )
        return

    report_path = Config.EXTENDED_GLOBAL_VARIABLES_PATH
    if not os.path.exists(report_path):
        Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
        if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS:
            raise FileNotFoundError(
                f"Missing extended global-field report: {report_path}. "
                f"This CONUS run requires {Config.EXPECTED_NUM_GLOBAL_CHANNELS} global channels."
            )
        return

    global_path, new_vars = _read_extended_global_variable_report(report_path)
    if global_path:
        Config.GLOBAL_DATA_PATH = global_path

    seen = set(Config.GLOBAL_VARIABLES)
    added = []
    for name in new_vars:
        if name not in seen:
            Config.GLOBAL_VARIABLES.append(name)
            seen.add(name)
            added.append(name)

    base_channels = global_base_channel_count(Config)
    Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
    if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS and base_channels != Config.EXPECTED_NUM_GLOBAL_CHANNELS:
        raise RuntimeError(
            f"Expected {Config.EXPECTED_NUM_GLOBAL_CHANNELS} global channels after loading "
            f"{report_path}, got {base_channels}."
        )
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Loaded extended global-field report: {report_path}")
        print(f"  GLOBAL_DATA_PATH: {Config.GLOBAL_DATA_PATH}")
        print(f"  Added global variables: {len(added)}")
        print(f"  Base global variables: {base_channels}")
        print(f"  Global lags: {Config.GLOBAL_LAG_DAYS}")
        print(f"  Total global input channels: {Config.NUM_GLOBAL_CHANNELS}")


def print_config_banner():
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    mode_name = "DETERMINISTIC (GraphCast)" if Config.DETERMINISTIC else "PROBABILISTIC (GenCast/CFM)"
    print(f"\n{'='*80}")
    print(f"DIRECT 15-DAY PREDICTION - {mode_name} + ICOSAHEDRAL MESH GNN")
    print(f"{'='*80}")
    print(f"Using device: {Config.DEVICE}")
    print(f"Mode: {mode_name}")
    print(f"Lead time: {Config.LEAD_TIME} days DIRECT (no rollout)")
    print(f"Sampling Steps: {Config.CFM_SAMPLING_STEPS} {'(ignored in deterministic)' if Config.DETERMINISTIC else ''}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Mesh refinement level: {Config.MESH_REFINEMENT_LEVEL}")
    print(f"Mesh processor rounds: {Config.MESH_PROCESSOR_ROUNDS}")
    print(f"Mesh latent dim: {Config.MESH_LATENT_DIM}")
    print(
        f"Global fields: {global_base_channel_count(Config)} variables x "
        f"{len(Config.GLOBAL_LAG_DAYS)} lags = {Config.NUM_GLOBAL_CHANNELS} "
        f"channels at {Config.GLOBAL_SIZE}"
    )
    print(f"Global data path: {Config.GLOBAL_DATA_PATH}")
    if Config.NUM_GLOBAL_CHANNELS > 0:
        print("Global encoder: circular longitude padding + periodic longitude sampling")
    if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
        print(
            "Training target: local climatological anomaly "
            f"(train-year-only {Config.CLIMATOLOGY_WINDOW_DAYS}-day daily climatology)"
        )
    print(f"Cross-validation: leave-k-years-out ({cv_split_tag(Config)})")
    if Config.RUN_NAME:
        print(f"Run name: {Config.RUN_NAME}")
    if Config.USE_EXTREME_LOSS:
        print(f"Extreme loss: ON (weight={Config.EXTREME_LOSS_WEIGHT}, a={Config.EXTREME_LOSS_HEAT_BIAS}, b={Config.EXTREME_LOSS_COLD_BIAS})")
    if Config.USE_VALIDATION_PATIENCE:
        print(
            "Validation patience: "
            f"metric={Config.EARLY_STOP_METRIC}, patience={Config.EARLY_STOP_PATIENCE}, "
            f"min_epoch={Config.EARLY_STOP_MIN_EPOCH}, min_delta={Config.EARLY_STOP_MIN_DELTA}"
        )


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


def get_target_variable(nc, config=Config):
    for name in config.TARGET_VARIABLE_CANDIDATES:
        if name in nc.variables:
            return name, nc.variables[name]
    available = sorted(nc.variables.keys())
    raise KeyError(
        f"None of target variables {config.TARGET_VARIABLE_CANDIDATES} were found in "
        f"{config.TRAINING_DATA_PATH}. Available variables include: {available[:40]}"
    )


def target_to_hw_time(raw, target_name):
    if raw.ndim == 4:
        return np.array(raw[:, :, 0, :], dtype=np.float32)
    if raw.ndim == 3:
        return np.array(raw, dtype=np.float32)
    raise ValueError(f"Unexpected {target_name} shape: {raw.shape}")


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
    Build valid sample indices ensuring predictor history and the direct target
    day fall within a single continuous run.
    """
    indices = []
    for start, end in runs:
        first_valid = start + min_history
        last_valid = end - lead_time
        for t in range(first_valid, last_valid + 1):
            indices.append(t)
    return indices


# ======================================================================================
# CROSS-VALIDATION SPLIT
# ======================================================================================
def build_crossval_split(valid_indices, time_values, val_stride=None, test_stride=None,
                         val_offset=None, test_offset=None, val_offsets=None,
                         test_offsets=None):
    """
    Leave-k-years-out cross-validation.
    Assigns each index to its calendar year, then holds out every Nth year
    for val and test. This ensures warm and cool decades appear in both
    train and val, removing climate trend confound.

    Default: hold out years at positions 3,8,13,... for val
             and positions 4,9,14,... for test. For k-fold hindcasts,
             pass fold-specific offset tuples so every year can be predicted
             by a model that never trained on that offset group.
    """
    val_stride = Config.CV_STRIDE if val_stride is None else val_stride
    test_stride = Config.CV_STRIDE if test_stride is None else test_stride
    if val_offsets is None:
        if val_offset is None:
            val_offsets = Config.CV_VAL_OFFSETS
        else:
            val_offsets = (val_offset,)
    if test_offsets is None:
        if test_offset is None:
            test_offsets = Config.CV_TEST_OFFSETS
        else:
            test_offsets = (test_offset,)
    val_offsets = set(parse_cv_offsets(val_offsets, val_stride) or ())
    test_offsets = set(parse_cv_offsets(test_offsets, test_stride) or ())
    overlap = val_offsets & test_offsets
    if overlap:
        raise ValueError(f"Validation and test CV offsets overlap: {sorted(overlap)}")

    base_date = datetime(1981, 5, 1)
    idx_years = np.array([
        (base_date + timedelta(days=float(time_values[i]))).year
        for i in valid_indices
    ])
    unique_years = sorted(set(idx_years))

    val_years = set()
    for offset in val_offsets:
        val_years.update(unique_years[offset::val_stride])
    test_years = set()
    for offset in test_offsets:
        test_years.update(unique_years[offset::test_stride])
    train_years = set(unique_years) - val_years - test_years

    train_indices = [i for i, y in zip(valid_indices, idx_years) if y in train_years]
    val_indices_out = [i for i, y in zip(valid_indices, idx_years) if y in val_years]
    test_indices_out = [i for i, y in zip(valid_indices, idx_years) if y in test_years]

    return train_indices, val_indices_out, test_indices_out, train_years, val_years, test_years


# ======================================================================================
# BUILD MESH
# ======================================================================================
def build_mesh_once(config, conus_mask, device, ddp=False):
    cache_path = os.path.join(config.OUTPUT_DIR, "data_cache",
                               f"mesh_level{config.MESH_REFINEMENT_LEVEL}"
                               f"_g2m{config.K_GRID2MESH}_m2g{config.K_MESH2GRID}.pkl")

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
# EXPONENTIAL EXTREME LOSS (Lopez-Gomez et al., 2023)
# ======================================================================================
def exponential_extreme_loss(pred, target, mask, a=1.0, b=0.0):
    """
    Loss that upweights prediction errors for extreme values.
    a=1, b=0: emphasize positive extremes (heat waves)
    a=0, b=1: emphasize negative extremes (cold spells)
    a=0.5, b=0.5: emphasize both tails

    Reference: Lopez-Gomez et al. (2023) "Global Extreme Heat Forecasting
    Using Neural Weather Models", AI for Earth Systems.
    """
    diff_sq = (target - pred) ** 2
    weight = a * torch.exp(target.clamp(-5, 5)) + b * torch.exp(-target.clamp(-5, 5))
    loss = weight * diff_sq
    return (loss * mask).sum() / (mask.sum() + 1e-8)


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
        (y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, t_idx, batch_mask) = batch
        x_tlead_true = y[0, 0, :h, :w]
        x_tlead_persist = x_t[0, 0, :h, :w]

        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES and val_dataset.target_climatology is not None:
            target_time_idx = int(t_idx.item()) + int(Config.LEAD_TIME)
            target_doy = int(val_dataset.doy_values[target_time_idx])
            climo = torch.from_numpy(
                np.array(val_dataset.target_climatology[target_doy], dtype=np.float32)
            )
            x_tlead_true = x_tlead_true + climo

        all_pred.append(x_tlead_persist)
        all_truth.append(x_tlead_true)
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

    # CRPS: for a deterministic (single-member) forecast, CRPS = MAE.
    # Computed per-sample then averaged, masked to land-only pixels.
    if mask_np is not None:
        mae_per_sample = []
        for b in range(pred_np.shape[0]):
            vp = pred_masked[b][~np.isnan(pred_masked[b])]
            vt = truth_masked[b][~np.isnan(truth_masked[b])]
            if len(vt) > 0:
                mae_per_sample.append(np.mean(np.abs(vt - vp)))
        mae = float(np.mean(mae_per_sample)) if mae_per_sample else 0.0
    else:
        mae = float(np.mean(np.abs(truth_np - pred_np)))

    # For a single deterministic forecast, CRPS = MAE exactly.
    # When ensemble members are available, use compute_ensemble_crps() instead.
    crps = mae

    return {
        'variance_ratio': float(variance_ratio),
        'gradient_ratio': float(gradient_ratio),
        'extreme_bias': float(extreme_bias),
        'correlation': float(correlation),
        'r2': float(r2),
        'mae': float(mae),
        'crps': float(crps),
        'pred_spatial_std': float(pred_spatial_std),
        'truth_spatial_std': float(truth_spatial_std),
    }


def masked_mse_value(pred, truth, mask=None):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]

    if mask is None:
        return float(np.mean((truth_np - pred_np) ** 2))

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
    if mask_np.ndim == 2:
        mask_np = np.broadcast_to(mask_np[None, :, :], pred_np.shape)
    elif mask_np.ndim == 4:
        mask_np = mask_np[:, 0]

    err2 = (truth_np - pred_np) ** 2
    valid = mask_np > 0.5
    return float(err2[valid].mean()) if np.any(valid) else 0.0


def calculate_spatial_anomaly_r2(pred, truth, mask=None):
    """
    R2 after removing each sample's masked spatial mean.
    This is a stricter diagnostic than absolute-temperature R2: it asks whether
    the forecast captures within-CONUS spatial anomaly structure, not just the
    easy domain-mean/seasonal/latitude signal.
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]

    mask_np = None
    if mask is not None:
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        if mask_np.ndim == 4:
            mask_np = mask_np[0, 0]
        elif mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_np = mask_np > 0.5

    ss_res = 0.0
    ss_tot = 0.0
    for b in range(pred_np.shape[0]):
        if mask_np is not None:
            p = pred_np[b][mask_np]
            t = truth_np[b][mask_np]
        else:
            p = pred_np[b].ravel()
            t = truth_np[b].ravel()
        if t.size == 0:
            continue
        p = p - np.mean(p)
        t = t - np.mean(t)
        ss_res += float(np.sum((t - p) ** 2))
        ss_tot += float(np.sum(t ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def build_train_daily_climatology_30day(shared_data, train_indices, norm_stats, config):
    """
    Train-year-only local daily climatology for TAC.

    The target date for a direct sample is t+LEAD_TIME, so climatology is built
    from the training targets' calendar days, then smoothed with a 30-day window.
    Values are returned in normalized units, so target anomalies are simply
    target_z - climatology_z.
    """
    heat_index = shared_data["heat_index"]
    time_values = np.array(shared_data["time_values"])
    doys = compute_doy_array(time_values).astype(np.int16)
    target_indices = np.array(train_indices, dtype=np.int64) + int(config.LEAD_TIME)
    target_doys = doys[target_indices].astype(np.int16)

    h, w = config.IMAGE_SIZE
    daily_sum = np.zeros((367, h, w), dtype=np.float32)
    daily_count = np.zeros((367, h, w), dtype=np.uint16)

    for ti, doy in zip(target_indices, target_doys):
        field = np.array(heat_index[:, :, ti], dtype=np.float32)
        valid = np.isfinite(field) & (field != 0.0)
        daily_sum[doy][valid] += field[valid]
        daily_count[doy][valid] += 1

    climo = np.zeros((367, h, w), dtype=np.float32)
    window = max(1, int(config.CLIMATOLOGY_WINDOW_DAYS))
    before = window // 2
    after = window - before - 1
    for doy in range(1, 367):
        lo = max(1, doy - before)
        hi = min(366, doy + after)
        sum_win = daily_sum[lo:hi + 1].sum(axis=0)
        count_win = daily_count[lo:hi + 1].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            climo[doy] = sum_win / np.maximum(count_win, 1)
        climo[doy][count_win == 0] = float(norm_stats["hi_mean"])

    del daily_sum, daily_count
    climo = (climo - float(norm_stats["hi_mean"])) / (float(norm_stats["hi_std"]) + 1e-8)
    climo = np.nan_to_num(climo, nan=0.0, posinf=0.0, neginf=0.0)
    return climo.astype(np.float16)


def get_climatology_cache_path(config=None):
    if config is None:
        config = Config
    return os.path.join(
        config.OUTPUT_DIR,
        "data_cache",
        f"local_daily_climo{config.CLIMATOLOGY_WINDOW_DAYS}_direct15_{cv_split_tag(config)}.npy",
    )


def load_or_build_train_climatology(shared_data, train_indices, norm_stats, config, ddp=False):
    if not config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
        return None

    path = get_climatology_cache_path(config)
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stats_path = get_norm_stats_path(config)
        climo_ok = os.path.exists(path)
        if climo_ok and os.path.exists(stats_path):
            climo_ok = os.path.getmtime(path) >= os.path.getmtime(stats_path)
        if climo_ok:
            print(f"Loading train-year-only local climatology from {path}")
        else:
            print(
                "Building train-year-only local climatology for anomaly training/TAC "
                f"({config.CLIMATOLOGY_WINDOW_DAYS}-day running daily means)..."
            )
            climo = build_train_daily_climatology_30day(
                shared_data, train_indices, norm_stats, config
            )
            np.save(path, climo)
            print(f"  Saved local climatology to {path}")

    if ddp:
        dist.barrier()

    # Keep this in RAM. It is ~0.6 GB as float16 for CONUS, and reading one
    # full climatology slice per sample from network storage would slow training.
    climo = np.load(path)
    if is_main_process():
        print("  Local climatology ready: train years only, no held-out leakage.")
    return climo


def restore_full_field_from_anomaly(field, target_doys, climo_by_doy, mask=None):
    if (
        not Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES
        or climo_by_doy is None
    ):
        return field

    doys = np.asarray(target_doys, dtype=np.int64)
    climo_np = np.asarray(climo_by_doy[doys], dtype=np.float32)
    climo = torch.from_numpy(climo_np).unsqueeze(1)
    climo = climo.to(device=field.device, dtype=field.dtype)
    out = field + climo
    if mask is not None:
        mask_t = mask.to(device=out.device, dtype=out.dtype)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(1)
        out = out * mask_t + Config.OCEAN_FILL * (1 - mask_t)
    return out


def temporal_anomaly_correlation_from_climo(pred, truth, target_doys, climo_by_doy, mask, chunk_pixels=20000):
    """
    Operational-style TAC:
      1. remove local daily climatology at every grid point,
      2. correlate forecast and observed anomalies across time at each grid point,
      3. average valid gridpoint correlations over CONUS land.
    """
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    truth_np = truth.cpu().numpy() if isinstance(truth, torch.Tensor) else np.asarray(truth)
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]
    target_doys = np.asarray(target_doys, dtype=np.int64)

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if mask_np.ndim == 4:
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:
        mask_np = mask_np[0]
    pixel_ids = np.flatnonzero(mask_np.reshape(-1) > 0.5)
    if pixel_ids.size == 0 or pred_np.shape[0] < 2:
        return 0.0

    pred_flat = pred_np.reshape(pred_np.shape[0], -1)
    truth_flat = truth_np.reshape(truth_np.shape[0], -1)
    climo_flat = climo_by_doy.reshape(climo_by_doy.shape[0], -1)

    corr_sum = 0.0
    corr_count = 0
    for start in range(0, pixel_ids.size, chunk_pixels):
        cols = pixel_ids[start:start + chunk_pixels]
        climo = climo_flat[np.ix_(target_doys, cols)].astype(np.float32)
        pred_anom = pred_flat[:, cols].astype(np.float32) - climo
        truth_anom = truth_flat[:, cols].astype(np.float32) - climo

        pred_anom -= pred_anom.mean(axis=0, keepdims=True)
        truth_anom -= truth_anom.mean(axis=0, keepdims=True)
        cov = np.mean(pred_anom * truth_anom, axis=0)
        pred_std = np.sqrt(np.mean(pred_anom ** 2, axis=0))
        truth_std = np.sqrt(np.mean(truth_anom ** 2, axis=0))
        corr = cov / (pred_std * truth_std + 1e-8)
        valid = np.isfinite(corr) & (pred_std > 1e-6) & (truth_std > 1e-6)
        if np.any(valid):
            corr_sum += float(np.sum(corr[valid]))
            corr_count += int(np.sum(valid))

    return corr_sum / max(corr_count, 1)


def compute_ensemble_crps(ensemble_preds, truth, mask=None):
    """
    CRPS for an ensemble forecast using the energy form:
        CRPS = E|X - y| - 0.5 * E|X - X'|

    ensemble_preds: (M, N_samples, H, W) array, M ensemble members
    truth:          (N_samples, H, W) array
    mask:           (H, W) binary array, optional

    Returns per-sample CRPS averaged over samples and land pixels.
    For M=1, this reduces to MAE (the E|X-X'| term is zero).
    """
    if isinstance(ensemble_preds, torch.Tensor):
        ensemble_preds = ensemble_preds.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

    M = ensemble_preds.shape[0]
    N = ensemble_preds.shape[1]

    crps_values = []
    for n in range(N):
        obs = truth[n]  # (H, W)
        members = ensemble_preds[:, n]  # (M, H, W)

        # Term 1: E|X - y| = mean over members of |member - obs|
        abs_errors = np.abs(members - obs[None, :, :])  # (M, H, W)
        term1 = np.mean(abs_errors, axis=0)  # (H, W)

        # Term 2: E|X - X'| = mean over all pairs
        if M > 1:
            pair_diffs = 0.0
            count = 0
            for i in range(M):
                for j in range(i + 1, M):
                    pair_diffs = pair_diffs + np.abs(members[i] - members[j])
                    count += 1
            term2 = pair_diffs / count  # (H, W), average over unique pairs
        else:
            term2 = np.zeros_like(term1)

        crps_map = term1 - 0.5 * term2  # (H, W)

        if mask is not None:
            valid = mask > 0.5
            if valid.any():
                crps_values.append(float(np.mean(crps_map[valid])))
        else:
            crps_values.append(float(np.mean(crps_map)))

    return float(np.mean(crps_values)) if crps_values else 0.0


# ======================================================================================
# DATASET
# ======================================================================================
class ClimateDataset(Dataset):
    def __init__(self, config, mode='train', train_indices=None, val_indices=None, test_indices=None,
                 normalization_stats=None, shared_data=None, target_climatology=None):
        if is_main_process():
            print(f"Initializing ClimateDataset ({mode} mode)...")
        self.config = config
        self.mode = mode
        self.target_climatology = target_climatology

        if shared_data is None:
            print(f"  Loading data to RAM...")
            with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
                target_name, target_var = get_target_variable(nc, config)
                raw_shape = target_var.shape
                print(f"  Raw target variable: {target_name}, shape: {raw_shape}")
                self.heat_index = target_to_hw_time(target_var[:], target_name)
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
                train_indices = list(
                    range(required_input_history(self.config), self.n_timesteps - self.config.LEAD_TIME)
                )

            train_t_indices = np.array(train_indices)
            target_indices = train_t_indices + self.config.LEAD_TIME
            hi_train = self.heat_index[:, :, target_indices]
            hi_valid = np.isfinite(hi_train) & (hi_train != 0.0)
            if not np.any(hi_valid):
                raise ValueError("No valid daily target values found in the training split.")
            self.hi_mean = torch.tensor(float(np.mean(hi_train[hi_valid])), dtype=torch.float32)
            self.hi_std = torch.tensor(float(np.std(hi_train[hi_valid])), dtype=torch.float32)

            print(
                f"    Target -> Mean: {self.hi_mean:.4f}, "
                f"Std: {self.hi_std:.4f}"
            )

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

            del hi_train, hi_valid, gp_train, sm_train, slp_train, t2m_train, q850_train, t850_train, u850_train, v850_train, z300_train, cond_train_subset
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
            self.indices = train_indices if train_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - self.config.LEAD_TIME)
            )
        elif mode == 'val':
            self.indices = val_indices if val_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - self.config.LEAD_TIME)
            )
        elif mode == 'test':
            self.indices = test_indices if test_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - self.config.LEAD_TIME)
            )

        if (
            self.config.TRAIN_ON_CLIMATOLOGY_ANOMALIES
            and self.target_climatology is None
            and normalization_stats is not None
        ):
            raise ValueError(
                "target_climatology is required when TRAIN_ON_CLIMATOLOGY_ANOMALIES=True."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        h, w = self.config.IMAGE_SIZE
        target_t = t + self.config.LEAD_TIME
        min_history = required_input_history(self.config)
        if t < min_history or target_t >= self.n_timesteps:
            raise IndexError(
                f"Invalid direct-forecast sample t={t}: needs t-{min_history} >= 0 and "
                f"t+lead={target_t} < {self.n_timesteps}."
            )

        x_t_slice      = self.heat_index[:, :, t]
        x_tm1_slice    = self.heat_index[:, :, t - 1]
        x_tm2_slice    = self.heat_index[:, :, t - 2]
        x_target_slice = self.heat_index[:, :, target_t]

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

        if self.config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            target_doy = int(self.doy_values[target_t])
            climo_np = np.array(self.target_climatology[target_doy], dtype=np.float32, copy=True)
            climo = torch.from_numpy(climo_np).unsqueeze(0)
            y = (y - climo) * land_mask + Config.OCEAN_FILL * (1 - land_mask)

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
        if topo.shape != (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]):
            topo = topo.T
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
            lagged_global_fields = []
            for lag in self.config.GLOBAL_LAG_DAYS:
                src_t = t - int(lag)
                if src_t < 0:
                    raise IndexError(f"Global lag {lag} for t={t} points before the record start.")
                global_channels = []
                for var_name, var_data in self.global_data.items():
                    g_slice = torch.from_numpy(np.array(var_data[:, :, src_t], dtype=np.float32))
                    global_channels.append(g_slice)
                lag_fields = torch.stack(global_channels, dim=0)
                lag_fields = (lag_fields - self.global_mean) / (self.global_std + 1e-8)
                lagged_global_fields.append(lag_fields)
            global_fields = torch.cat(lagged_global_fields, dim=0)
            if global_fields.shape[0] != Config.NUM_GLOBAL_CHANNELS:
                raise RuntimeError(
                    f"Built {global_fields.shape[0]} lagged global channels, "
                    f"expected {Config.NUM_GLOBAL_CHANNELS}."
                )
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        mask = land_mask

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
# SSIM
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
        target_name, hi = get_target_variable(nc, config)
        if hi.ndim == 4:
            hi_sample = hi[:, :, 0, 0]
        elif hi.ndim == 3:
            hi_sample = hi[:, :, 0]
        else:
            raise ValueError(f"Unexpected {target_name} ndim: {hi.ndim}")
        mask = (np.abs(hi_sample) > 0.01).astype(np.float32)
    land_fraction = mask.sum() / mask.size
    print(
        f"  Mask loaded from {target_name}: {land_fraction*100:.1f}% land, "
        f"{(1-land_fraction)*100:.1f}% ocean"
    )
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
# DIRECT PREDICTION HELPER
# ======================================================================================
@torch.inference_mode()
def predict_direct(model, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask, device):
    """
    Single forward pass: predict the day-15 field directly.
    All inputs should be batched (B, C, H, W) and on device.
    Returns prediction tensor (B, 1, H, W).

    Model signature: model(x_input, t, vec_c, global_fields=...)
    where x_input = cat([x_t, x_tm1, x_tm2, spatial_c]) for deterministic.
    """
    h, w = Config.IMAGE_SIZE
    if Config.DETERMINISTIC:
        from mode_dispatch import _deterministic_input
        x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
        dummy_t = torch.full((x_t.shape[0],), 0.5, device=device)
        pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)
        pred = pred * mask + Config.OCEAN_FILL * (1 - mask)
        pred = pred.clamp(-4.0, 4.0)
    else:
        # CFM mode: generate_sample signature is
        # (model, spatial_c, vec_c, x_t, x_tm1, x_tm2, global_fields, device, h, w, mask, ...)
        pred_np = generate_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask,
            deterministic=False, n_steps=Config.CFM_SAMPLING_STEPS
        )
        pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(device)
        pred = pred * mask
    return pred


# ======================================================================================
# VALIDATION (Direct single-pass prediction)
# ======================================================================================
@torch.inference_mode()
def calculate_validation_metrics_cfm(model, val_dataset, device, mask,
                                      n_samples=100, rank=0, world_size=1, ddp=False,
                                      tac_climatology=None):
    """
    Validation via direct single-step prediction at LEAD_TIME=15.
    No autoregressive rollout. Each sample is one forward pass.
    """
    import time
    validation_start = time.time()

    model.eval()
    if len(val_dataset) == 0:
        return 0.0, 0.0, 0.0, {}

    n_samples = min(n_samples, len(val_dataset))
    per_rank = max(1, n_samples // world_size)
    actual_n_samples = per_rank * world_size

    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, len(val_dataset))
    rank_sample_count = end_idx - start_idx

    if is_main_process():
        print(f"\n  Generating {actual_n_samples} direct 15-day predictions "
              f"across {world_size} GPUs ({per_rank} per GPU)...")

    h, w = Config.IMAGE_SIZE
    if isinstance(mask, torch.Tensor):
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)

    all_preds = []
    all_truth = []
    all_persist = []
    all_target_doys = []

    for sample_i in range(rank_sample_count):
        dataset_idx = start_idx + sample_i
        batch = val_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        # Move to device, add batch dimension
        y_dev = y.unsqueeze(0).to(device)
        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        # Single forward pass
        pred = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                              spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)

        all_preds.append(pred[0, 0, :h, :w].cpu())
        all_truth.append(y[0, :h, :w].cpu())
        all_persist.append(x_t[0, :h, :w].cpu())
        target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
        all_target_doys.append(int(val_dataset.doy_values[target_time_idx]))

    local_preds = torch.stack(all_preds)
    local_truth = torch.stack(all_truth)
    local_persist = torch.stack(all_persist)
    local_target_doys = torch.tensor(all_target_doys, dtype=torch.long)

    if ddp:
        local_preds = local_preds.to(device)
        local_truth = local_truth.to(device)
        local_persist = local_persist.to(device)
        local_target_doys = local_target_doys.to(device)

        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_truth = [torch.zeros_like(local_truth) for _ in range(world_size)]
        gathered_persist = [torch.zeros_like(local_persist) for _ in range(world_size)]
        gathered_target_doys = [torch.zeros_like(local_target_doys) for _ in range(world_size)]

        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_truth, local_truth)
        dist.all_gather(gathered_persist, local_persist)
        dist.all_gather(gathered_target_doys, local_target_doys)

        all_preds = torch.cat(gathered_preds, dim=0).cpu()
        all_truth = torch.cat(gathered_truth, dim=0).cpu()
        all_persist = torch.cat(gathered_persist, dim=0).cpu()
        all_target_doys = torch.cat(gathered_target_doys, dim=0).cpu().numpy()
    else:
        all_preds = local_preds.cpu()
        all_truth = local_truth.cpu()
        all_persist = local_persist.cpu()
        all_target_doys = local_target_doys.cpu().numpy()

    if not is_main_process():
        return 0.0, 0.0, 0.0, {}

    all_preds = all_preds.unsqueeze(1)
    all_truth = all_truth.unsqueeze(1)
    all_persist = all_persist.unsqueeze(1)
    all_preds = torch.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
    all_truth = torch.nan_to_num(all_truth, nan=0.0, posinf=0.0, neginf=0.0)
    all_persist = torch.nan_to_num(all_persist, nan=0.0, posinf=0.0, neginf=0.0)

    mask_2d = mask_4d[0, 0, :h, :w].cpu()
    all_preds_eval = restore_full_field_from_anomaly(
        all_preds, all_target_doys, tac_climatology, mask_2d
    )
    all_truth_eval = restore_full_field_from_anomaly(
        all_truth, all_target_doys, tac_climatology, mask_2d
    )

    preds_np = all_preds_eval.numpy()
    truth_np = all_truth_eval.numpy()
    m = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds_eval).numpy()

    err2 = (preds_np - truth_np) ** 2
    mse_per_sample = (err2 * m).sum(axis=(1,2,3)) / (m.sum(axis=(1,2,3)) + 1e-8)
    avg_mse  = float(mse_per_sample.mean())
    avg_rmse = float(np.sqrt(avg_mse))

    def to_ssim_space(x):
        return ((x + 3.0) / 6.0).clamp(0.0, 1.0)

    preds_ssim = to_ssim_space(all_preds_eval).to(device)
    truth_ssim = to_ssim_space(all_truth_eval).to(device)
    mask_ssim = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds_eval).to(device)
    avg_ssim = masked_ssim_01(preds_ssim, truth_ssim, mask_ssim, fill=0.5).item()

    improved = calculate_improved_metrics(all_preds_eval, all_truth_eval, mask=mask_2d)
    persistence = calculate_improved_metrics(all_persist, all_truth_eval, mask=mask_2d)
    zero_baseline = calculate_improved_metrics(torch.zeros_like(all_truth_eval), all_truth_eval, mask=mask_2d)
    persistence_mse = masked_mse_value(all_persist, all_truth_eval, mask=mask_2d)
    mse_skill_vs_persistence = 1.0 - avg_mse / (persistence_mse + 1e-8)
    spatial_anom_r2 = calculate_spatial_anomaly_r2(all_preds_eval, all_truth_eval, mask=mask_2d)
    persistence_spatial_anom_r2 = calculate_spatial_anomaly_r2(all_persist, all_truth_eval, mask=mask_2d)
    tac = float("nan")
    persistence_tac = float("nan")
    if tac_climatology is not None:
        tac = temporal_anomaly_correlation_from_climo(
            all_preds_eval, all_truth_eval, all_target_doys, tac_climatology, mask_2d
        )
        persistence_tac = temporal_anomaly_correlation_from_climo(
            all_persist, all_truth_eval, all_target_doys, tac_climatology, mask_2d
        )

    validation_total = time.time() - validation_start
    print(f"\n  Validation time: {validation_total:.2f}s "
          f"({actual_n_samples/max(validation_total, 0.01):.1f} samples/sec)")
    print(f"  Metrics (direct 15-day): MSE={avg_mse:.6f}, SSIM={avg_ssim:.4f}, "
          f"R2={improved['r2']:.4f}, CRPS={improved['crps']:.4f}")
    print(f"  Leakage diagnostics: persistence_R2={persistence['r2']:.4f}, "
          f"zero_R2={zero_baseline['r2']:.4f}, "
          f"MSE_skill_vs_persistence={mse_skill_vs_persistence:.4f}")
    print(f"  Spatial-anomaly R2: model={spatial_anom_r2:.4f}, "
          f"persistence={persistence_spatial_anom_r2:.4f}")
    if tac_climatology is not None:
        print(f"  Temporal anomaly correlation (TAC): model={tac:.4f}, "
              f"persistence={persistence_tac:.4f}")

    return avg_mse, avg_rmse, avg_ssim, {
        'variance_ratio':    improved['variance_ratio'],
        'gradient_ratio':    improved['gradient_ratio'],
        'extreme_bias':      improved['extreme_bias'],
        'correlation':       improved['correlation'],
        'r2':                improved['r2'],
        'mae':               improved['mae'],
        'crps':              improved['crps'],
        'pred_spatial_std':  improved['pred_spatial_std'],
        'truth_spatial_std': improved['truth_spatial_std'],
        'persistence_r2':    persistence['r2'],
        'persistence_mae':   persistence['mae'],
        'persistence_mse':   persistence_mse,
        'zero_r2':           zero_baseline['r2'],
        'mse_skill_vs_persistence': mse_skill_vs_persistence,
        'spatial_anom_r2':   spatial_anom_r2,
        'persistence_spatial_anom_r2': persistence_spatial_anom_r2,
        'tac':               tac,
        'persistence_tac':   persistence_tac,
    }


def _empty_tac_stats(h, w):
    return {
        "pred_sum": np.zeros((h, w), dtype=np.float64),
        "truth_sum": np.zeros((h, w), dtype=np.float64),
        "pred_sq_sum": np.zeros((h, w), dtype=np.float64),
        "truth_sq_sum": np.zeros((h, w), dtype=np.float64),
        "pred_truth_sum": np.zeros((h, w), dtype=np.float64),
        "persist_sum": np.zeros((h, w), dtype=np.float64),
        "persist_sq_sum": np.zeros((h, w), dtype=np.float64),
        "persist_truth_sum": np.zeros((h, w), dtype=np.float64),
        "count": np.zeros((h, w), dtype=np.float64),
    }


def _accumulate_tac_stats(stats, pred, truth, persist, target_doy, climo_by_doy, mask_np):
    climo = climo_by_doy[int(target_doy)].astype(np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    persist = np.asarray(persist, dtype=np.float32)
    valid = (
        (mask_np > 0.5)
        & np.isfinite(pred)
        & np.isfinite(truth)
        & np.isfinite(persist)
        & np.isfinite(climo)
    )
    if not np.any(valid):
        return

    pred_anom = np.zeros_like(pred, dtype=np.float32)
    truth_anom = np.zeros_like(truth, dtype=np.float32)
    persist_anom = np.zeros_like(persist, dtype=np.float32)
    pred_anom[valid] = pred[valid] - climo[valid]
    truth_anom[valid] = truth[valid] - climo[valid]
    persist_anom[valid] = persist[valid] - climo[valid]
    valid_f = valid.astype(np.float64)

    pa = pred_anom.astype(np.float64, copy=False)
    ta = truth_anom.astype(np.float64, copy=False)
    xa = persist_anom.astype(np.float64, copy=False)

    stats["pred_sum"] += pa
    stats["truth_sum"] += ta
    stats["pred_sq_sum"] += pa * pa
    stats["truth_sq_sum"] += ta * ta
    stats["pred_truth_sum"] += pa * ta
    stats["persist_sum"] += xa
    stats["persist_sq_sum"] += xa * xa
    stats["persist_truth_sum"] += xa * ta
    stats["count"] += valid_f


def _corr_from_tac_sums(x_sum, y_sum, x_sq_sum, y_sq_sum, xy_sum, count, mask_np):
    valid = (mask_np > 0.5) & (count >= 2)
    cov = xy_sum - (x_sum * y_sum) / np.maximum(count, 1.0)
    x_var = x_sq_sum - (x_sum * x_sum) / np.maximum(count, 1.0)
    y_var = y_sq_sum - (y_sum * y_sum) / np.maximum(count, 1.0)
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr, float(np.nanmean(corr[valid])) if np.any(valid) else float("nan")


def summarize_tac_stats(stats, mask_np):
    model_corr, model_tac = _corr_from_tac_sums(
        stats["pred_sum"], stats["truth_sum"],
        stats["pred_sq_sum"], stats["truth_sq_sum"],
        stats["pred_truth_sum"], stats["count"], mask_np,
    )
    persistence_corr, persistence_tac = _corr_from_tac_sums(
        stats["persist_sum"], stats["truth_sum"],
        stats["persist_sq_sum"], stats["truth_sq_sum"],
        stats["persist_truth_sum"], stats["count"], mask_np,
    )
    return model_tac, persistence_tac, model_corr, persistence_corr


def norm_stats_match_npz(s, config=Config):
    stats_target_half_window = int(s["target_half_window"]) if "target_half_window" in s else 0
    stats_lags = (
        tuple(np.atleast_1d(s["global_lag_days"]).astype(int).tolist())
        if "global_lag_days" in s else ()
    )
    stats_anomaly_target = (
        bool(int(s["train_on_climatology_anomalies"]))
        if "train_on_climatology_anomalies" in s else False
    )
    return (
        "global_mean" in s
        and int(s["global_mean"].shape[0]) == global_base_channel_count(config)
        and "lead_time" in s
        and int(s["lead_time"]) == config.LEAD_TIME
        and stats_target_half_window == 0
        and stats_lags == tuple(int(x) for x in config.GLOBAL_LAG_DAYS)
        and stats_anomaly_target == bool(config.TRAIN_ON_CLIMATOLOGY_ANOMALIES)
        and "cv_stride" in s
        and int(s["cv_stride"]) == config.CV_STRIDE
        and "cv_val_offsets" in s
        and tuple(np.atleast_1d(s["cv_val_offsets"]).astype(int).tolist()) == config.CV_VAL_OFFSETS
        and "cv_test_offsets" in s
        and tuple(np.atleast_1d(s["cv_test_offsets"]).astype(int).tolist()) == config.CV_TEST_OFFSETS
    )


def load_norm_stats_npz(stats_path):
    s = np.load(stats_path)
    if not norm_stats_match_npz(s, Config):
        raise RuntimeError(
            f"Normalization stats at {stats_path} do not match the active "
            f"{global_base_channel_count(Config)}-variable/"
            f"{Config.NUM_GLOBAL_CHANNELS}-lagged-channel global stack, "
            f"anomaly_target={Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES}, "
            f"and {cv_split_tag(Config)}. "
            "Re-run training setup for this fold or delete stale norm stats."
        )
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
    s.close()
    return norm_stats


@torch.inference_mode()
def export_hindcast_tac_stats(model, dataset, split_name, output_path, device, mask,
                              tac_climatology, rank=0, world_size=1, ddp=False,
                              max_samples=None):
    model.eval()
    h, w = Config.IMAGE_SIZE
    mask_np = mask[:h, :w].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask[:h, :w])
    mask_4d = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
    stats = _empty_tac_stats(h, w)

    total = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    rank_indices = list(range(rank, total, world_size))
    iterator = tqdm(rank_indices, desc=f"Export hindcast {split_name}", disable=not is_main_process())

    for dataset_idx in iterator:
        batch = dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        pred = predict_direct(
            model,
            x_t.unsqueeze(0).to(device),
            x_tm1.unsqueeze(0).to(device),
            x_tm2.unsqueeze(0).to(device),
            spatial_c.unsqueeze(0).to(device),
            vec_c.unsqueeze(0).to(device),
            global_fields.unsqueeze(0).to(device),
            batch_mask.unsqueeze(0).to(device),
            device,
        )

        target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
        target_doy = int(dataset.doy_values[target_time_idx])
        pred_np = pred[0, 0, :h, :w].detach().cpu().numpy()
        truth_np = y[0, :h, :w].detach().cpu().numpy()
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            climo_np = np.asarray(tac_climatology[target_doy], dtype=np.float32)
            pred_np = pred_np + climo_np
            truth_np = truth_np + climo_np
        _accumulate_tac_stats(
            stats,
            pred_np,
            truth_np,
            x_t[0, :h, :w].detach().cpu().numpy(),
            target_doy,
            tac_climatology,
            mask_np,
        )

    if ddp:
        for key, value in list(stats.items()):
            tensor = torch.from_numpy(value).to(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            stats[key] = tensor.cpu().numpy()

    if not is_main_process():
        return

    model_tac, persistence_tac, _, _ = summarize_tac_stats(stats, mask_np)
    years = sorted({
        (datetime(1981, 5, 1) + timedelta(days=float(dataset.time_values[int(t)]))).year
        for t in dataset.indices[:total]
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        **stats,
        mask=mask_np.astype(np.uint8),
        split_name=np.array(split_name),
        run_name=np.array(Config.RUN_NAME),
        cv_split=np.array(cv_split_tag(Config)),
        years=np.array(years, dtype=np.int16),
        n_samples=np.array(total, dtype=np.int32),
        model_tac=np.array(model_tac, dtype=np.float32),
        persistence_tac=np.array(persistence_tac, dtype=np.float32),
    )
    print(
        f"  Saved hindcast TAC stats for {split_name} to {output_path}\n"
        f"  {split_name}: n={total}, years={years}, TAC={model_tac:.4f}, "
        f"persistence_TAC={persistence_tac:.4f}"
    )


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
    else:
        mask_np = np.array(mask)

    model.eval()

    collected = 0
    for dataset_idx in range(min(n_samples, len(val_dataset))):
        batch = val_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        # Single forward pass
        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_tensor = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                                         spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)

        pred = pred_tensor[0, 0, :h, :w].cpu().numpy()
        truth = y[0, :h, :w].numpy()
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES and val_dataset.target_climatology is not None:
            target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
            target_doy = int(val_dataset.doy_values[target_time_idx])
            climo = np.asarray(val_dataset.target_climatology[target_doy], dtype=np.float32)
            pred = pred + climo
            truth = truth + climo

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
        axes[0].set_title('Ground Truth (t+15)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0], label='Z-score')
        im = axes[1].imshow(pred_display, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'Direct 15-day Prediction\nMAE={mae:.3f}, r={corr:.3f}, R2={r2:.3f}',
                          fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Z-score')
        diff_display = np.where(mask_np[:h, :w] > 0.5, pred - truth, np.nan)
        im = axes[2].imshow(diff_display, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
        axes[2].set_title('Prediction - Truth', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[2], label='Difference (Z-score)')
        fig.suptitle(f'Epoch {epoch} - Val Sample {collected + 1} (direct 15-day)', fontsize=15, fontweight='bold')
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
def prepare_shared_data(config, rank, world_size, ddp):
    cache_dir = os.path.join(config.OUTPUT_DIR, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    target_meta_path = os.path.join(cache_dir, "_target_cache_meta.npz")

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
    global_meta_path = os.path.join(global_cache_dir, "_global_cache_meta.npz")
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
                else:
                    with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc_probe:
                        source_target_name, _ = get_target_variable(nc_probe, config)
                    meta = np.load(target_meta_path, allow_pickle=True)
                    cached_target_name = str(meta["target_variable"].item())
                    if cached_target_name != source_target_name:
                        cache_ok = False
            except Exception:
                cache_ok = False

        if not cache_ok:
            with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc:
                target_name, target_var = get_target_variable(nc, config)
                print(f"Rank 0: caching target variable {target_name} with shape {target_var.shape}")
                hi = target_to_hw_time(target_var[:], target_name)
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
                np.save(paths["time_values"], np.array(nc.variables["time"][:], dtype=np.float64))
                np.savez(target_meta_path, target_variable=np.array(target_name, dtype=object))

            with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
                topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
                topo = np.flipud(topo)
            np.save(paths["topography"], topo)
            print(f"Rank 0: wrote CONUS data cache to {cache_dir}")
        else:
            print(f"Rank 0: using existing CONUS data cache in {cache_dir}")

        global_cache_ok = all(os.path.exists(p) for p in global_paths.values())
        if global_cache_ok:
            try:
                meta = np.load(global_meta_path, allow_pickle=True)
                cached_vars = [str(v) for v in meta["variables"].tolist()]
                cached_path = str(meta["global_data_path"].item())
                if cached_vars != list(config.GLOBAL_VARIABLES) or cached_path != config.GLOBAL_DATA_PATH:
                    global_cache_ok = False
            except Exception:
                global_cache_ok = False
        if not global_cache_ok:
            print(f"Rank 0: caching {global_base_channel_count(config)} base global teleconnection fields...")
            print(f"  Source: {config.GLOBAL_DATA_PATH}")
            with NetCDFDataset(config.GLOBAL_DATA_PATH, "r") as nc_g:
                missing = [name for name in config.GLOBAL_VARIABLES if name not in nc_g.variables]
                if missing:
                    raise KeyError(
                        f"Missing {len(missing)} GLOBAL_VARIABLES in {config.GLOBAL_DATA_PATH}: {missing[:20]}"
                    )
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
            np.savez(
                global_meta_path,
                variables=np.array(config.GLOBAL_VARIABLES, dtype=object),
                global_data_path=np.array(config.GLOBAL_DATA_PATH, dtype=object),
            )
            print(f"Rank 0: wrote global data cache")
        else:
            print(
                f"Rank 0: using existing global data cache "
                f"({global_base_channel_count(config)} base fields)"
            )

    if ddp:
        dist.barrier()

    shm_dir = "/dev/shm/cfm_cache"

    if not ddp or dist.get_rank() == 0:
        os.makedirs(shm_dir, exist_ok=True)
        for key, src_path in paths.items():
            dst = os.path.join(shm_dir, os.path.basename(src_path))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        shm_global = os.path.join(shm_dir, "global")
        os.makedirs(shm_global, exist_ok=True)
        shm_global_meta = os.path.join(shm_global, os.path.basename(global_meta_path))
        shm_global_ok = all(
            os.path.exists(os.path.join(shm_global, os.path.basename(p)))
            for p in global_paths.values()
        )
        if shm_global_ok:
            try:
                meta = np.load(shm_global_meta, allow_pickle=True)
                cached_vars = [str(v) for v in meta["variables"].tolist()]
                cached_path = str(meta["global_data_path"].item())
                if cached_vars != list(config.GLOBAL_VARIABLES) or cached_path != config.GLOBAL_DATA_PATH:
                    shm_global_ok = False
            except Exception:
                shm_global_ok = False
        for var_name, src_path in global_paths.items():
            dst = os.path.join(shm_global, os.path.basename(src_path))
            if not shm_global_ok or not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        if not shm_global_ok or not os.path.exists(shm_global_meta):
            shutil.copy2(global_meta_path, shm_global_meta)
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
    if len(global_data) != global_base_channel_count(config):
        raise RuntimeError(
            f"Loaded {len(global_data)} base global fields, expected "
            f"{global_base_channel_count(config)}."
        )
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
        print("DIRECT 15-DAY PREDICTION + ICOSAHEDRAL MESH GNN")
        print(f"World size: {world_size}")
        print(f"Mesh level: {Config.MESH_REFINEMENT_LEVEL}, Rounds: {Config.MESH_PROCESSOR_ROUNDS}")
        print(f"Lead time: {Config.LEAD_TIME} days DIRECT (no rollout)")
        print(
            f"Global fields: {global_base_channel_count(Config)} variables x "
            f"{len(Config.GLOBAL_LAG_DAYS)} lags = {Config.NUM_GLOBAL_CHANNELS} channels "
            f"from {Config.GLOBAL_DATA_PATH}"
        )
        print(
            "Cross-validation: leave-k-years-out "
            f"({cv_split_tag(Config)}, run_name={Config.RUN_NAME or 'default'})"
        )
        print("=" * 80 + "\n")

    # Data
    shared_data = prepare_shared_data(Config, rank, world_size, ddp)
    n_timesteps = shared_data['heat_index'].shape[-1]

    # Read time values for season boundary detection
    time_values = np.array(shared_data['time_values'])

    runs = detect_continuous_runs(time_values)
    if is_main_process():
        print(f"Detected {len(runs)} continuous runs (expected ~43 MJJAS seasons)")

    # Build valid indices: both train and val need lead_time=15
    all_valid = build_valid_indices(
        runs,
        lead_time=Config.LEAD_TIME,
        min_history=required_input_history(Config),
    )

    if is_main_process():
        print(f"Total valid indices (lead={Config.LEAD_TIME}): {len(all_valid)}")

    # Cross-validation split: leave-k-years-out
    train_indices, val_indices, test_indices, train_years, val_years, test_years = \
        build_crossval_split(all_valid, time_values)

    if is_main_process():
        print(f"\nLeave-k-Years-Out Cross-Validation Split:")
        print(f"  Train: {len(train_indices)} samples ({len(train_years)} years)")
        print(f"  Val:   {len(val_indices)} samples ({len(val_years)} years: {sorted(val_years)})")
        print(f"  Test:  {len(test_indices)} samples ({len(test_years)} years: {sorted(test_years)})")
        print()

    # Normalization stats are fold-specific so k-fold hindcast runs cannot leak.
    stats_path = get_norm_stats_path(Config)

    if is_main_process():
        stats_ok = os.path.exists(stats_path)
        if stats_ok:
            try:
                stats_probe = np.load(stats_path)
                stats_ok = norm_stats_match_npz(stats_probe, Config)
                stats_probe.close()
                if not stats_ok:
                    print(
                        "  Existing normalization stats do not match the active "
                        f"{Config.NUM_GLOBAL_CHANNELS}-channel lagged global stack "
                        "or anomaly-target setting; recomputing."
                    )
            except Exception:
                stats_ok = False
        if not stats_ok:
            print("  Rank 0: Calculating Z-Score statistics (cross-val train set)...")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats = get_normalization_stats(tmp_dataset)
            save_dict = {
                'hi_mean': float(norm_stats['hi_mean']), 'hi_std': float(norm_stats['hi_std']),
                'stats_mean': norm_stats['stats_mean'].numpy(), 'stats_std': norm_stats['stats_std'].numpy(),
                'cond_mean': norm_stats['cond_mean'].numpy(), 'cond_std': norm_stats['cond_std'].numpy(),
                'topo_mean': float(norm_stats['topo_mean']), 'topo_std': float(norm_stats['topo_std']),
                'toa_mean': float(norm_stats['toa_mean']), 'toa_std': float(norm_stats['toa_std']),
                'lead_time': int(Config.LEAD_TIME),
                'target_half_window': 0,
                'global_lag_days': np.array(Config.GLOBAL_LAG_DAYS, dtype=np.int16),
                'train_on_climatology_anomalies': int(Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES),
                'cv_stride': int(Config.CV_STRIDE),
                'cv_val_offsets': np.array(Config.CV_VAL_OFFSETS, dtype=np.int16),
                'cv_test_offsets': np.array(Config.CV_TEST_OFFSETS, dtype=np.int16),
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

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=ddp
    )

    train_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                   normalization_stats=norm_stats, shared_data=shared_data,
                                   target_climatology=tac_climatology)
    val_dataset = ClimateDataset(Config, mode="val", val_indices=val_indices,
                                 normalization_stats=norm_stats, shared_data=shared_data,
                                 target_climatology=tac_climatology)
    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data,
                                  target_climatology=tac_climatology)

    if is_main_process() and len(train_dataset) > 0:
        sample_t = int(train_dataset.indices[0])
        print(
            "Direct lead check: model inputs use t, t-1, t-2; "
            f"global fields use lags {Config.GLOBAL_LAG_DAYS}; "
            f"target uses t+{Config.LEAD_TIME} "
            f"(first train sample t={sample_t}, target_t={sample_t + Config.LEAD_TIME})."
        )

    if is_main_process() and checkpoint_path is None:
        baseline_samples = min(Config.NUM_VALIDATION_SAMPLES, len(val_dataset))
        print(f"Computing persistence baseline (day-0 as day-15 forecast, n={baseline_samples})...")
        persistence_metrics = compute_persistence_baseline(val_dataset, conus_mask, n_samples=baseline_samples)
        print(f"  Persistence baseline: R2={persistence_metrics['r2']:.4f}, "
              f"corr={persistence_metrics['correlation']:.4f}, "
              f"CRPS={persistence_metrics['crps']:.4f}")

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
        print(f"  Mode: {'DETERMINISTIC (direct 15-day)' if Config.DETERMINISTIC else 'PROBABILISTIC (CFM)'}")

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
    best_tac = -999.0
    early_stop_best = -float("inf")
    early_stop_failures = 0

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
        best_tac = checkpoint.get("best_tac", best_tac)
        if checkpoint.get("early_stop_metric", Config.EARLY_STOP_METRIC) == Config.EARLY_STOP_METRIC:
            early_stop_best = checkpoint.get("early_stop_best", early_stop_best)
            early_stop_failures = checkpoint.get("early_stop_failures", early_stop_failures)

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
        stop_training = False
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_extreme_loss = 0.0
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

            if epoch == 0 and batch_idx == 0:
                deterministic_channels = (
                    x_t.shape[1] + x_tm1.shape[1] + x_tm2.shape[1] + spatial_c.shape[1]
                )
                if Config.DETERMINISTIC and deterministic_channels != Config.NUM_SPATIAL_CONDITIONS:
                    raise RuntimeError(
                        "Deterministic input channel mismatch: "
                        f"got {deterministic_channels}, expected {Config.NUM_SPATIAL_CONDITIONS}."
                    )
                if global_fields.shape[1] != Config.NUM_GLOBAL_CHANNELS:
                    raise RuntimeError(
                        f"Global channel mismatch: got {global_fields.shape[1]}, "
                        f"expected {Config.NUM_GLOBAL_CHANNELS}."
                    )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, components = compute_loss(
                    model, fm, y, x_t, x_tm1, x_tm2,
                    spatial_c, vec_c, global_fields, mask,
                    deterministic=Config.DETERMINISTIC)

                # Extreme loss fine-tuning (Lopez-Gomez et al., 2023)
                ext_loss_val = 0.0
                if Config.USE_EXTREME_LOSS:
                    # Use prediction already computed by compute_loss (no second forward pass)
                    pred_for_ext = components.get('pred', None)

                    if pred_for_ext is not None:
                        ext_loss = exponential_extreme_loss(
                            pred_for_ext, y, mask,
                            a=Config.EXTREME_LOSS_HEAT_BIAS,
                            b=Config.EXTREME_LOSS_COLD_BIAS,
                        )
                        w = Config.EXTREME_LOSS_WEIGHT
                        loss = (1 - w) * loss + w * ext_loss
                        ext_loss_val = ext_loss.item()

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
                    f.write(f"Direct 15-day prediction mode\n")
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
            epoch_extreme_loss += ext_loss_val
            n_good_batches += 1

            if is_main_process():
                postfix = {"loss": f"{loss.item():.4f}"}
                if Config.USE_EXTREME_LOSS:
                    postfix["ext"] = f"{ext_loss_val:.4f}"
                pbar.set_postfix(postfix)

            if Config.MAX_TRAIN_BATCHES is not None and (batch_idx + 1) >= Config.MAX_TRAIN_BATCHES:
                if is_main_process():
                    print(f"  Stopping epoch early after {Config.MAX_TRAIN_BATCHES} batches.")
                break

        if n_good_batches == 0 and is_main_process():
            print(f"\n  *** ALERT: Epoch {epoch+1} - ALL batches had NaN/Inf loss! ***")

        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            torch.cuda.empty_cache()
            gc.collect()

            if ddp:
                for param in ema.ema.parameters():
                    dist.broadcast(param.data, src=0)

            val_mse, val_rmse, val_ssim, improved_metrics = calculate_validation_metrics_cfm(
                ema.ema, val_dataset, device, conus_mask,
                n_samples=Config.NUM_VALIDATION_SAMPLES,
                rank=rank, world_size=world_size, ddp=ddp,
                tac_climatology=tac_climatology if is_main_process() else None
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
                if Config.USE_EXTREME_LOSS:
                    avg_ext = epoch_extreme_loss / max(n_good_batches, 1)
                    print(f"  Extreme Loss:      {avg_ext:.6f}")
                print(f"  Validation MSE:    {val_mse:.6f}")
                print(f"  Validation SSIM:   {val_ssim:.4f}")
                print(f"  Variance Ratio:    {improved_metrics['variance_ratio']:.4f}")
                print(f"  Gradient Ratio:    {improved_metrics['gradient_ratio']:.4f}")
                print(f"  Extreme Bias:      {improved_metrics['extreme_bias']:.4f}")
                print(f"  Correlation:       {improved_metrics['correlation']:.4f}")
                print(f"  R2:                {improved_metrics['r2']:.4f}")
                print(f"  Persistence R2:    {improved_metrics['persistence_r2']:.4f}")
                print(f"  Zero Baseline R2:  {improved_metrics['zero_r2']:.4f}")
                print(f"  MSE Skill vs Pers: {improved_metrics['mse_skill_vs_persistence']:.4f}")
                print(f"  Spatial Anom R2:   {improved_metrics['spatial_anom_r2']:.4f}")
                print(f"  Pers Spatial R2:   {improved_metrics['persistence_spatial_anom_r2']:.4f}")
                print(f"  TAC:               {improved_metrics['tac']:.4f}")
                print(f"  Persistence TAC:   {improved_metrics['persistence_tac']:.4f}")
                print(f"  MAE:               {improved_metrics['mae']:.4f}")
                print(f"  CRPS:              {improved_metrics['crps']:.4f}")
                print(f"  LR:                {scheduler.get_last_lr()[0]:.2e}")

                val_r2 = improved_metrics["r2"]
                val_tac = improved_metrics.get("tac", float("nan"))
                monitor_score = early_stop_score(
                    Config.EARLY_STOP_METRIC, improved_metrics, val_mse, val_ssim
                )
                monitor_value = early_stop_display_value(Config.EARLY_STOP_METRIC, monitor_score)
                best_monitor_value = early_stop_display_value(Config.EARLY_STOP_METRIC, early_stop_best)
                monitor_valid = np.isfinite(monitor_score)
                monitor_improved = (
                    monitor_valid
                    and monitor_score > early_stop_best + Config.EARLY_STOP_MIN_DELTA
                )
                if monitor_improved:
                    early_stop_best = monitor_score
                    early_stop_failures = 0
                    best_monitor_value = early_stop_display_value(
                        Config.EARLY_STOP_METRIC, early_stop_best
                    )
                elif (
                    Config.USE_VALIDATION_PATIENCE
                    and monitor_valid
                    and (epoch + 1) >= Config.EARLY_STOP_MIN_EPOCH
                ):
                    early_stop_failures += 1

                if Config.USE_VALIDATION_PATIENCE and monitor_valid:
                    print(
                        f"  Early-stop monitor ({Config.EARLY_STOP_METRIC}): "
                        f"value={monitor_value:.4f}, best={best_monitor_value:.4f}, "
                        f"failures={early_stop_failures}/{Config.EARLY_STOP_PATIENCE}"
                    )

                raw_model = model.module if ddp else model
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_ssim": val_ssim,
                    "val_r2": val_r2,
                    "val_tac": val_tac,
                    "best_ssim": max(best_ssim, val_ssim),
                    "best_r2": max(best_r2, val_r2),
                    "best_tac": max(best_tac, val_tac) if np.isfinite(val_tac) else best_tac,
                    "early_stop_metric": Config.EARLY_STOP_METRIC,
                    "early_stop_best": early_stop_best,
                    "early_stop_failures": early_stop_failures,
                }
                torch.save(ckpt, os.path.join(Config.CHECKPOINT_DIR,
                                               f"checkpoint_epoch_{epoch+1:04d}.pth"))
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    torch.save(ckpt, Config.MODEL_SAVE_PATH)
                    print(f"  New best model saved (R2={val_r2:.4f})")
                if np.isfinite(val_tac) and val_tac > best_tac:
                    best_tac = val_tac
                    torch.save(ckpt, Config.TAC_MODEL_SAVE_PATH)
                    print(f"  New best TAC model saved (TAC={val_tac:.4f})")
                if (
                    Config.USE_VALIDATION_PATIENCE
                    and monitor_valid
                    and (epoch + 1) >= Config.EARLY_STOP_MIN_EPOCH
                    and early_stop_failures >= Config.EARLY_STOP_PATIENCE
                ):
                    stop_training = True
                    print(
                        "  Early stopping: "
                        f"{Config.EARLY_STOP_METRIC} did not improve for "
                        f"{early_stop_failures} validation checks."
                    )

        if ddp:
            stop_tensor = torch.tensor(float(stop_training), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item() > 0.5)
            dist.barrier()

        if stop_training:
            break

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
    parser = argparse.ArgumentParser(description='MeshFlowNet Direct 15-Day Prediction')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'visualize', 'resume', 'export_hindcast'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_size', type=int, default=20)
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic mode (default is already deterministic)')
    parser.add_argument('--extreme_loss', action='store_true',
                       help='Enable exponential extreme loss for fine-tuning')
    parser.add_argument('--extreme_weight', type=float, default=0.3,
                       help='Weight for extreme loss blending')
    parser.add_argument('--dry_run', action='store_true',
                       help='Run a tiny one-GPU smoke/proxy pass: 1 epoch, 2 train batches, 4 validation samples.')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs.')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override per-GPU batch size.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                       help='Override validation/checkpoint frequency in epochs.')
    parser.add_argument('--num_validation_samples', type=int, default=None,
                       help='Override number of validation samples.')
    parser.add_argument('--max_train_batches', type=int, default=None,
                       help='Stop each epoch after this many train batches.')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                       help='Stop after this many validation checks without monitor improvement.')
    parser.add_argument('--early_stop_metric', type=str, default=None,
                       choices=['tac', 'r2', 'mse_skill', 'spatial_anom_r2', 'ssim', 'val_mse'],
                       help='Validation metric used for patience; larger is better except val_mse is internally negated.')
    parser.add_argument('--early_stop_min_epoch', type=int, default=None,
                       help='Do not count patience failures before this epoch.')
    parser.add_argument('--early_stop_min_delta', type=float, default=None,
                       help='Minimum monitor improvement required to reset patience.')
    parser.add_argument('--disable_early_stop', action='store_true',
                       help='Disable validation-patience early stopping.')
    parser.add_argument('--mesh_level', type=int, default=None,
                       help='Override icosahedral mesh refinement level.')
    parser.add_argument('--mesh_rounds', type=int, default=None,
                       help='Override mesh processor rounds.')
    parser.add_argument('--cv_stride', type=int, default=None,
                       help='Cross-validation stride for leave-k-years-out hindcasts.')
    parser.add_argument('--val_offset', type=str, default=None,
                       help='Comma-separated CV validation offsets, e.g. 3 or 1,2.')
    parser.add_argument('--test_offset', type=str, default=None,
                       help='Comma-separated CV test/hindcast offsets, e.g. 4 or 3,4.')
    parser.add_argument('--cv_fold', type=int, default=None,
                       help='Convenience fold id: test offset=fold, validation offset=(fold+1) mod stride.')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name used to isolate checkpoints, plots, and exported hindcast stats.')
    parser.add_argument('--hindcast_splits', type=str, default='test',
                       help='Comma-separated held-out splits to export in export_hindcast mode: val,test.')
    parser.add_argument('--hindcast_output_dir', type=str, default=None,
                       help='Directory for compact hindcast TAC-stat files.')
    parser.add_argument('--max_hindcast_samples', type=int, default=None,
                       help='Optional smoke-test cap for export_hindcast samples per split.')

    args = parser.parse_args()

    if args.dry_run:
        Config.MAX_EPOCHS = 1
        Config.CHECKPOINT_FREQ = 1
        Config.NUM_VALIDATION_SAMPLES = 4
        Config.BATCH_SIZE = 1
        Config.MAX_TRAIN_BATCHES = 2
        Config.WARMUP_EPOCHS = 1

    if args.epochs is not None:
        Config.MAX_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.checkpoint_freq is not None:
        Config.CHECKPOINT_FREQ = args.checkpoint_freq
    if args.num_validation_samples is not None:
        Config.NUM_VALIDATION_SAMPLES = args.num_validation_samples
    if args.max_train_batches is not None:
        Config.MAX_TRAIN_BATCHES = args.max_train_batches
    if args.early_stop_patience is not None:
        Config.EARLY_STOP_PATIENCE = args.early_stop_patience
    if args.early_stop_metric is not None:
        Config.EARLY_STOP_METRIC = args.early_stop_metric
    if args.early_stop_min_epoch is not None:
        Config.EARLY_STOP_MIN_EPOCH = args.early_stop_min_epoch
    if args.early_stop_min_delta is not None:
        Config.EARLY_STOP_MIN_DELTA = args.early_stop_min_delta
    if args.disable_early_stop:
        Config.USE_VALIDATION_PATIENCE = False
    if args.mesh_level is not None:
        Config.MESH_REFINEMENT_LEVEL = args.mesh_level
    if args.mesh_rounds is not None:
        Config.MESH_PROCESSOR_ROUNDS = args.mesh_rounds
    if args.cv_stride is not None:
        Config.CV_STRIDE = args.cv_stride
    if args.cv_fold is not None:
        fold = int(args.cv_fold) % int(Config.CV_STRIDE)
        Config.CV_TEST_OFFSETS = (fold,)
        Config.CV_VAL_OFFSETS = ((fold + 1) % int(Config.CV_STRIDE),)
        if args.run_name is None:
            args.run_name = f"cvfold{fold}"
    if args.val_offset is not None:
        parsed = parse_cv_offsets(args.val_offset, Config.CV_STRIDE)
        Config.CV_VAL_OFFSETS = parsed if parsed is not None else Config.CV_VAL_OFFSETS
    if args.test_offset is not None:
        parsed = parse_cv_offsets(args.test_offset, Config.CV_STRIDE)
        Config.CV_TEST_OFFSETS = parsed if parsed is not None else Config.CV_TEST_OFFSETS
    if args.hindcast_output_dir is not None:
        Config.HINDCAST_STATS_DIR = args.hindcast_output_dir
    apply_run_name(args.run_name)
    if args.sampling_steps is not None:
        Config.CFM_SAMPLING_STEPS = args.sampling_steps
    if args.ensemble:
        Config.ENSEMBLE_MODE = True
        Config.ENSEMBLE_SIZE = args.ensemble_size
    if args.deterministic:
        Config.DETERMINISTIC = True
    if args.extreme_loss:
        Config.USE_EXTREME_LOSS = True
        Config.EXTREME_LOSS_WEIGHT = args.extreme_weight

    apply_extended_global_fields()
    print_config_banner()
    if args.dry_run and is_main_process():
        print("Dry run: 1 epoch, batch size 1, 2 train batches, 4 validation samples.")
        print("This checks wiring/finite loss/global channels; it is not a skill estimate.")

    if args.mode in ('train', 'resume'):
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        _train_worker(rank, world_size, args)
    elif args.mode == 'export_hindcast':
        _export_hindcast(args)
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


def _export_hindcast(args):
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size > 1
    if ddp:
        setup_ddp(rank, world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    conus_mask = load_conus_mask(Config).to(device)
    shared_data = prepare_shared_data(Config, rank=rank, world_size=world_size, ddp=ddp)
    time_values = np.array(shared_data['time_values'])
    runs = detect_continuous_runs(time_values)
    all_valid = build_valid_indices(
        runs,
        lead_time=Config.LEAD_TIME,
        min_history=required_input_history(Config),
    )
    train_indices, val_indices, test_indices, train_years, val_years, test_years = \
        build_crossval_split(all_valid, time_values)

    if is_main_process():
        print("\n" + "=" * 80)
        print("EXPORT CROSS-VALIDATED HINDCAST TAC STATS")
        print("=" * 80)
        print(f"CV split: {cv_split_tag(Config)}")
        print(f"Train years: {len(train_years)}")
        print(f"Val years:   {sorted(val_years)}")
        print(f"Test years:  {sorted(test_years)}")

    stats_path = get_norm_stats_path(Config)
    stats_ok = os.path.exists(stats_path)
    if is_main_process() and stats_ok:
        try:
            probe = np.load(stats_path)
            stats_ok = norm_stats_match_npz(probe, Config)
            probe.close()
        except Exception:
            stats_ok = False
    if ddp:
        flag = torch.tensor(float(stats_ok), device=device)
        dist.broadcast(flag, src=0)
        stats_ok = bool(flag.item() > 0.5)

    if not stats_ok:
        if is_main_process():
            print(f"Missing/stale fold-specific normalization stats; recomputing: {stats_path}")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats_tmp = get_normalization_stats(tmp_dataset)
            save_dict = {
                'hi_mean': float(norm_stats_tmp['hi_mean']),
                'hi_std': float(norm_stats_tmp['hi_std']),
                'stats_mean': norm_stats_tmp['stats_mean'].numpy(),
                'stats_std': norm_stats_tmp['stats_std'].numpy(),
                'cond_mean': norm_stats_tmp['cond_mean'].numpy(),
                'cond_std': norm_stats_tmp['cond_std'].numpy(),
                'topo_mean': float(norm_stats_tmp['topo_mean']),
                'topo_std': float(norm_stats_tmp['topo_std']),
                'toa_mean': float(norm_stats_tmp['toa_mean']),
                'toa_std': float(norm_stats_tmp['toa_std']),
                'lead_time': int(Config.LEAD_TIME),
                'target_half_window': 0,
                'global_lag_days': np.array(Config.GLOBAL_LAG_DAYS, dtype=np.int16),
                'train_on_climatology_anomalies': int(Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES),
                'cv_stride': int(Config.CV_STRIDE),
                'cv_val_offsets': np.array(Config.CV_VAL_OFFSETS, dtype=np.int16),
                'cv_test_offsets': np.array(Config.CV_TEST_OFFSETS, dtype=np.int16),
            }
            if norm_stats_tmp['global_mean'] is not None:
                save_dict['global_mean'] = norm_stats_tmp['global_mean'].numpy()
                save_dict['global_std'] = norm_stats_tmp['global_std'].numpy()
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            np.savez(stats_path, **save_dict)
            del tmp_dataset
            gc.collect()
    if ddp:
        dist.barrier()
    norm_stats = load_norm_stats_npz(stats_path)

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=ddp
    )

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

    checkpoint_path = args.checkpoint or Config.MODEL_SAVE_PATH
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict'))
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no model_state_dict or ema_state_dict")
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    datasets = {
        "val": ClimateDataset(Config, mode="val", val_indices=val_indices,
                              normalization_stats=norm_stats, shared_data=shared_data,
                              target_climatology=tac_climatology),
        "test": ClimateDataset(Config, mode="test", test_indices=test_indices,
                               normalization_stats=norm_stats, shared_data=shared_data,
                               target_climatology=tac_climatology),
    }
    selected_splits = [s.strip() for s in args.hindcast_splits.split(",") if s.strip()]
    for split_name in selected_splits:
        if split_name not in datasets:
            raise ValueError(f"Unknown hindcast split '{split_name}'. Use val,test.")
        output_name = (
            f"hindcast_tac_stats_{Config.RUN_NAME or cv_split_tag(Config)}_{split_name}.npz"
        )
        output_path = os.path.join(Config.HINDCAST_STATS_DIR, output_name)
        export_hindcast_tac_stats(
            model,
            datasets[split_name],
            split_name,
            output_path,
            device,
            conus_mask,
            tac_climatology,
            rank=rank,
            world_size=world_size,
            ddp=ddp,
            max_samples=args.max_hindcast_samples,
        )

    if ddp:
        cleanup_ddp()


def _test(args):
    device = torch.device("cuda:0")
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

    time_values = np.array(shared_data['time_values'])
    runs = detect_continuous_runs(time_values)
    all_valid = build_valid_indices(
        runs,
        lead_time=Config.LEAD_TIME,
        min_history=required_input_history(Config),
    )

    train_indices, _, test_indices, _, _, _ = build_crossval_split(all_valid, time_values)

    stats_path = get_norm_stats_path(Config)
    s = np.load(stats_path)
    if not norm_stats_match_npz(s, Config):
        raise RuntimeError(
            f"Normalization stats at {stats_path} do not match the active "
            f"{Config.NUM_GLOBAL_CHANNELS}-channel lagged global stack or anomaly target. Re-run training setup "
            "or delete stale norm_stats_direct15_*.npz files for this fold."
        )
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

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=False
    )
    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data,
                                  target_climatology=tac_climatology)

    h, w = Config.IMAGE_SIZE

    model.eval()
    all_preds, all_truth = [], []

    for dataset_idx in tqdm(range(len(test_dataset)), desc="Test (direct 15-day)"):
        batch = test_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                                  spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)

        pred_cpu = pred[0, 0, :h, :w].cpu()
        truth_cpu = y[0, :h, :w].cpu()
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
            target_doy = int(test_dataset.doy_values[target_time_idx])
            climo = torch.from_numpy(np.asarray(tac_climatology[target_doy], dtype=np.float32))
            pred_cpu = pred_cpu + climo
            truth_cpu = truth_cpu + climo
        all_preds.append(pred_cpu)
        all_truth.append(truth_cpu)

    predictions = torch.stack(all_preds).unsqueeze(1)
    ground_truth = torch.stack(all_truth).unsqueeze(1)

    mask_2d = conus_mask[:h, :w].cpu()
    metrics = calculate_improved_metrics(predictions, ground_truth, mask=mask_2d)

    print(f"\n{'='*80}")
    print("TEST SET METRICS (direct 15-day prediction)")
    print(f"{'='*80}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
