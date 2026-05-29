#!/usr/bin/env python3
"""
================================================================================
Conditional Flow Matching (CFM) - Optimal Transport Path
================================================================================
Physics-Guided Heat Wave Forecasting with DETERMINISTIC ODE SOLVER
+ GLOBAL CONTEXT ENCODER (Multi-Resolution Cross-Attention)

Key features:
- Uses Optimal Transport (OT) displacement path: z_t = (1-t)*z_0 + t*z_1
- Model predicts the velocity field v_t pointing towards the target heat index.
- Replaces complex VDM gamma schedules with uniform time t in [0, 1].
- Uses deterministic Euler integration for sampling.
- NEW: Global context encoder processes 181x360 teleconnection fields
  and injects via cross-attention at the U-Net bottleneck.
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

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

# datetime
from datetime import datetime, timedelta

# ======================================================================================
# Logger
# ======================================================================================
import logging

def setup_spike_logger(output_dir):
    """Create a logger that writes spike warnings to a separate file."""
    logger = logging.getLogger("spike_warnings")
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # don't echo to root/stdout
    
    # Remove existing handlers (in case of re-init)
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
# CFM COMPONENTS - FIXED LINEAR SCHEDULE (Reference-Aligned)
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
    """
    Conditional Flow Matching for Forecasting
    
    Interpolation: x_flow = (1 - t) * x_0 + t * x_1
      where x_0 = current state (x_t), x_1 = target (y)
    
    Velocity target: v = x_1 - x_0
    """
    
    def sample_xt(self, x_0, x_1, t):
        """Interpolate between current state (x_0) and target (x_1) at time t."""
        t = t.view(-1, 1, 1, 1)
        x_flow = (1 - t) * x_0 + t * x_1
        return x_flow
    
    def velocity_target(self, x_0, x_1):
        """The target velocity field is the residual change: v = y - x_t."""
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
    # Variable names as they appear in the NetCDF, in channel order
    # Two have pressure_level dim (size 1) that gets squeezed automatically
    GLOBAL_VARIABLES = [
        'u_wind_200',                  # jet stream (has pressure_level dim)
        'total_column_water_vapour',   # moisture transport
        'geopotential_200',            # upper-level ridges (has pressure_level dim)
        'olr',                         # outgoing longwave radiation
        'sst',                         # sea surface temperature
    ]
    GLOBAL_SIZE = (181, 360)       # lat x lon
    NUM_GLOBAL_CHANNELS = 5
    
    # ==================== MODEL ARCHITECTURE ====================
    IMAGE_SIZE = (621, 1405)
    IMAGE_CHANNELS = 1
    # Input: [flow(1), x_t(1), x_tm1(1), x_tm2(1), physics(9), topo(1), lat(1), lon(1)] = 15 channels
    NUM_SPATIAL_CONDITIONS = 15
    LEAD_TIME = 15  # because target uses t+1
    CONDITION_DIM = 5
    
    BASE_DIM = 64
    DIM_MULTS = (1, 2, 4, 8)
    DROPOUT_RATE = 0.3
    
    # Global encoder config
    GLOBAL_ENCODER_DIM = 256        # feature dim inside global encoder
    GLOBAL_BOTTLENECK_DIM = 1024    # must match U-Net bottleneck = BASE_DIM * DIM_MULTS[-1]
    
    BATCH_SIZE = 4
    OCEAN_FILL = 0  # Represents the mean in z-score space
    # ==================== CFM SCHEDULE (Reference-Aligned) ====================
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
    NUM_VALIDATION_SAMPLES = 50
    WARMUP_EPOCHS = 50
    
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"\n{'='*80}")
    print(f"CFM TRAINING - CONDITIONAL FLOW MATCHING + GLOBAL CONTEXT")
    print(f"{'='*80}")
    print(f"Using device: {Config.DEVICE}")
    print(f"Sampling Steps: {Config.CFM_SAMPLING_STEPS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
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
# IMPROVED METRICS
# ======================================================================================
@torch.inference_mode()
def compute_persistence_baseline(val_dataset, mask, n_samples=200):
    """
    Persistence forecast: predict x_{t+1} = x_t
    This is the baseline your model must beat.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    h, w = Config.IMAGE_SIZE
    
    # Prepare global mask
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
        # Unpack - now includes global_fields
        (y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, _, batch_mask) = batch
        # persistence: predict x_{t+1} = x_t (no change)
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
    """Calculate metrics that detect over-smoothing and loss of spatial detail."""
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
    
    # 1. Spatial variance ratio
    if mask_np is not None:
        pred_spatial_std = np.nanstd(pred_masked, axis=(1, 2)).mean()
        truth_spatial_std = np.nanstd(truth_masked, axis=(1, 2)).mean()
    else:
        pred_spatial_std = np.std(pred_np, axis=(1, 2)).mean()
        truth_spatial_std = np.std(truth_np, axis=(1, 2)).mean()
    
    variance_ratio = pred_spatial_std / (truth_spatial_std + 1e-8)
    
    # 2. Gradient magnitude ratio
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
    
    # 3. Extreme value bias
    if mask_np is not None:
        valid_pred = pred_masked[~np.isnan(pred_masked)]
        valid_truth = truth_masked[~np.isnan(truth_masked)]
        truth_p95 = np.percentile(valid_truth, 95)
        pred_p95 = np.percentile(valid_pred, 95)
    else:
        truth_p95 = np.percentile(truth_np, 95)
        pred_p95 = np.percentile(pred_np, 95)
    
    extreme_bias = pred_p95 - truth_p95
    
    # 4. Correlation
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
    
    # 5. R²
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
# MODEL ARCHITECTURE
# ======================================================================================

class ContinuousTimeEmbedding(nn.Module):
    """Embed continuous time t in [0,1] for CFM."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time * 1000.0
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim, dropout_rate):
        super().__init__()
        # Project to 2x out_channels to create both a SCALE and a SHIFT vector
        self.time_mlp = nn.Linear(time_emb_dim, out_channels * 2)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels * 2)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, c):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 1. Calculate combined embedding
        time_emb = self.time_mlp(t)
        cond_emb = self.cond_mlp(c)
        emb = time_emb + cond_emb
        
        # 2. Split into scale (gamma) and shift (beta)
        scale, shift = emb.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # 3. Apply AdaGN: h = h * (1 + scale) + shift
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

class GlobalAttentionBlock(nn.Module):
    """Self-attention for CONUS features."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        
        # Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(x_norm)  # Shape: (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # Each is (B, C, H, W)
        
        # Reshape for multi-head attention: (Batch, Heads, Sequence_Length, Head_Dim)
        head_dim = C // self.num_heads
        seq_len = H * W
        
        q = q.view(B, self.num_heads, head_dim, seq_len).transpose(-1, -2) 
        k = k.view(B, self.num_heads, head_dim, seq_len).transpose(-1, -2)
        v = v.view(B, self.num_heads, head_dim, seq_len).transpose(-1, -2)
        
        # PyTorch 2.0 Native Flash Attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back to image format: (B, C, H, W)
        attn_output = attn_output.transpose(-1, -2).reshape(B, C, H, W)
        
        # Final projection and residual connection
        out = self.proj(attn_output)
        
        return x + out


# ======================================================================================
# GLOBAL CONTEXT ENCODER + CROSS-ATTENTION
# ======================================================================================

class GlobalContextEncoder(nn.Module):
    """
    Lightweight CNN encoder for global-scale (181x360) teleconnection fields.
    
    Input:  (B, NUM_GLOBAL_CHANNELS, 181, 360)
    Output: (B, bottleneck_dim, Hg, Wg) where Hg~12, Wg~23
    
    These tokens become keys/values for cross-attention at the CONUS bottleneck.
    
    Architecture: 4 strided conv blocks with GroupNorm + SiLU.
    181x360 -> 91x180 -> 46x90 -> 23x45 -> 12x23 (stride 2 each)
    """
    def __init__(self, in_channels, encoder_dim, bottleneck_dim, time_emb_dim):
        super().__init__()
        
        # Time conditioning for the global encoder too
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, bottleneck_dim),
            nn.SiLU(),
        )
        
        # 4-block encoder: in_channels -> encoder_dim -> 2*encoder_dim -> 4*encoder_dim -> bottleneck_dim
        self.blocks = nn.ModuleList([
            self._make_block(in_channels, encoder_dim, stride=2),         # 181x360 -> 91x180
            self._make_block(encoder_dim, encoder_dim * 2, stride=2),     # 91x180 -> 46x90
            self._make_block(encoder_dim * 2, encoder_dim * 4, stride=2), # 46x90 -> 23x45
            self._make_block(encoder_dim * 4, bottleneck_dim, stride=2),  # 23x45 -> 12x23
        ])
        
        # Final norm before outputting tokens
        self.final_norm = nn.GroupNorm(8, bottleneck_dim)
        self.final_act = nn.SiLU()
    
    def _make_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(),
        )
    
    def forward(self, global_fields, time_emb):
        """
        Args:
            global_fields: (B, NUM_GLOBAL_CHANNELS, 181, 360)
            time_emb: (B, time_emb_dim) - already computed time embedding
        Returns:
            global_tokens: (B, bottleneck_dim, Hg, Wg) - spatial feature map
        """
        x = global_fields
        
        for block in self.blocks:
            x = block(x)
        
        # Add time conditioning as a channel-wise bias
        t = self.time_proj(time_emb)  # (B, bottleneck_dim)
        x = x + t[:, :, None, None]
        
        x = self.final_norm(x)
        x = self.final_act(x)
        
        return x  # (B, bottleneck_dim, ~12, ~23)


class GlobalCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # CONUS queries
        self.norm_q = nn.GroupNorm(8, channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Global keys and values
        self.norm_kv = nn.GroupNorm(8, channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # === THE GATE ===
        # Hardcoded dynamic gate (controlled by the training loop)
        # Defaults to 1.0 so inference/testing uses 100% global context
        self.gate_strength = 1.0

    def forward(self, conus_features, global_tokens):
        B, C, Hc, Wc = conus_features.shape
        _, _, Hg, Wg = global_tokens.shape
        
        q = self.q_proj(self.norm_q(conus_features))
        
        kv_normed = self.norm_kv(global_tokens)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)
        
        q = q.view(B, self.num_heads, self.head_dim, Hc * Wc).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim, Hg * Wg).transpose(-1, -2)
        v = v.view(B, self.num_heads, self.head_dim, Hg * Wg).transpose(-1, -2)
        
        # Flash attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, Hc, Wc)
        attn_out = self.out_proj(attn_out)
        
        # === APPLY GATING ===
        return conus_features + (self.gate_strength * attn_out)


# ======================================================================================
# U-NET BLOCKS
# ======================================================================================

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim, dropout_rate):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, cond_emb_dim, dropout_rate)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, t, c):
        x = self.res(x, t, c)
        return x, self.downsample(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim, dropout_rate):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.res = ResidualBlock(out_channels * 2, out_channels, time_emb_dim, cond_emb_dim, dropout_rate)

    def forward(self, x, skip_x, t, c):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res(x, t, c)
        return x

# ======================================================================================
# FLOW U-NET WITH GLOBAL CONTEXT
# ======================================================================================

class FlowUNet(nn.Module):
    def __init__(self, img_channels, spatial_cond_channels, condition_dim, base_dim,
                 dim_mults, dropout_rate,
                 num_global_channels=0, global_encoder_dim=256):
        super().__init__()
        
        self.use_global = num_global_channels > 0

        self.init_conv = nn.Conv2d(img_channels + spatial_cond_channels, base_dim, kernel_size=3, padding=1)

        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            ContinuousTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, base_dim * 4)
        )

        dims = [base_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_blocks = nn.ModuleList([
            DownBlock(in_c, out_c, time_dim, base_dim * 4, dropout_rate)
            for in_c, out_c in in_out
        ])

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim, base_dim * 4, dropout_rate)
        self.mid_attn = GlobalAttentionBlock(mid_dim, num_heads=4)
        
        # === NEW: Global context encoder + cross-attention ===
        if self.use_global:
            self.global_encoder = GlobalContextEncoder(
                in_channels=num_global_channels,
                encoder_dim=global_encoder_dim,
                bottleneck_dim=mid_dim,  # must match U-Net bottleneck
                time_emb_dim=time_dim,
            )
            self.global_cross_attn = GlobalCrossAttention(mid_dim, num_heads=8)
        
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim, base_dim * 4, dropout_rate)

        self.up_blocks = nn.ModuleList()
        curr_dim = dims[-1]
        for skip_dim in reversed(dims[1:]):
            self.up_blocks.append(UpBlock(curr_dim, skip_dim, time_dim, base_dim * 4, dropout_rate))
            curr_dim = skip_dim

        self.final_res_block = ResidualBlock(base_dim * 2, base_dim, time_dim, base_dim * 4, dropout_rate)
        self.final_conv = nn.Conv2d(base_dim, img_channels, kernel_size=1)

    def forward(self, x, t, cond, global_fields=None):
        """
        Args:
            x: (B, img_channels + spatial_cond_channels, H, W) - concatenated input
            t: (B,) - time in [0, 1]
            cond: (B, condition_dim) - teleconnection indices
            global_fields: (B, NUM_GLOBAL_CHANNELS, 181, 360) or None
        """
        t_emb = self.time_mlp(t)
        c = self.cond_mlp(cond)

        x = self.init_conv(x)
        initial_skip = x.clone()

        skip_connections = []
        for down_block in self.down_blocks:
            skip_x, x = down_block(x, t_emb, c)
            skip_connections.append(skip_x)

        x = self.mid_block1(x, t_emb, c)
        x = self.mid_attn(x)
        
        # === NEW: Inject global context via cross-attention ===
        if self.use_global and global_fields is not None:
            global_tokens = self.global_encoder(global_fields, t_emb)
            x = self.global_cross_attn(x, global_tokens)
        
        x = self.mid_block2(x, t_emb, c)

        for up_block, skip_x in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip_x, t_emb, c)

        x = torch.cat((x, initial_skip), dim=1)
        x = self.final_res_block(x, t_emb, c)

        return self.final_conv(x)


# ======================================================================================
# DATASET - NOW WITH GLOBAL FIELDS
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
            # Global data not available in fallback
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
            # New CONUS variables
            self.temperature_2m = shared_data['temperature_2m']
            self.specific_humidity_850 = shared_data['specific_humidity_850']
            self.temperature_850 = shared_data['temperature_850']
            self.u_wind_850 = shared_data['u_wind_850']
            self.v_wind_850 = shared_data['v_wind_850']
            self.geopotential_300 = shared_data['geopotential_300']
            # Global data (mmap)
            self.global_data = shared_data.get('global_data', None)
            self.n_timesteps = self.heat_index.shape[-1]

        # stats
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

            # Topography stats
            if self.topography is not None:
                topo_land = self.topography[self.topography != 0.0]
                self.topo_mean = torch.tensor(float(np.mean(topo_land)), dtype=torch.float32)
                self.topo_std  = torch.tensor(float(np.std(topo_land)),  dtype=torch.float32)
                print(f"    Topography -> Mean: {self.topo_mean:.4f}, Std: {self.topo_std:.4f}")

            # Global field stats (per-channel mean/std over training times)
            if self.global_data is not None:
                global_means = []
                global_stds = []
                for var_name, var_data in self.global_data.items():
                    # var_data shape: (lat, lon, time) - select training times
                    var_train = var_data[:, :, train_t_indices]
                    global_means.append(float(np.mean(var_train)))
                    global_stds.append(float(np.std(var_train)))
                    print(f"    Global {var_name} -> Mean: {global_means[-1]:.4f}, Std: {global_stds[-1]:.4f}")
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
        # Create normalized coordinates from -1 to 1
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

        # 1. Load raw slices
        x_t_slice      = self.heat_index[:, :, t]
        x_tm1_slice    = self.heat_index[:, :, t - 1]
        x_tm2_slice    = self.heat_index[:, :, t - 2] # New lag
        x_target_slice = self.heat_index[:, :, t + self.config.LEAD_TIME]

        # Create land mask from current timestep
        raw_x_t = torch.from_numpy(x_t_slice.copy())
        land_mask = (raw_x_t != 0.0).float().unsqueeze(0)

        # 2. Helper for consistent Z-score normalization & ocean masking
        def normalize_hi(data_slice):
            tensor = torch.from_numpy(data_slice.copy())
            normed = (tensor - self.hi_mean) / (self.hi_std + 1e-8)
            normed = normed.unsqueeze(0) # Add channel dim
            # Apply land mask: fill ocean with Config.OCEAN_FILL (usually 0)
            return normed * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        # Process all heat index fields
        x_t   = normalize_hi(x_t_slice)
        x_tm1 = normalize_hi(x_tm1_slice)
        x_tm2 = normalize_hi(x_tm2_slice)
        y     = normalize_hi(x_target_slice)

        # 3. Scalar Teleconnection Indices
        cond_slice = self.cond_train[:, t]
        vec_c = (torch.from_numpy(cond_slice.copy()) - self.cond_mean) / (self.cond_std + 1e-8)

        # 4. Local Physics (ERA5 variables)
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

        # Static Topography
        topo = torch.from_numpy(self.topography.copy())
        if topo.shape != (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]):
            topo = topo.T
        topo = ((topo - self.topo_mean) / (self.topo_std + 1e-8)).unsqueeze(0)
        topo = topo * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        
        # Coordinate Grids
        lat_c = self.lat_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        lon_c = self.lon_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        
        # Concatenate ERA5 Physics + Topography + Lat + Lon
        physics = torch.cat([physics, topo, lat_c, lon_c], dim=0) # Total 12 channels

        # 5. Global Fields (181x360)
        if self.global_data is not None and self.global_mean is not None:
            global_channels = []
            for var_name, var_data in self.global_data.items():
                g_slice = torch.from_numpy(np.array(var_data[:, :, t], dtype=np.float32))
                global_channels.append(g_slice)
            global_fields = torch.stack(global_channels, dim=0)
            global_fields = (global_fields - self.global_mean) / (self.global_std + 1e-8)
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        # 6. Padding for U-Net (Ensures dimensions are divisible by 64/32)
        y     = F.pad(y,     self.padding, "constant", Config.OCEAN_FILL)
        x_t   = F.pad(x_t,   self.padding, "constant", Config.OCEAN_FILL)
        x_tm1 = F.pad(x_tm1, self.padding, "constant", Config.OCEAN_FILL)
        x_tm2 = F.pad(x_tm2, self.padding, "constant", Config.OCEAN_FILL) # Pad the new lag
        physics = F.pad(physics, self.padding, "constant", Config.OCEAN_FILL)
        mask    = F.pad(land_mask, self.padding, "constant", 0)

        # Return tuple expanded to include x_tm2
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
    # Include global data if present
    if dataset.global_data is not None:
        stats['shared_data']['global_data'] = dataset.global_data
    stats['global_mean'] = dataset.global_mean
    stats['global_std'] = dataset.global_std
    return stats

# ======================================================================================
# CFM LOSS (Reference-Aligned)
# ======================================================================================
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

def masked_ssim_01(pred, target, mask, fill=0.5):
    mask = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    pred_m = pred * mask + fill * (1 - mask)
    targ_m = target * mask + fill * (1 - mask)
    return structural_similarity_index_measure(pred_m, targ_m, data_range=1.0)

    
def sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """
    Logit-normal time sampling (Stable Diffusion 3).
    Concentrates samples around t=0.5 where the learning signal is strongest.
    """
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


def cfm_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask,
             return_components=True):
    """
    Conditional Flow Matching loss.
    
    y = x_{t+lead} (target, what we're forecasting)
    condition on x_t, x_{t-1}, spatial_c, vec_c, global_fields
    """
    batch_size = y.shape[0]
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    times = sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0)
    times = times.clamp(1e-5, 1.0 - 1e-5)

    # Flow directly from today's weather (x_t) to the target (y)
    x_t_flow = fm.sample_xt(x_0=x_t, x_1=y, t=times)
    x_t_flow = x_t_flow * mask + Config.OCEAN_FILL * (1 - mask)

    v_target = fm.velocity_target(x_0=x_t, x_1=y)

    x_input = torch.cat([x_t_flow, x_t, x_tm1, x_tm2, spatial_c], dim=1)
    v_pred = model(x_input, times, vec_c, global_fields=global_fields)

    loss_per_pixel = (v_pred - v_target) ** 2 * mask
    loss_per_sample = loss_per_pixel.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    total_loss = loss_per_sample.mean()

    if return_components:
        t_view = times.view(-1, 1, 1, 1)
        y_recon = x_t_flow + (1 - t_view) * v_pred

        recon_mse = ((y_recon - y) ** 2 * mask).sum(dim=(1, 2, 3))
        recon_mse = (recon_mse / (mask.sum(dim=(1, 2, 3)) + 1e-8)).mean()

        return total_loss, {
            "cfm_loss": total_loss.detach(),
            "recon_mse": recon_mse.detach(),
        }

    return total_loss

# ======================================================================================
# OCEAN MASKING UTILITIES
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
# SAMPLING (ODE Integration with Global Context)
# ======================================================================================
@torch.inference_mode()
def generate_cfm_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2, global_fields, device,
                        h, w, pad_h, pad_w, mask, n_steps=20):
    """
    Generate a sample by integrating the learned velocity field from t=0 to t=1.
    Now includes global_fields for cross-attention context.
    """
    model.eval()

    # Start integration from today's heat index (x_t), not random noise
    z = x_t.clone()
    z = z * mask + Config.OCEAN_FILL * (1 - mask)
    
    dt = 1.0 / n_steps
    VAL_MIN, VAL_MAX = -4.0, 4.0

    for i in range(n_steps):
        t_i = torch.tensor([i * dt], device=device)
        t_next = torch.tensor([(i + 1) * dt], device=device).clamp(max=1.0)
    
        # Heun's method step 1
        x_input = torch.cat([z, x_t, x_tm1, x_tm2, spatial_c], dim=1)
        v1 = model(x_input, t_i.expand(1), vec_c, global_fields=global_fields)
        v1 = v1 * mask
        
        z_euler = z + v1 * dt
        z_euler = z_euler.clamp(VAL_MIN, VAL_MAX)
        z_euler = z_euler * mask + Config.OCEAN_FILL * (1 - mask)
    
        # Heun's method step 2
        x_input2 = torch.cat([z_euler, x_t, x_tm1, x_tm2, spatial_c], dim=1)
        v2 = model(x_input2, t_next.expand(1), vec_c, global_fields=global_fields)
        v2 = v2 * mask
    
        # Heun update
        z = z + (v1 + v2) * 0.5 * dt
        z = z.clamp(VAL_MIN, VAL_MAX)
        z = z * mask + Config.OCEAN_FILL * (1 - mask)
    
    return z[0, 0, :h, :w].cpu().numpy()

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
    val_loader = DataLoader(rank_subset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)

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

        # === GENCAST FIX: SMALL VALIDATION ENSEMBLE ===
        val_ensemble_size = 4 
        member_preds = []
        
        for _ in range(val_ensemble_size):
            pred = generate_cfm_sample(
                model,
                spatial_c, vec_c, x_t, x_tm1, x_tm2,
                global_fields,
                device, h, w, pad_h, pad_w,
                mask=batch_mask.to(device),
                n_steps=Config.CFM_SAMPLING_STEPS,
            )
            member_preds.append(pred)
            
        # Take the mean of the generative samples to smooth noise and boost R²
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
    avg_ssim = masked_ssim_01(preds_ssim, truth_ssim,
                               all_masks.to(device), fill=0.5).item()

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


@torch.inference_mode()
def generate_test_predictions_cfm(model, test_dataset, device, mask=None):
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    if isinstance(model, nn.DataParallel):
        print("  Unwrapping DataParallel for inference")
        model = model.module
    model = model.to(device)

    print("\n" + "=" * 80)
    print("GENERATING TEST PREDICTIONS - CFM (ODE Solver) + GLOBAL CONTEXT")
    print("=" * 80)
    print(f"Sampling steps: {Config.CFM_SAMPLING_STEPS}")
    print(f"Mode: {'ENSEMBLE' if Config.ENSEMBLE_MODE else 'SINGLE'}")

    h, w = Config.IMAGE_SIZE
    pad_h, pad_w = calculate_padding(h, w, Config.WINDOW_SIZE)

    all_predictions = []
    all_ground_truth = []
    all_ensemble_stats = [] if Config.ENSEMBLE_MODE else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np_global = mask[:h, :w].detach().cpu().numpy()
        else:
            mask_np_global = np.array(mask[:h, :w])
    else:
        mask_np_global = None

    for batch in tqdm(test_loader, desc="Generating"):
        (y, x_t, x_tm1, spatial_c, vec_c, global_fields, _, batch_mask) = batch
        y = y.to(device, non_blocking=True)
        x_t = x_t.to(device, non_blocking=True)
        x_tm1 = x_tm1.to(device, non_blocking=True)
        x_tm2 = x_tm2.to(device, non_blocking=True)
        spatial_c = spatial_c.to(device, non_blocking=True)
        vec_c = vec_c.to(device, non_blocking=True)
        global_fields = global_fields.to(device, non_blocking=True)
        b_size = y.shape[0]

        truth_batch = y[:, 0, :h, :w].detach().cpu().numpy()
        all_ground_truth.append(truth_batch)

        if Config.ENSEMBLE_MODE:
            batch_forecasts = []
            batch_ensemble_stats = []

            for b_idx in range(b_size):
                ensemble_members = []
                batch_mask_padded = batch_mask.to(device, non_blocking=True)
                for _ in range(Config.ENSEMBLE_SIZE):
                    pred = generate_cfm_sample(
                        model,
                        spatial_c[b_idx:b_idx+1],
                        vec_c[b_idx:b_idx+1],
                        x_t[b_idx:b_idx+1],
                        x_tm1[b_idx:b_idx+1],
                        x_tm2[b_idx:b_idx+1],
                        global_fields[b_idx:b_idx+1],
                        device, h, w, pad_h, pad_w,
                        mask=batch_mask_padded[b_idx:b_idx+1],
                        n_steps=Config.CFM_SAMPLING_STEPS
                    )
                    ensemble_members.append(pred)

                ensemble = np.stack(ensemble_members, axis=0)

                if mask_np_global is not None:
                    mask_np_single = mask_np_global
                else:
                    mask_np_single = batch_mask[b_idx, 0, :h, :w].cpu().numpy()

                stats = compute_ensemble_statistics(ensemble, mask=mask_np_single)
                batch_ensemble_stats.append(stats)
                batch_forecasts.append(stats["mean"])

            all_predictions.append(np.stack(batch_forecasts, axis=0))
            all_ensemble_stats.extend(batch_ensemble_stats)

        else:
            batch_forecasts = []
            batch_mask_padded = batch_mask.to(device, non_blocking=True)
            for b_idx in range(b_size):
                pred = generate_cfm_sample(
                    model,
                    spatial_c[b_idx:b_idx+1],
                    vec_c[b_idx:b_idx+1],
                    x_t[b_idx:b_idx+1],
                    x_tm1[b_idx:b_idx+1],
                    x_tm2[b_idx:b_idx+1],
                    global_fields[b_idx:b_idx+1],
                    device, h, w, pad_h, pad_w,
                    mask=batch_mask_padded[b_idx:b_idx+1],
                    n_steps=Config.CFM_SAMPLING_STEPS
                )
                batch_forecasts.append(pred)

            all_predictions.append(np.stack(batch_forecasts, axis=0))

    predictions = torch.cat([torch.from_numpy(p) for p in all_predictions], dim=0)
    ground_truth = torch.cat([torch.from_numpy(g) for g in all_ground_truth], dim=0)

    print(f"\n Generated {len(predictions)} test predictions")

    if Config.ENSEMBLE_MODE:
        return predictions, ground_truth, all_ensemble_stats
    return predictions, ground_truth



# ======================================================================================
# VISUALIZATION
# ======================================================================================
def save_validation_plots(model, val_dataset, device, mask, epoch, save_dir, n_samples=5):
    """Plot validation predictions vs ground truth during training."""
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

        pred = generate_cfm_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields,
            device, h, w, pad_h, pad_w,
            mask=batch_mask.to(device),
            n_steps=Config.CFM_SAMPLING_STEPS
        )

        truth = y[0, 0, :h, :w].cpu().numpy()

        pred_display = np.where(mask_np[:h, :w] > 0.5, pred, np.nan)
        truth_display = np.where(mask_np[:h, :w] > 0.5, truth, np.nan)
        diff_display = np.where(mask_np[:h, :w] > 0.5, pred - truth, np.nan)

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


def save_test_prediction_plots(predictions, ground_truth, dataset, save_dir, mask=None, num_samples=20):
    print(f"\nGenerating {num_samples} test prediction plots...")
    os.makedirs(save_dir, exist_ok=True)
    
    h, w = Config.IMAGE_SIZE
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    if len(predictions) <= num_samples:
        sample_indices = list(range(len(predictions)))
    else:
        np.random.seed(42)
        sample_indices = sorted(np.random.choice(len(predictions), num_samples, replace=False))
    
    for plot_idx, data_idx in enumerate(tqdm(sample_indices, desc="Creating plots")):
        pred = predictions[data_idx, :h, :w] if predictions.ndim == 3 else predictions[data_idx, 0, :h, :w]
        truth = ground_truth[data_idx, :h, :w] if ground_truth.ndim == 3 else ground_truth[data_idx, 0, :h, :w]
        
        if mask_np is not None:
            valid = mask_np > 0.5
            pred_valid = pred[valid]
            truth_valid = truth[valid]
            mse = np.mean((truth_valid - pred_valid) ** 2)
            mae = np.mean(np.abs(truth_valid - pred_valid))
            rmse = np.sqrt(mse)
            corr = np.corrcoef(truth_valid, pred_valid)[0, 1] if truth_valid.std() > 1e-6 and pred_valid.std() > 1e-6 else 0.0
            ss_res = np.sum((truth_valid - pred_valid)**2)
            ss_tot = np.sum((truth_valid - truth_valid.mean())**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            var_ratio = np.std(pred_valid) / (np.std(truth_valid) + 1e-8)
            pred_display = np.where(mask_np > 0.5, pred, np.nan)
            truth_display = np.where(mask_np > 0.5, truth, np.nan)
        else:
            mse = np.mean((truth - pred) ** 2)
            mae = np.mean(np.abs(truth - pred))
            rmse = np.sqrt(mse)
            corr = np.corrcoef(truth.flatten(), pred.flatten())[0, 1]
            r2 = 1 - (np.sum((truth - pred)**2) / np.sum((truth - truth.mean())**2))
            var_ratio = np.std(pred) / (np.std(truth) + 1e-8)
            pred_display = pred
            truth_display = truth    
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        vmin, vmax = -3.0, 3.0
        
        im1 = axes[0].imshow(truth_display, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title(f'Ground Truth', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='Z-score')
        
        mask_label = " (LAND)" if mask_np is not None else ""
        axes[1].set_title(
            f'Prediction{mask_label}\nMAE={mae:.4f}, r={corr:.3f}, R²={r2:.3f}, VR={var_ratio:.3f}',
            fontsize=14, fontweight='bold'
        )
        im2 = axes[1].imshow(pred_display, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im2, ax=axes[1], label='Z-score')
        
        fig.suptitle(f'Test Sample {plot_idx + 1} - Index {data_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'test_{plot_idx+1:02d}_idx{data_idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(sample_indices)} plots to: {save_dir}")


# ======================================================================================
# TRAINING LOOP
# ======================================================================================
def save_first_xt_and_xtp1_plot(x_t, y, mask, out_dir, fname="train_first_xt_xtp1.png", vmin=-3.0, vmax=3.0):
    os.makedirs(out_dir, exist_ok=True)

    xt = x_t.detach().cpu()
    yy = y.detach().cpu()
    mm = mask.detach().cpu()

    if xt.ndim == 4: xt = xt[0,0]
    if yy.ndim == 4: yy = yy[0,0]
    if mm.ndim == 4: mm = mm[0,0]

    xtp1 = yy

    xt_m = np.where(mm.numpy() > 0.5, xt.numpy(), np.nan)
    xtp1_m = np.where(mm.numpy() > 0.5, xtp1.numpy(), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(16,10))

    im = axes[0,0].imshow(xt.numpy(), vmin=vmin, vmax=vmax, aspect="auto"); axes[0,0].set_title("x_t (today)"); plt.colorbar(im, ax=axes[0,0])
    im = axes[0,1].imshow(xtp1.numpy(), vmin=vmin, vmax=vmax, aspect="auto"); axes[0,1].set_title(f"Target (t+{Config.LEAD_TIME})"); plt.colorbar(im, ax=axes[0,1])
    im = axes[1,0].imshow(xt_m, vmin=vmin, vmax=vmax, aspect="auto"); axes[1,0].set_title("x_t masked"); plt.colorbar(im, ax=axes[1,0])
    im = axes[1,1].imshow(xtp1_m, vmin=vmin, vmax=vmax, aspect="auto"); axes[1,1].set_title(f"Target masked (t+{Config.LEAD_TIME})"); plt.colorbar(im, ax=axes[1,1])

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved x_t / x_{{t+1}} plot: {out_path}")


def save_first_io_plot(x_input, y_target, mask, out_dir, fname="first_input_target.png",
                       vmin=-3.0, vmax=3.0, channel_names=None):
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(x_input, np.ndarray):
        x_input = torch.from_numpy(x_input)
    if isinstance(y_target, np.ndarray):
        y_target = torch.from_numpy(y_target)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    x_input = x_input.detach().cpu()
    y_target = y_target.detach().cpu()
    mask = mask.detach().cpu()

    if x_input.ndim == 4:
        x_input = x_input[0]
    if y_target.ndim == 4:
        y_target = y_target[0, 0]
    elif y_target.ndim == 3:
        y_target = y_target[0]
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]

    x_np = x_input.numpy()
    y_np = y_target.numpy()
    m_np = mask.numpy()

    C = x_np.shape[0]

    if channel_names is None:
        channel_names = [f"input ch {i}" for i in range(C)]

    cols = 3
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        if i < C:
            im = ax.imshow(x_np[i], vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(channel_names[i])
            plt.colorbar(im, ax=ax)
        else:
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname.replace(".png", "_inputs.png"))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved multi-channel inputs plot: {out_path}")

    y_m = np.where(m_np > 0.5, y_np, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(y_np, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Target")
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(y_m, vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Target masked")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname.replace(".png", "_target.png"))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved target plot: {out_path}")


def save_mask_plot(mask, out_dir, fname="conus_mask.png"):
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)

    if m.ndim == 3:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")

    land_frac = float(np.mean(m))

    plt.figure(figsize=(12, 5))
    plt.imshow(m, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    plt.title(f"Land/Ocean Mask (land fraction = {land_frac:.3f})")
    plt.colorbar(label="1=land, 0=ocean")
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved mask plot: {out_path}")


def prepare_shared_data(config, rank, world_size, ddp):
    """Rank 0 saves data to .npy cache once; all ranks mmap it."""
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
    
    # === GLOBAL DATA PATHS ===
    global_cache_dir = os.path.join(cache_dir, "global")
    os.makedirs(global_cache_dir, exist_ok=True)
    
    global_paths = {}
    for var_name in config.GLOBAL_VARIABLES:
        global_paths[var_name] = os.path.join(global_cache_dir, f"{var_name}.npy")

    if is_main_process():
        # Check CONUS cache
        cache_ok = all(os.path.exists(p) for p in paths.values())

        if cache_ok:
            try:
                nc_mtime = max(
                    os.path.getmtime(config.TRAINING_DATA_PATH),
                    os.path.getmtime(TOPO_PATH),
                )
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
                np.save(paths["geopotential"],
                        np.array(nc.variables["geopotential"][:], dtype=np.float32))
                np.save(paths["soil_moisture"],
                        np.array(nc.variables["soil_moisture"][:], dtype=np.float32))
                np.save(paths["slp"],
                        np.array(nc.variables["sea_level_pressure"][:], dtype=np.float32))
                np.save(paths["cond_train"],
                        np.array(nc.variables["CondTrain"][:], dtype=np.float32))
                np.save(paths["temperature_2m"],
                        np.array(nc.variables["temperature_2m"][:], dtype=np.float32))
                np.save(paths["specific_humidity_850"],
                        np.array(nc.variables["specific_humidity_850"][:], dtype=np.float32))
                np.save(paths["temperature_850"],
                        np.array(nc.variables["temperature_850"][:], dtype=np.float32))
                np.save(paths["u_wind_850"],
                        np.array(nc.variables["u_wind_850"][:], dtype=np.float32))
                np.save(paths["v_wind_850"],
                        np.array(nc.variables["v_wind_850"][:], dtype=np.float32))
                np.save(paths["geopotential_300"],
                        np.array(nc.variables["geopotential_300"][:], dtype=np.float32))

            with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
                topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
                topo = np.flipud(topo)
            np.save(paths["topography"], topo)

            print(f"Rank 0: wrote CONUS data cache to {cache_dir}")
        else:
            print(f"Rank 0: using existing CONUS data cache in {cache_dir}")

        # === CACHE GLOBAL DATA ===
        global_cache_ok = all(os.path.exists(p) for p in global_paths.values())
        
        if not global_cache_ok:
            print(f"Rank 0: caching global teleconnection fields from {config.GLOBAL_DATA_PATH}...")
            if not os.path.exists(config.GLOBAL_DATA_PATH):
                raise FileNotFoundError(f"Global data file not found: {config.GLOBAL_DATA_PATH}")
            
            with NetCDFDataset(config.GLOBAL_DATA_PATH, "r") as nc_g:
                print(f"  Variables in file: {list(nc_g.variables.keys())}")
                
                for var_name in config.GLOBAL_VARIABLES:
                    if var_name not in nc_g.variables:
                        raise KeyError(f"Variable '{var_name}' not found in {config.GLOBAL_DATA_PATH}")
                    
                    data = np.array(nc_g.variables[var_name][:], dtype=np.float32)
                    print(f"  {var_name}: raw shape={data.shape}")
                    
                    # Squeeze pressure_level dimension if present
                    # e.g. (5831, 1, 181, 360) -> (5831, 181, 360)
                    if data.ndim == 4:
                        assert data.shape[1] == 1, \
                            f"{var_name}: expected pressure_level=1, got shape={data.shape}"
                        data = data[:, 0, :, :]
                        print(f"    Squeezed pressure_level -> {data.shape}")
                    
                    # Data is (time, lat, lon) from ERA5. Transpose to (lat, lon, time)
                    # for consistency with CONUS data layout.
                    assert data.ndim == 3, f"{var_name}: expected 3D after squeeze, got {data.ndim}D"
                    data = np.transpose(data, (1, 2, 0))  # (lat, lon, time)
                    
                    assert data.shape[0] == config.GLOBAL_SIZE[0], \
                        f"{var_name}: expected lat={config.GLOBAL_SIZE[0]}, got {data.shape[0]}"
                    assert data.shape[1] == config.GLOBAL_SIZE[1], \
                        f"{var_name}: expected lon={config.GLOBAL_SIZE[1]}, got {data.shape[1]}"
                    
                    # Fill NaN with 0 (SST has NaN over land, OLR can have fill values)
                    nan_count = np.isnan(data).sum()
                    if nan_count > 0:
                        print(f"    Filling {nan_count} NaN values with 0.0")
                        data = np.nan_to_num(data, nan=0.0)
                    
                    np.save(global_paths[var_name], data)
                    print(f"    Cached: {data.shape} -> {global_paths[var_name]}")
                    del data
            
            print(f"Rank 0: wrote global data cache to {global_cache_dir}")
        else:
            print(f"Rank 0: using existing global data cache in {global_cache_dir}")

    if ddp:
        dist.barrier()

    # Load CONUS data as mmap
    shared = {
        "heat_index":   np.load(paths["heat_index"],   mmap_mode="r"),
        "geopotential": np.load(paths["geopotential"], mmap_mode="r"),
        "soil_moisture":np.load(paths["soil_moisture"],mmap_mode="r"),
        "slp":          np.load(paths["slp"],          mmap_mode="r"),
        "cond_train":   np.load(paths["cond_train"],   mmap_mode="r"),
        "topography":   np.load(paths["topography"],   mmap_mode="r"),
        "temperature_2m":        np.load(paths["temperature_2m"],        mmap_mode="r"),
        "specific_humidity_850": np.load(paths["specific_humidity_850"], mmap_mode="r"),
        "temperature_850":       np.load(paths["temperature_850"],       mmap_mode="r"),
        "u_wind_850":            np.load(paths["u_wind_850"],            mmap_mode="r"),
        "v_wind_850":            np.load(paths["v_wind_850"],            mmap_mode="r"),
        "geopotential_300":      np.load(paths["geopotential_300"],      mmap_mode="r"),
    }
    
    # Load global data as mmap (ordered dict to preserve channel order)
    from collections import OrderedDict
    global_data = OrderedDict()
    for var_name in config.GLOBAL_VARIABLES:
        global_data[var_name] = np.load(global_paths[var_name], mmap_mode="r")
    shared['global_data'] = global_data
    
    if is_main_process():
        for var_key, arr in global_data.items():
            print(f"  Global {var_key}: shape={arr.shape}, mmap=True")
    
    return shared
    
torch.backends.cudnn.benchmark = True

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
        print("CONDITIONAL FLOW MATCHING (CFM) - DDP + GLOBAL CONTEXT")
        print(f"World size: {world_size}")
        print(f"Global channels: {Config.NUM_GLOBAL_CHANNELS} at {Config.GLOBAL_SIZE}")
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
        print(f"  Train: {len(train_indices)} samples (indices {train_indices[0]}..{train_indices[-1]})")
        print(f"  Val:   {len(val_indices)} samples (indices {val_indices[0]}..{val_indices[-1]})")
        print(f"  Test:  {len(test_indices)} samples (indices {test_indices[0]}..{test_indices[-1]})")
        print(f"  Buffer gaps: {Config.LEAD_TIME} days between each split\n")
    
    # Compute normalization stats
    stats_path = os.path.join(Config.OUTPUT_DIR, "data_cache", "norm_stats_v2.npz")  # v2 includes global

    if is_main_process():
        if not os.path.exists(stats_path):
            print("  Rank 0: Calculating Z-Score statistics...")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats = get_normalization_stats(tmp_dataset)
            
            save_dict = {
                'hi_mean':    float(norm_stats['hi_mean']),
                'hi_std':     float(norm_stats['hi_std']),
                'stats_mean': norm_stats['stats_mean'].numpy(),
                'stats_std':  norm_stats['stats_std'].numpy(),
                'cond_mean':  norm_stats['cond_mean'].numpy(),
                'cond_std':   norm_stats['cond_std'].numpy(),
                'topo_mean':  float(norm_stats['topo_mean']),
                'topo_std':   float(norm_stats['topo_std']),
            }
            if norm_stats['global_mean'] is not None:
                save_dict['global_mean'] = norm_stats['global_mean'].numpy()
                save_dict['global_std'] = norm_stats['global_std'].numpy()
            
            np.savez(stats_path, **save_dict)
            del tmp_dataset
            gc.collect()
            print(f"  Rank 0: saved normalization stats to {stats_path}")
        else:
            print(f"  Rank 0: Found existing normalization stats at {stats_path}")

    if ddp:
        dist.barrier()

    s = np.load(stats_path)
    norm_stats = {
        'hi_mean':    torch.tensor(float(s['hi_mean'])),
        'hi_std':     torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']),
        'stats_std':  torch.from_numpy(s['stats_std']),
        'cond_mean':  torch.from_numpy(s['cond_mean']),
        'cond_std':   torch.from_numpy(s['cond_std']),
        'topo_mean':  torch.tensor(float(s['topo_mean'])),
        'topo_std':   torch.tensor(float(s['topo_std'])),
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
        print(f"  Persistence baseline: R²={persistence_metrics['r2']:.4f}, "
              f"corr={persistence_metrics['correlation']:.4f}, "
              f"var_ratio={persistence_metrics['variance_ratio']:.4f}")
              
    # Model - NOW WITH GLOBAL CONTEXT
    model = FlowUNet(
        img_channels=Config.IMAGE_CHANNELS,
        spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
        condition_dim=Config.CONDITION_DIM,
        base_dim=Config.BASE_DIM,
        dim_mults=Config.DIM_MULTS,
        dropout_rate=Config.DROPOUT_RATE,
        num_global_channels=Config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
    ).to(device)
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        global_params = 0
        if model.use_global:
            global_params += sum(p.numel() for p in model.global_encoder.parameters())
            global_params += sum(p.numel() for p in model.global_cross_attn.parameters())
        print(f"\n  Total parameters: {total_params/1e6:.1f}M")
        print(f"  Global encoder + cross-attn: {global_params/1e6:.1f}M")
        print(f"  CONUS U-Net: {(total_params - global_params)/1e6:.1f}M\n")

    # Sampler + DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    effective_batch_size = Config.BATCH_SIZE * world_size
    if is_main_process():
        print(f"Per-GPU batch: {Config.BATCH_SIZE}, Effective batch: {effective_batch_size}")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.1)
    
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

    if checkpoint_path is not None:
        if is_main_process():
            print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        # Allow partial loading (new global encoder params won't be in old checkpoints)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if is_main_process() and missing:
            print(f"  New parameters (not in checkpoint): {len(missing)} keys")
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        start_epoch = checkpoint.get("epoch", 0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        best_r2 = checkpoint.get("best_r2", best_r2)

    # DDP wrapping
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema = EMA(model.module if ddp else model, decay=0.999)
    
    if checkpoint_path is not None and "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        ema_cleaned = {k.replace("module.", ""): v for k, v in ema_state.items()}
        # Also allow partial load for EMA
        ema.ema.load_state_dict(ema_cleaned, strict=False)
        if is_main_process():
            print("  Restored EMA state from checkpoint (partial OK)")

# Sanity check plot (rank 0 only)
    if is_main_process():
        batch = next(iter(train_loader))
        y_b, x_t_b, x_tm1_b, x_tm2_b, spatial_c_b, vec_c_b, global_b, idx_b, mask_b = batch
        save_first_io_plot(
            # 1. Added x_tm2_b[0] to the concatenation
            torch.cat([x_t_b[0], x_tm1_b[0], x_tm2_b[0], spatial_c_b[0]], dim=0),
            y_b[0], mask_b[0], Config.PLOTS_DIR,
            fname="train_first_input_target.png", vmin=-3.0, vmax=3.0,
            channel_names=[
            "x_t (HeatIndex today)",
            "x_{t-1} (HeatIndex yesterday)",
            "x_{t-2} (HeatIndex 2 days ago)", # <--- ADDED
            "ERA5: geopotential 500",
            "ERA5: soil_moisture",
            "ERA5: sea_level_pressure",
            "ERA5: 2m temperature",
            "ERA5: specific humidity 850",
            "ERA5: temperature 850",
            "ERA5: u-wind 850",
            "ERA5: v-wind 850",
            "ERA5: geopotential 300",
            "Topography (ETOPO2022)",
            "Latitude Grid",  # <--- ADDED
            "Longitude Grid", # <--- ADDED
        ],
        )
        save_first_xt_and_xtp1_plot(x_t_b[:1], y_b[:1], mask_b[:1], Config.PLOTS_DIR)
        
        # Also plot global fields for first sample
        g = global_b[0].numpy()  # (5, 181, 360)
        fig, axes = plt.subplots(1, 5, figsize=(25, 4))
        g_names = Config.GLOBAL_VARIABLES
        for i in range(5):
            im = axes[i].imshow(g[i], aspect='auto', cmap='RdBu_r')
            axes[i].set_title(f"Global: {g_names[i]}")
            plt.colorbar(im, ax=axes[i])
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, "global_fields_sample.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved global fields sample plot")
    
    # Sync scheduler on resume
    if start_epoch > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()

    # Training loop
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        # Calculate dynamic gate: 0.0 at epoch 0, 1.0 at epoch 100+
        current_gate = min(1.0, epoch / 100.0)
        
        # Inject the gate strength into the model and EMA
        raw_m = model.module if ddp else model
        if raw_m.use_global:
            raw_m.global_cross_attn.gate_strength = current_gate
            ema.ema.global_cross_attn.gate_strength = current_gate

        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_good_batches = 0
        consecutive_skips = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process(),
            mininterval=10.0)

        for batch_idx, batch in enumerate(pbar):
            # Added x_tm2 to the tuple unpacking
            (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, _, mask) = batch
            
            y = y.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            x_tm1 = x_tm1.to(device, non_blocking=True)
            x_tm2 = x_tm2.to(device, non_blocking=True) # Move new lag to GPU
            spatial_c = spatial_c.to(device, non_blocking=True)
            vec_c = vec_c.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
        
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, components = cfm_loss(
                    model, fm, y, x_t, x_tm1, x_tm2, # Pass x_tm2 here
                    spatial_c, vec_c, global_fields, mask)

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

        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            torch.cuda.empty_cache()
            gc.collect()

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
                print(f"\n  Training Loss:     {avg_loss:.6f}")
                print(f"  Validation MSE:    {val_mse:.6f}")
                print(f"  Validation SSIM:   {val_ssim:.4f}")
                print(f"  Variance Ratio:    {improved_metrics['variance_ratio']:.4f}")
                print(f"  Gradient Ratio:    {improved_metrics['gradient_ratio']:.4f}")
                print(f"  Extreme Bias:      {improved_metrics['extreme_bias']:.4f}")
                print(f"  Correlation:       {improved_metrics['correlation']:.4f}")
                print(f"  R²:                {improved_metrics['r2']:.4f}")
                print(f"  LR:                {scheduler.get_last_lr()[0]:.2e}")

                # === MODIFIED: Monitor the Gate Value ===
                raw_m = model.module if ddp else model
                if raw_m.use_global:
                    print(f"  [Epoch {epoch + 1}] Global context:    ACTIVE")
                    print(f"  [Epoch {epoch + 1}] Gate Strength:     {current_gate:.4f} (Scheduled)")

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
# SAVE TO NETCDF
# ======================================================================================
def save_predictions_to_netcdf(predictions, ground_truth, test_dataset, output_path, ensemble_stats=None):
    h, w = Config.IMAGE_SIZE
    
    if isinstance(predictions, torch.Tensor):
        predictions_norm = predictions[:, :h, :w].cpu().numpy() if predictions.ndim == 3 else predictions[:, 0, :h, :w].cpu().numpy()
    else:
        predictions_norm = predictions[:, :h, :w] if predictions.ndim == 3 else predictions[:, 0, :h, :w]
    
    if isinstance(ground_truth, torch.Tensor):
        ground_truth_norm = ground_truth[:, :h, :w].cpu().numpy() if ground_truth.ndim == 3 else ground_truth[:, 0, :h, :w].cpu().numpy()
    else:
        ground_truth_norm = ground_truth[:, :h, :w] if ground_truth.ndim == 3 else ground_truth[:, 0, :h, :w]
    
    nc_out = NetCDFDataset(output_path, 'w', format='NETCDF4')
    nc_out.createDimension('time', predictions_norm.shape[0])
    nc_out.createDimension('lat', h)
    nc_out.createDimension('lon', w)
    
    if ensemble_stats is not None:
        for name, long_name in [('ensemble_mean', 'Ensemble mean'), ('ensemble_std', 'Ensemble std'),
                                 ('ensemble_p10', '10th percentile'), ('ensemble_p90', '90th percentile')]:
            v = nc_out.createVariable(name, 'f4', ('time','lat','lon'), zlib=True, complevel=4)
            v.long_name = long_name
        for i, stats in enumerate(ensemble_stats):
            nc_out.variables['ensemble_mean'][i] = stats['mean']
            nc_out.variables['ensemble_std'][i] = stats['std']
            nc_out.variables['ensemble_p10'][i] = stats['p10']
            nc_out.variables['ensemble_p90'][i] = stats['p90']
        nc_out.ensemble_size = Config.ENSEMBLE_SIZE
    else:
        forecasts = nc_out.createVariable('Forecasts', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)
        forecasts[:] = predictions_norm
    
    truth = nc_out.createVariable('Truth', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)
    truth[:] = ground_truth_norm
    
    nc_out.description = f'CFM {Config.LEAD_TIME}-day-ahead heat wave forecasts (with global context)'
    nc_out.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nc_out.hi_mean = float(test_dataset.hi_mean)
    nc_out.hi_std = float(test_dataset.hi_std)
    
    nc_out.close()
    print(f"Saved {len(predictions_norm)} predictions to: {output_path}")


# ======================================================================================
# MAIN
# ======================================================================================
def _test(args):
    device = torch.device("cuda:0")
    conus_mask = load_conus_mask(Config)

    model = FlowUNet(
        img_channels=Config.IMAGE_CHANNELS,
        spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
        condition_dim=Config.CONDITION_DIM,
        base_dim=Config.BASE_DIM,
        dim_mults=Config.DIM_MULTS,
        dropout_rate=Config.DROPOUT_RATE,
        num_global_channels=Config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
    ).to(device)

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint['model_state_dict'])
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    fm = FlowMatching().to(device)

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

    if is_main_process():
        print(f"\nStrict Temporal Split (Lead Time = {Config.LEAD_TIME} days):")
        print(f"  Train: {len(train_indices)} samples (indices {train_indices[0]}..{train_indices[-1]})")
        print(f"  Val:   {len(val_indices)} samples (indices {val_indices[0]}..{val_indices[-1]})")
        print(f"  Test:  {len(test_indices)} samples (indices {test_indices[0]}..{test_indices[-1]})")

    train_dataset = ClimateDataset(Config, mode='train', train_indices=train_indices,
                                   shared_data=shared_data)
    norm_stats = get_normalization_stats(train_dataset)
    test_dataset = ClimateDataset(Config, mode='test', test_indices=test_indices,
                                  normalization_stats=norm_stats,
                                  shared_data=norm_stats['shared_data'])

    compute_persistence_baseline(test_dataset, conus_mask, n_samples=200)

    result = generate_test_predictions_cfm(model, test_dataset, device, mask=conus_mask)

    if Config.ENSEMBLE_MODE:
        predictions, ground_truth, ensemble_stats = result
    else:
        predictions, ground_truth = result
        ensemble_stats = None

    improved_metrics = calculate_improved_metrics(predictions, ground_truth, mask=conus_mask)
    print(f"\n{'='*80}")
    print("TEST SET METRICS")
    print(f"{'='*80}")
    for k, v in improved_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*80}\n")

    save_predictions_to_netcdf(predictions, ground_truth, test_dataset,
                               Config.OUTPUT_NC_FILE, ensemble_stats)
    save_test_prediction_plots(predictions, ground_truth, test_dataset,
                               Config.PLOTS_DIR, mask=conus_mask, num_samples=20)


def _visualize(args):
    conus_mask = load_conus_mask(Config)
    ncout = NetCDFDataset(Config.OUTPUT_NC_FILE, 'r')

    if 'Forecasts' in ncout.variables:
        predictions_np = np.array(ncout.variables['Forecasts'][:], dtype=np.float32)
    elif 'ensemble_mean' in ncout.variables:
        predictions_np = np.array(ncout.variables['ensemble_mean'][:], dtype=np.float32)

    groundtruth_np = np.array(ncout.variables['Truth'][:], dtype=np.float32)
    hi_mean = float(ncout.hi_mean)
    hi_std = float(ncout.hi_std)
    ncout.close()

    class DummyDataset:
        def __init__(self, hi_mean, hi_std):
            self.hi_mean = hi_mean
            self.hi_std = hi_std

    save_test_prediction_plots(
        torch.from_numpy(predictions_np), torch.from_numpy(groundtruth_np),
        DummyDataset(hi_mean, hi_std), Config.PLOTS_DIR, mask=conus_mask, num_samples=20
    )
    
def main():
    parser = argparse.ArgumentParser(description='CFM Heat Wave Forecasting + Global Context')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'visualize', 'resume'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_size', type=int, default=20)
    parser.add_argument('--sampling_steps', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.sampling_steps is not None:
        Config.CFM_SAMPLING_STEPS = args.sampling_steps
        
    if args.ensemble:
        Config.ENSEMBLE_MODE = True
        Config.ENSEMBLE_SIZE = args.ensemble_size

    if args.mode in ('train', 'resume'):
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        _train_worker(rank, world_size, args)
    elif args.mode == 'test':
        _test(args)
    elif args.mode == 'visualize':
        _visualize(args)

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
        result = generate_test_predictions_cfm(ema_model, test_dataset, device, mask=conus_mask)

        if Config.ENSEMBLE_MODE:
            predictions, ground_truth, ensemble_stats = result
        else:
            predictions, ground_truth = result
            ensemble_stats = None

        save_predictions_to_netcdf(predictions, ground_truth, test_dataset,
                                   Config.OUTPUT_NC_FILE, ensemble_stats)
        save_test_prediction_plots(predictions, ground_truth, test_dataset,
                                   Config.PLOTS_DIR, mask=conus_mask, num_samples=20)

if __name__ == "__main__":
    main()