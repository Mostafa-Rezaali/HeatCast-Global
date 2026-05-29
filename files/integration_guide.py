"""
================================================================================
Integration Guide: Replacing FlowUNet with MeshFlowNet
================================================================================

This file shows the EXACT changes to your main training script.
The CFM framework, dataset, loss function, and sampling are UNCHANGED.
Only the backbone model and its construction differ.

Three things change:
  1. Mesh is built ONCE at startup (expensive, but cached)
  2. FlowUNet() -> MeshFlowNet() with mesh attached
  3. DDP wrapping needs find_unused_parameters=False (already set)

Everything else (dataset, loss, sampling, validation, plotting) stays identical.
================================================================================
"""

# ===========================================================================
# CHANGE 1: Add imports at top of your main script
# ===========================================================================

# Add these after your existing imports:
from icosahedral_mesh import IcosahedralMesh, grid_to_flat, flat_to_grid
from mesh_backbone import MeshFlowNet, build_mesh_flow_net, count_parameters


# ===========================================================================
# CHANGE 2: Add mesh config to your Config class
# ===========================================================================

# Add these to class Config:
#
#   MESH_REFINEMENT_LEVEL = 5   # 5 -> ~2,500 regional nodes, 6 -> ~10,000
#   MESH_PROCESSOR_ROUNDS = 8   # message passing rounds (GraphCast uses 16)
#   MESH_LATENT_DIM = 256       # hidden dim for GNN layers
#   MESH_BUFFER_DEG = 5.0       # extra degrees around CONUS for mesh nodes
#   K_GRID2MESH = 3             # neighbors per grid point in encoder
#   K_MESH2GRID = 3             # neighbors per grid point in decoder
#   CONUS_LAT_RANGE = (25.0, 50.0)
#   CONUS_LON_RANGE = (-130.0, -60.0)


# ===========================================================================
# CHANGE 3: Build mesh once in train_model(), BEFORE model creation
# ===========================================================================

def build_mesh_once(config, conus_mask, device, ddp=False):
    """
    Build the icosahedral mesh. This is expensive (~30 seconds for level 5)
    but only happens once. The mesh object is shared across all epochs.
    
    Call this AFTER loading the CONUS mask but BEFORE creating the model.
    """
    import pickle, os
    
    cache_path = os.path.join(config.OUTPUT_DIR, "data_cache", 
                               f"mesh_level{config.MESH_REFINEMENT_LEVEL}.pkl")
    
    # Build on rank 0, load on all ranks
    if not ddp or dist.get_rank() == 0:
        if os.path.exists(cache_path):
            print(f"Loading cached mesh from {cache_path}")
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            # Need lat/lon arrays matching your grid
            grid_lat = np.linspace(
                config.CONUS_LAT_RANGE[0], 
                config.CONUS_LAT_RANGE[1], 
                config.IMAGE_SIZE[0]
            )
            grid_lon = np.linspace(
                config.CONUS_LON_RANGE[0], 
                config.CONUS_LON_RANGE[1], 
                config.IMAGE_SIZE[1]
            )
            
            # Convert mask to numpy if needed
            mask_np = conus_mask.cpu().numpy() if isinstance(conus_mask, torch.Tensor) else conus_mask
            
            mesh = IcosahedralMesh(
                refinement_level=config.MESH_REFINEMENT_LEVEL,
                lat_range=config.CONUS_LAT_RANGE,
                lon_range=config.CONUS_LON_RANGE,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                land_mask=mask_np,
                buffer_deg=config.MESH_BUFFER_DEG,
                k_grid2mesh=config.K_GRID2MESH,
                k_mesh2grid=config.K_MESH2GRID,
            )
            
            # Cache for future runs
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(mesh, f)
            print(f"Cached mesh to {cache_path}")
    
    if ddp:
        dist.barrier()
        
        if dist.get_rank() != 0:
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)
    
    # Move graph tensors to GPU
    mesh.to_torch(device)
    
    return mesh


# ===========================================================================
# CHANGE 4: Replace model creation in train_model()
# ===========================================================================

# BEFORE (your current code):
#
#   model = FlowUNet(
#       img_channels=Config.IMAGE_CHANNELS,
#       spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
#       condition_dim=Config.CONDITION_DIM,
#       base_dim=Config.BASE_DIM,
#       dim_mults=Config.DIM_MULTS,
#       dropout_rate=Config.DROPOUT_RATE,
#       num_global_channels=Config.NUM_GLOBAL_CHANNELS,
#       global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
#   ).to(device)

# AFTER:
#
#   # Build mesh (once, cached)
#   mesh = build_mesh_once(Config, conus_mask, device, ddp=ddp)
#
#   model = MeshFlowNet(
#       img_channels=Config.IMAGE_CHANNELS,
#       spatial_cond_channels=Config.NUM_SPATIAL_CONDITIONS,
#       condition_dim=Config.CONDITION_DIM,
#       latent_dim=Config.MESH_LATENT_DIM,
#       hidden_dim=Config.MESH_LATENT_DIM * 2,
#       num_processor_rounds=Config.MESH_PROCESSOR_ROUNDS,
#       mesh=mesh,
#       image_size=Config.IMAGE_SIZE,
#       num_global_channels=Config.NUM_GLOBAL_CHANNELS,
#       global_encoder_dim=Config.GLOBAL_ENCODER_DIM,
#   ).to(device)
#
#   if is_main_process():
#       count_parameters(model)


# ===========================================================================
# CHANGE 5: Update generate_cfm_sample to pass mesh
# ===========================================================================

# The model.forward() signature is IDENTICAL:
#   v_pred = model(x_input, t, cond, global_fields=global_fields)
#
# So cfm_loss() needs NO changes.
# generate_cfm_sample() needs NO changes.
# 
# The mesh is stored inside the model (self._mesh), so it's automatically
# available during forward passes. The model handles:
#   - Flattening grid to land-only nodes
#   - Running the GNN
#   - Scattering back to grid
#   - Padding to match U-Net dimensions
#
# All existing code that calls model() works as-is.


# ===========================================================================
# CHANGE 6: DDP wrapping (minor)
# ===========================================================================

# When wrapping with DDP, the mesh tensors are NOT model parameters.
# They're just data stored on the model. DDP handles this fine.
# Your existing find_unused_parameters=False is correct.
#
# IMPORTANT: After DDP wrapping, access the mesh via model.module._mesh
# For non-DDP: model._mesh


# ===========================================================================
# WHAT STAYS THE SAME
# ===========================================================================

# - ClimateDataset: identical (still returns (B, C, H, W) grids)
# - cfm_loss(): identical (model() has same signature)
# - generate_cfm_sample(): identical
# - calculate_validation_metrics_cfm(): identical
# - save_validation_plots(): identical
# - EMA: identical
# - Optimizer, scheduler: identical
# - All visualization code: identical


# ===========================================================================
# PERFORMANCE NOTES
# ===========================================================================

# 1. MEMORY: The mesh GNN uses far less memory than the U-Net for your
#    621x1405 grid because:
#    - U-Net processes the full H*W grid at every layer
#    - GNN processes only ~2,500 mesh nodes (refinement level 5)
#    - Memory scales with N_mesh, not H*W
#
# 2. SPEED: Each forward pass is faster because message passing on
#    ~2,500 nodes with ~15,000 edges is much cheaper than convolutions
#    on 621x1405 feature maps. The bottleneck shifts to grid<->mesh
#    scattering.
#
# 3. BATCH SIZE: You can likely increase from 4 to 8 or 16 with the
#    same GPU memory budget.
#
# 4. REFINEMENT LEVEL TRADEOFF:
#    Level 4: ~700 regional nodes   -> fast, possibly too coarse
#    Level 5: ~2,500 regional nodes -> good balance (recommended start)
#    Level 6: ~10,000 regional nodes -> higher fidelity, slower
#
# 5. PROCESSOR ROUNDS:
#    8 rounds  -> ~8 hops across mesh -> good for regional (~3000km range)
#    12 rounds -> broader effective range, more parameters
#    16 rounds -> GraphCast default (global model, full planet)
#
#    For CONUS with 15-day lead time, 8-10 rounds should suffice since
#    the mesh is regional and teleconnection info comes from global_fields.


# ===========================================================================
# MESH + GLOBAL CONTEXT INTERACTION
# ===========================================================================

# The global context integration changes philosophy:
#
# U-Net version:  Global fields -> CNN -> cross-attention at bottleneck
#                 (single injection point, gated)
#
# Mesh version:   Global fields -> CNN -> bilinear sample at EVERY mesh node
#                 (per-node injection, no gate needed)
#
# This is more natural because each mesh node gets global context based on
# its geographic position. A mesh node over the Gulf of Mexico receives
# local SST directly. No gate is needed because the processor GNN learns
# to integrate local and global information across message passing rounds.
#
# You removed the gate from the U-Net version for the right reason
# (teleconnections should dominate at 15-day lead). The mesh architecture
# makes this natural by construction.


# ===========================================================================
# QUICK TEST
# ===========================================================================

if __name__ == "__main__":
    import torch
    import numpy as np
    
    # Minimal test with fake data
    class FakeConfig:
        IMAGE_SIZE = (621, 1405)
        IMAGE_CHANNELS = 1
        NUM_SPATIAL_CONDITIONS = 15
        CONDITION_DIM = 5
        NUM_GLOBAL_CHANNELS = 5
        GLOBAL_ENCODER_DIM = 64
        MESH_REFINEMENT_LEVEL = 4  # small for testing
        MESH_PROCESSOR_ROUNDS = 4
        MESH_LATENT_DIM = 128
        MESH_BUFFER_DEG = 5.0
        K_GRID2MESH = 3
        K_MESH2GRID = 3
        CONUS_LAT_RANGE = (25.0, 50.0)
        CONUS_LON_RANGE = (-130.0, -60.0)
        OUTPUT_DIR = "/tmp/mesh_test"
        WINDOW_SIZE = 64
    
    config = FakeConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build mesh
    grid_lat = np.linspace(25.0, 50.0, 621)
    grid_lon = np.linspace(-130.0, -60.0, 1405)
    mask = np.ones((621, 1405), dtype=np.float32)  # all land for test
    
    mesh = IcosahedralMesh(
        refinement_level=config.MESH_REFINEMENT_LEVEL,
        lat_range=config.CONUS_LAT_RANGE,
        lon_range=config.CONUS_LON_RANGE,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        land_mask=mask,
        buffer_deg=config.MESH_BUFFER_DEG,
    )
    mesh.to_torch(device)
    
    # Build model
    model = MeshFlowNet(
        img_channels=config.IMAGE_CHANNELS,
        spatial_cond_channels=config.NUM_SPATIAL_CONDITIONS,
        condition_dim=config.CONDITION_DIM,
        latent_dim=config.MESH_LATENT_DIM,
        hidden_dim=config.MESH_LATENT_DIM * 2,
        num_processor_rounds=config.MESH_PROCESSOR_ROUNDS,
        mesh=mesh,
        image_size=config.IMAGE_SIZE,
        num_global_channels=config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=config.GLOBAL_ENCODER_DIM,
    ).to(device)
    
    count_parameters(model)
    
    # Padded dimensions
    from icosahedral_mesh import IcosahedralMesh
    H, W = config.IMAGE_SIZE
    pad_h = (config.WINDOW_SIZE - H % config.WINDOW_SIZE) % config.WINDOW_SIZE
    pad_w = (config.WINDOW_SIZE - W % config.WINDOW_SIZE) % config.WINDOW_SIZE
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    # Fake forward pass
    B = 2
    x = torch.randn(B, 1 + 15, H_pad, W_pad, device=device)
    t = torch.rand(B, device=device)
    cond = torch.randn(B, 5, device=device)
    g = torch.randn(B, 5, 181, 360, device=device)
    
    with torch.no_grad():
        v = model(x, t, cond, global_fields=g)
    
    print(f"\nInput:  {x.shape}")
    print(f"Output: {v.shape}")
    print(f"Expected: ({B}, {config.IMAGE_CHANNELS}, {H_pad}, {W_pad})")
    assert v.shape == (B, config.IMAGE_CHANNELS, H_pad, W_pad)
    print("\nForward pass successful!")
