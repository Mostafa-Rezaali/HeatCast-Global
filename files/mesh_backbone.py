"""
================================================================================
GraphCast-style GNN Backbone for CFM Heat Wave Forecasting
================================================================================

Supports two modes via the `deterministic` flag:

  DETERMINISTIC (GraphCast):
    - Input: [x_t, x_tm1, x_tm2, spatial_c] (no flow state)
    - t fixed at 0.5 (embedding becomes a learned bias)
    - Predicts residual: y - x_t
    - Single forward pass at inference

  PROBABILISTIC (GenCast/CFM):
    - Input: [x_flow, x_t, x_tm1, x_tm2, spatial_c]
    - t is the flow time in [0, 1]
    - Predicts velocity field v_t
    - ODE integration at inference

The encoder-processor-decoder backbone is IDENTICAL in both modes.
Only the input channel count differs (no flow channel in deterministic mode).
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from icosahedral_mesh import IcosahedralMesh, grid_to_flat, flat_to_grid


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time * 1000.0
        device = time.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000) / (half - 1))
        )
        args = time[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, residual=True):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = out + x
        return self.norm(out)


class InteractionNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, bipartite=False, sender_dim=None):
        super().__init__()
        s_dim = sender_dim if sender_dim is not None else node_dim
        self.bipartite = bipartite
        self.edge_mlp = MLP(edge_dim + s_dim + node_dim, hidden_dim, edge_dim, num_layers=2)
        # NOTE: residual inside MLP is disabled (in_dim != out_dim) since we concat
        # node+agg. We handle the residual EXPLICITLY below instead.
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim, num_layers=2, residual=False)

    def forward(self, sender_features, receiver_features, edge_index, edge_attr):
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = sender_features[src_idx]
        h_dst = receiver_features[dst_idx]

        updated_edges = self.edge_mlp(torch.cat([edge_attr, h_src, h_dst], dim=-1))

        # FIX: Force scatter_add_ to float32 to avoid bf16 accumulation errors
        # With ~42 messages per node, bf16 rounding kills gradient signal
        num_recv = receiver_features.shape[0]
        with torch.amp.autocast('cuda', enabled=False):
            updated_edges_f32 = updated_edges.float()
            agg = torch.zeros(num_recv, updated_edges_f32.shape[-1],
                              device=receiver_features.device, dtype=torch.float32)
            agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(updated_edges_f32), updated_edges_f32)
            agg = agg.to(receiver_features.dtype)

        updated_receiver = self.node_mlp(torch.cat([receiver_features, agg], dim=-1))

        # FIX: Explicit residual connection (GraphCast-style)
        # The MLP's internal residual is disabled because in_dim(node+edge) != out_dim(node).
        # Without this, 8 processor rounds create severe vanishing gradients.
        # For bipartite graphs (encoder/decoder), skip only when dims match.
        if not self.bipartite and receiver_features.shape[-1] == updated_receiver.shape[-1]:
            updated_receiver = updated_receiver + receiver_features

        return updated_receiver, updated_edges


# =============================================================================
# ENCODER / PROCESSOR / DECODER
# =============================================================================

class Grid2MeshEncoder(nn.Module):
    def __init__(self, grid_input_dim, mesh_pos_dim, latent_dim, edge_dim,
                 hidden_dim, time_dim, cond_dim):
        super().__init__()
        self.grid_encoder = MLP(grid_input_dim, hidden_dim, latent_dim, num_layers=2, residual=False)
        self.mesh_encoder = MLP(mesh_pos_dim, hidden_dim, latent_dim, num_layers=2, residual=False)
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.time_mlp = nn.Sequential(
            ContinuousTimeEmbedding(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        self.interaction = InteractionNetwork(
            node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim,
            bipartite=True, sender_dim=latent_dim,
        )

    def forward(self, grid_features, mesh_pos_features, g2m_edge_index, g2m_edge_attr, t, cond):
        B = grid_features.shape[0]
        grid_h = self.grid_encoder(grid_features)
        mesh_h = self.mesh_encoder(mesh_pos_features).unsqueeze(0).expand(B, -1, -1)
        edge_h = self.edge_encoder(g2m_edge_attr)

        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        scale, shift = (t_emb + c_emb).chunk(2, dim=-1)
        grid_h = grid_h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        mesh_out = []
        for b in range(B):
            m_h, _ = self.interaction(grid_h[b], mesh_h[b], g2m_edge_index, edge_h)
            mesh_out.append(m_h)
        return torch.stack(mesh_out, dim=0)


class MeshProcessor(nn.Module):
    def __init__(self, latent_dim, edge_dim, hidden_dim, num_rounds=8):
        super().__init__()
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.rounds = nn.ModuleList([
            InteractionNetwork(node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim)
            for _ in range(num_rounds)
        ])

    def forward(self, mesh_features, mesh_edge_index, mesh_edge_attr):
        B = mesh_features.shape[0]
        edge_h = self.edge_encoder(mesh_edge_attr)
        all_h = []
        for b in range(B):
            h, e = mesh_features[b], edge_h.clone()
            for rnd in self.rounds:
                e_in = e
                h, e = rnd(h, h, mesh_edge_index, e)
                # Edge residual (GraphCast-style): edges also accumulate updates
                e = e + e_in
            all_h.append(h)
        return torch.stack(all_h, dim=0)


class Mesh2GridDecoder(nn.Module):
    def __init__(self, latent_dim, edge_dim, hidden_dim, output_dim):
        super().__init__()
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.grid_proj = MLP(latent_dim, hidden_dim, latent_dim, num_layers=1, residual=True)
        self.interaction = InteractionNetwork(
            node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim,
            bipartite=True, sender_dim=latent_dim,
        )
        self.output_mlp = MLP(latent_dim, hidden_dim, output_dim, num_layers=2, residual=False)

    def forward(self, mesh_features, grid_skip, m2g_edge_index, m2g_edge_attr):
        B = mesh_features.shape[0]
        grid_h = self.grid_proj(grid_skip)
        edge_h = self.edge_encoder(m2g_edge_attr)
        grid_out = []
        for b in range(B):
            g_h, _ = self.interaction(mesh_features[b], grid_h[b], m2g_edge_index, edge_h)
            grid_out.append(g_h)
        return self.output_mlp(torch.stack(grid_out, dim=0))


# =============================================================================
# GLOBAL CONTEXT ENCODER
# =============================================================================

class GlobalContextForMesh(nn.Module):
    def __init__(self, in_channels, encoder_dim, output_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(time_dim, encoder_dim * 4), nn.SiLU())
        self.blocks = nn.Sequential(
            self._block(in_channels, encoder_dim),
            self._block(encoder_dim, encoder_dim * 2),
            self._block(encoder_dim * 2, encoder_dim * 4),
        )
        self.proj = nn.Conv2d(encoder_dim * 4, output_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, output_dim)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c), nn.SiLU(),
        )

    def forward(self, global_fields, time_emb, mesh_lat, mesh_lon):
        B = global_fields.shape[0]
        x = self.blocks(global_fields)
        x = x + self.time_proj(time_emb)[:, :, None, None]
        x = self.norm(self.proj(x))

        # ERA5 global grid: lat 90N->90S (top->bottom), lon 0E->359E (left->right)
        # grid_sample maps [-1,+1] to [first,last] pixel
        # Latitude: 90N -> y=-1, 90S -> y=+1
        norm_lat = -(mesh_lat / 90.0)
        # Longitude: convert [-180,180] to [0,360] then to [-1,+1]
        mesh_lon_360 = (mesh_lon + 360.0) % 360.0
        norm_lon = (mesh_lon_360 / 180.0) - 1.0

        coords = torch.stack([
            torch.from_numpy(norm_lon).float().to(x.device),
            torch.from_numpy(norm_lat).float().to(x.device),
        ], dim=-1).unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)

        return F.grid_sample(x, coords, mode='bilinear',
                             padding_mode='border', align_corners=True
                             ).squeeze(-1).permute(0, 2, 1)


# =============================================================================
# FULL MODEL
# =============================================================================

class MeshFlowNet(nn.Module):
    """
    GraphCast-style mesh GNN backbone.

    deterministic=False (GenCast/CFM):
      Input:  [x_flow, x_t, x_tm1, x_tm2, spatial_c]
      Output: velocity field v_t
      t:      flow time in [0, 1]

    deterministic=True (GraphCast):
      Input:  [x_t, x_tm1, x_tm2, spatial_c]
      Output: residual y - x_t
      t:      fixed at 0.5 internally (acts as learned bias)
    """
    def __init__(
        self,
        img_channels=1,
        spatial_cond_channels=15,
        condition_dim=5,
        latent_dim=256,
        hidden_dim=512,
        num_processor_rounds=8,
        mesh=None,
        image_size=(621, 1405),
        num_global_channels=5,
        global_encoder_dim=64,
        deterministic=False,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.image_size = image_size
        self.use_global = num_global_channels > 0
        self.deterministic = deterministic

        # Probabilistic: [x_flow(1) + conditions(15)] = 16
        # Deterministic:  [conditions(15)] = 15  (no flow channel)
        if deterministic:
            grid_input_dim = spatial_cond_channels
        else:
            grid_input_dim = img_channels + spatial_cond_channels

        time_dim = latent_dim * 2

        self.time_mlp = nn.Sequential(
            ContinuousTimeEmbedding(latent_dim),
            nn.Linear(latent_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.use_global:
            self.global_encoder = GlobalContextForMesh(
                in_channels=num_global_channels, encoder_dim=global_encoder_dim,
                output_dim=latent_dim, time_dim=time_dim,
            )

        self.encoder = Grid2MeshEncoder(
            grid_input_dim=grid_input_dim, mesh_pos_dim=9,
            latent_dim=latent_dim, edge_dim=3,
            hidden_dim=hidden_dim, time_dim=time_dim, cond_dim=condition_dim,
        )
        self.skip_proj = MLP(grid_input_dim, hidden_dim, latent_dim,
                             num_layers=2, residual=False)
        self.processor = MeshProcessor(latent_dim=latent_dim, edge_dim=4,
                                       hidden_dim=hidden_dim, num_rounds=num_processor_rounds)
        self.decoder = Mesh2GridDecoder(latent_dim=latent_dim, edge_dim=3,
                                        hidden_dim=hidden_dim, output_dim=img_channels)
        self._mesh = mesh

    def set_mesh(self, mesh):
        self._mesh = mesh

    def forward(self, x, t, cond, global_fields=None):
        """
        Same signature for both modes.
        In deterministic mode, t is overridden to 0.5 internally.
        """
        mesh = self._mesh
        assert mesh is not None, "Call set_mesh() before forward()"

        B = x.shape[0]
        H, W = self.image_size
        H_pad, W_pad = x.shape[2], x.shape[3]

        # Deterministic: t becomes a constant
        if self.deterministic:
            t = torch.full((B,), 0.5, device=x.device, dtype=x.dtype)

        self.downsample_factor = 4

        x_crop = x[:, :, :H, :W]
        # Downsample before mesh encoding
        x_down = F.avg_pool2d(x_crop, kernel_size=self.downsample_factor)
        
        # FIX 1: Define H_down, W_down
        H_down, W_down = x_down.shape[2], x_down.shape[3]
        
        grid_flat = grid_to_flat(x_down, mesh.grid_node_indices, (H_down, W_down))

        # ============ DIAGNOSTIC: forward pass data flow ============
        if not hasattr(self, '_fwd_counter'):
            self._fwd_counter = 0
        self._fwd_counter += 1
        _do_diag = (self._fwd_counter % 500 == 1)
        if _do_diag:
            with torch.no_grad():
                print(f"\n  [DIAG forward] step={self._fwd_counter}")
                print(f"    x_down        shape={tuple(x_down.shape)}  "
                      f"mean={x_down.mean().item():.4f}  std={x_down.std().item():.4f}")
                print(f"    grid_flat     shape={tuple(grid_flat.shape)}  "
                      f"mean={grid_flat.mean().item():.4f}  std={grid_flat.std().item():.4f}")
                # Per-channel stats for grid_flat to check if all channels carry signal
                for ch in range(min(grid_flat.shape[-1], 16)):
                    ch_data = grid_flat[:, :, ch]
                    print(f"      ch{ch:02d}  mean={ch_data.mean().item():.4f}  "
                          f"std={ch_data.std().item():.4f}  "
                          f"range=[{ch_data.min().item():.4f}, {ch_data.max().item():.4f}]")
        # ============ END DIAGNOSTIC ============

        t_emb = self.time_mlp(t)
        
        if self.use_global and global_fields is not None:
            global_ctx = self.global_encoder(global_fields, t_emb, mesh.mesh_lat, mesh.mesh_lon)

        grid_skip = self.skip_proj(grid_flat)

        mesh_h = self.encoder(
            grid_flat, mesh.mesh_node_features_t,
            mesh.g2m_edge_index_t, mesh.g2m_edge_attr_t, t, cond,
        )

        # ============ DIAGNOSTIC: mesh processing ============
        if _do_diag:
            with torch.no_grad():
                print(f"    mesh_h (post-encoder)  mean={mesh_h.mean().item():.4f}  "
                      f"std={mesh_h.std().item():.4f}")
                if self.use_global and global_fields is not None:
                    print(f"    global_ctx             mean={global_ctx.mean().item():.4f}  "
                          f"std={global_ctx.std().item():.4f}  "
                          f"ratio_to_mesh={global_ctx.std().item()/(mesh_h.std().item()+1e-8):.4f}")
        # ============ END DIAGNOSTIC ============

        if self.use_global and global_fields is not None:
            mesh_h = mesh_h + global_ctx

        mesh_h_pre = mesh_h
        mesh_h = self.processor(mesh_h, mesh.mesh_edge_index_t, mesh.mesh_edge_attr_t)

        # ============ DIAGNOSTIC: processor effect ============
        if _do_diag:
            with torch.no_grad():
                delta = (mesh_h - mesh_h_pre).abs().mean().item()
                print(f"    mesh_h (post-processor) mean={mesh_h.mean().item():.4f}  "
                      f"std={mesh_h.std().item():.4f}")
                print(f"    processor delta (L1)   {delta:.6f}  "
                      f"relative={delta/(mesh_h_pre.abs().mean().item()+1e-8):.4f}")
        # ============ END DIAGNOSTIC ============

        grid_out = self.decoder(mesh_h, grid_skip,
                                mesh.m2g_edge_index_t, mesh.m2g_edge_attr_t)

        # FIX 2: Reconstruct at downsampled resolution, then interpolate up
        out_down = flat_to_grid(grid_out, mesh.grid_node_indices, (H_down, W_down), fill_value=0.0)
        out = F.interpolate(out_down, size=(H, W), mode='bilinear', align_corners=False)

        # ============ DIAGNOSTIC: output stats ============
        if _do_diag:
            with torch.no_grad():
                print(f"    grid_out      shape={tuple(grid_out.shape)}  "
                      f"mean={grid_out.mean().item():.4f}  std={grid_out.std().item():.4f}")
                print(f"    grid_skip     mean={grid_skip.mean().item():.4f}  "
                      f"std={grid_skip.std().item():.4f}")
                print(f"    out (final)   mean={out.mean().item():.4f}  "
                      f"std={out.std().item():.4f}  "
                      f"range=[{out.min().item():.4f}, {out.max().item():.4f}]")
        # ============ END DIAGNOSTIC ============

        if H_pad > H or W_pad > W:
            out = F.pad(out, (0, W_pad - W, 0, H_pad - H), value=0.0)
        return out

    def _sample_global_to_grid(self, global_fields, t_emb, grid_indices, image_size):
        H, W = image_size
        lat_1d = np.linspace(25.0, 50.0, H)
        lon_1d = np.linspace(-130.0, -60.0, W)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        return self.global_encoder(global_fields, t_emb,
                                   lat_grid.ravel()[grid_indices],
                                   lon_grid.ravel()[grid_indices])


def count_parameters(model):
    total = 0
    components = {}
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        components[component] = components.get(component := name.split('.')[0], 0) + n
    print(f"\nParameter breakdown ({total/1e6:.1f}M total):")
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        print(f"  {comp:25s} {count/1e6:8.1f}M ({100*count/total:5.1f}%)")
    return total
