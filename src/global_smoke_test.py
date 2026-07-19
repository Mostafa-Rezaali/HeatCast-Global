"""Single-process CPU smoke test for the true Phase A global grid shape."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset

from data_pipeline.build_cache import CACHE_CHANNELS
from data_pipeline.regrid import grid_for_resolution
from global_dataset import assemble_global_tensors, identity_preprocessor
from icosahedral_mesh import IcosahedralMesh
from mesh_backbone import MeshFlowNet
from mode_dispatch import deterministic_loss


def _synthetic_assembled_sample(height: int, width: int, seed: int):
    rng = np.random.default_rng(int(seed))
    context = rng.normal(0.0, 0.25, size=(3, height, width, len(CACHE_CHANNELS))).astype(np.float32)
    channel_index = {name: index for index, name in enumerate(CACHE_CHANNELS)}
    context[..., channel_index["sst_valid"]] = 1.0
    context[..., channel_index["land_mask"]] = 1.0
    context[..., channel_index["orography"]] = np.linspace(
        0.0, 3000.0, height, dtype=np.float32
    )[None, :, None]
    target = rng.normal(0.0, 0.5, size=(14, height, width)).astype(np.float32)
    grid = grid_for_resolution("1.5deg")
    return assemble_global_tensors(
        context,
        target,
        (20000501, 20000430, 20000429),
        tuple(20000516 + value for value in range(14)),
        grid.lat,
        grid.lon,
        np.zeros(8, dtype=np.float32),
        identity_preprocessor((height, width)),
    )


def _export_dry_run(path: Path, grid, mean, sigma, probability) -> None:
    """Write and reopen the global week3/week4/W34 smoke product."""
    windows = {
        "week3": slice(0, 7),
        "week4": slice(7, 14),
        "w34": slice(0, 14),
    }
    with NetCDFDataset(path, "w") as dataset:
        dataset.createDimension("lat", grid.shape[0])
        dataset.createDimension("lon", grid.shape[1])
        dataset.createVariable("lat", "f4", ("lat",))[:] = grid.lat
        dataset.createVariable("lon", "f4", ("lon",))[:] = grid.lon
        for label, window in windows.items():
            dataset.createVariable(f"{label}_mean", "f4", ("lat", "lon"), zlib=True)[:] = np.mean(mean[window], axis=0)
            dataset.createVariable(f"{label}_sigma", "f4", ("lat", "lon"), zlib=True)[:] = (
                np.sqrt(np.sum(np.square(sigma[window]), axis=0)) / (window.stop - window.start)
            )
            dataset.createVariable(f"{label}_exceedance_probability", "f4", ("lat", "lon"), zlib=True)[:] = np.mean(
                probability[window], axis=0
            )
    with NetCDFDataset(path) as dataset:
        if dataset.variables["w34_mean"].shape != grid.shape:
            raise RuntimeError("Smoke export shape verification failed.")


def run_global_smoke_test(seed: int = 42) -> dict:
    """Run two train steps, validation, sampling, and export on ``121 x 240``."""
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, previous_threads))
    try:
        device = torch.device("cpu")
        grid = grid_for_resolution("1.5deg")
        mesh = IcosahedralMesh(
            refinement_level=0,
            lat_range=(-90.0, 90.0),
            lon_range=(0.0, 360.0),
            grid_lat=grid.lat,
            grid_lon=grid.lon,
            land_mask=None,
            k_grid2mesh=1,
            k_mesh2grid=1,
            global_domain=True,
        ).to_torch(device)
        model = MeshFlowNet(
            img_channels=2,
            spatial_cond_channels=26,
            condition_dim=8,
            latent_dim=8,
            hidden_dim=16,
            num_processor_rounds=1,
            mesh=mesh,
            image_size=grid.shape,
            num_global_channels=0,
            deterministic=True,
            dropout=0.0,
            predict_persistence_residual=False,
            multi_lead_tube=True,
            prediction_leads=tuple(range(15, 29)),
            tube_temporal_heads=1,
            tube_decode_chunk_size=2,
            tube_loss_weights=(0.80, 0.0, 0.20),
            distributional_head=True,
            sigma_floor=0.1,
            gradient_checkpointing=False,
        ).to(device)
        model.crps_loss = True
        model.mse_anchor_weight = 0.0
        model.exceedance_bce_weight = 0.0
        model.exceedance_count_weight = 0.0
        model.exceedance_pos_weight = 1.0
        model.exceedance_focal_gamma = 0.0
        sample = _synthetic_assembled_sample(*grid.shape, seed)
        batched = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
            if torch.is_tensor(value)
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_losses = []
        model.train()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = deterministic_loss(
                model,
                batched["target"], batched["x_t"], batched["x_tm1"], batched["x_tm2"],
                batched["spatial_c"], batched["vector"], batched["global_fields"], batched["mask"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Smoke training produced non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            validation_loss, components = deterministic_loss(
                model,
                batched["target"], batched["x_t"], batched["x_tm1"], batched["x_tm2"],
                batched["spatial_c"], batched["vector"], batched["global_fields"], batched["mask"],
            )
            mean = components["pred"]
            sigma = components["sigma"]
            sampled_member = mean + sigma * torch.randn_like(mean)
            probability = 0.5 * (1.0 + torch.erf(mean / (sigma * math.sqrt(2.0))))
        if mean.shape != (1, 14, 121, 240) or sigma.shape != mean.shape:
            raise RuntimeError(f"Smoke distributional shape mismatch: mean={mean.shape}, sigma={sigma.shape}.")
        if sampled_member.shape != mean.shape or not torch.isfinite(sampled_member).all():
            raise RuntimeError("Smoke distributional sampling failed.")
        with tempfile.TemporaryDirectory(prefix="heatcast_global_smoke_") as directory:
            export_path = Path(directory) / "smoke_global.nc"
            _export_dry_run(
                export_path,
                grid,
                mean[0].cpu().numpy(),
                sigma[0].cpu().numpy(),
                probability[0].cpu().numpy(),
            )
        result = {
            "status": "pass",
            "grid_shape": list(grid.shape),
            "prediction_shape": list(mean.shape),
            "train_steps": 2,
            "train_losses": train_losses,
            "validation_loss": float(validation_loss),
            "sigma_min": float(sigma.min()),
            "sample_finite": True,
            "export_dry_run": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        torch.set_num_threads(previous_threads)
