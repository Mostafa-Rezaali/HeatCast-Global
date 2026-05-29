"""
================================================================================
Mode Dispatch: Deterministic (GraphCast) vs Probabilistic (GenCast/CFM)
================================================================================

Import the appropriate loss and sampling functions based on Config.DETERMINISTIC.

Usage in training script:
    from mode_dispatch import compute_loss, generate_sample, generate_autoregressive_rollout

    # Training:
    loss, components = compute_loss(model, fm, y, x_t, x_tm1, x_tm2,
                                     spatial_c, vec_c, global_fields, mask,
                                     deterministic=Config.DETERMINISTIC)

    # Inference (single step):
    pred = generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                           global_fields, device, h, w, mask,
                           deterministic=Config.DETERMINISTIC,
                           n_steps=Config.CFM_SAMPLING_STEPS)

    # Inference (autoregressive rollout):
    pred = generate_autoregressive_rollout(
        model, dataset, t_start, device, h, w, mask,
        deterministic=Config.DETERMINISTIC,
        n_steps=Config.CFM_SAMPLING_STEPS,
        rollout_steps=Config.ROLLOUT_STEPS,
        return_all_steps=False)
================================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np

OCEAN_FILL = 0


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _input_mode(model):
    return getattr(_raw_model(model), "input_mode", "standard")


def _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c):
    if _input_mode(model) == "jepa_spatial_global":
        return spatial_c
    return torch.cat([x_t, x_tm1, x_tm2, spatial_c], dim=1)


def _cfm_input(model, flow_state, x_t, x_tm1, x_tm2, spatial_c):
    if _input_mode(model) == "jepa_spatial_global":
        return torch.cat([flow_state, spatial_c], dim=1)
    return torch.cat([flow_state, x_t, x_tm1, x_tm2, spatial_c], dim=1)


# =============================================================================
# UNIFIED DISPATCH
# =============================================================================

def compute_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                 global_fields, mask, deterministic=False):
    if deterministic:
        return deterministic_loss(model, y, x_t, x_tm1, x_tm2,
                                  spatial_c, vec_c, global_fields, mask)
    else:
        return cfm_loss(model, fm, y, x_t, x_tm1, x_tm2,
                        spatial_c, vec_c, global_fields, mask)


def generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                    global_fields, device, h, w, mask,
                    deterministic=False, n_steps=50):
    """
    Single-step generation. Returns (h, w) numpy array.
    """
    if deterministic:
        return generate_deterministic_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask)
    else:
        return generate_cfm_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask,
            n_steps=n_steps)


# =============================================================================
# DETERMINISTIC MODE (GraphCast)
# =============================================================================

def deterministic_loss(model, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                       global_fields, mask):
    """
    Deterministic direct anomaly training loss.
    Standard input: [x_t(1), x_tm1(1), x_tm2(1), spatial_c(18)] = 21 channels.
    JEPA input mode: spatial_c is the exported Met-JEPA map tensor, e.g. 32 channels.
    Target: anomaly at t + lead_time, normalized by local daily climatology std.
    """
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((y.shape[0],), 0.5, device=device)

    anom_pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)

    mask_expanded = mask.expand_as(anom_pred)
    valid = mask_expanded > 0.5
    if valid.any():
        total_loss = F.huber_loss(anom_pred[valid], y[valid], delta=2.0)
        recon_mse = F.mse_loss(anom_pred[valid], y[valid])
    else:
        total_loss = anom_pred.sum() * 0.0
        recon_mse = total_loss.detach()

    zero = torch.tensor(0.0, device=device)
    return total_loss, {
        "det_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
        "loss_t<0.33": zero,
        "loss_0.33<t<0.67": zero,
        "loss_t>0.67": zero,
    }


@torch.inference_mode()
def generate_deterministic_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                                  global_fields, device, h, w, mask):
    model.eval()
    mask = mask.to(device=device, dtype=x_t.dtype)
    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((1,), 0.5, device=device)

    y_hat = model(x_input, dummy_t, vec_c, global_fields=global_fields)
    y_hat = y_hat * mask + OCEAN_FILL * (1 - mask)
    y_hat = y_hat.clamp(-4.0, 4.0)

    return y_hat[0, 0, :h, :w].cpu().numpy()


# =============================================================================
# PROBABILISTIC MODE (GenCast/CFM)
# =============================================================================

def sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


def cfm_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
             global_fields, mask):
    batch_size = y.shape[0]
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    times = sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0)
    times = times.clamp(1e-5, 1.0 - 1e-5)

    x_t_flow = fm.sample_xt(x_0=x_t, x_1=y, t=times)
    x_t_flow = x_t_flow * mask + OCEAN_FILL * (1 - mask)
    v_target = fm.velocity_target(x_0=x_t, x_1=y)

    x_input = _cfm_input(model, x_t_flow, x_t, x_tm1, x_tm2, spatial_c)
    v_pred = model(x_input, times, vec_c, global_fields=global_fields)

    loss_per_pixel = (v_pred - v_target) ** 2 * mask
    loss_per_sample = loss_per_pixel.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    total_loss = loss_per_sample.mean()

    with torch.no_grad():
        low_mask = times < 0.33
        mid_mask = (times >= 0.33) & (times < 0.67)
        high_mask = times >= 0.67
        loss_low = loss_per_sample[low_mask].mean() if low_mask.any() else torch.tensor(0.0, device=device)
        loss_mid = loss_per_sample[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=device)
        loss_high = loss_per_sample[high_mask].mean() if high_mask.any() else torch.tensor(0.0, device=device)

    t_view = times.view(-1, 1, 1, 1)
    y_recon = x_t_flow + (1 - t_view) * v_pred
    recon_mse = ((y_recon - y) ** 2 * mask).sum(dim=(1, 2, 3))
    recon_mse = (recon_mse / (mask.sum(dim=(1, 2, 3)) + 1e-8)).mean()

    return total_loss, {
        "cfm_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
        "loss_t<0.33": loss_low.detach(),
        "loss_0.33<t<0.67": loss_mid.detach(),
        "loss_t>0.67": loss_high.detach(),
    }


@torch.inference_mode()
def generate_cfm_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                        global_fields, device, h, w, mask,
                        n_steps=50):
    """
    ODE integration (Heun's method) for one step.
    """
    model.eval()
    mask = mask.to(device=device, dtype=x_t.dtype)

    z = x_t.clone()
    z = z * mask + OCEAN_FILL * (1 - mask)

    dt = 1.0 / n_steps
    VAL_MIN, VAL_MAX = -4.0, 4.0

    for i in range(n_steps):
        t_i = torch.tensor([i * dt], device=device)
        t_next = torch.tensor([(i + 1) * dt], device=device).clamp(max=1.0)

        x_input = _cfm_input(model, z, x_t, x_tm1, x_tm2, spatial_c)
        v1 = model(x_input, t_i.expand(1), vec_c, global_fields=global_fields)
        v1 = v1 * mask

        z_euler = z + v1 * dt
        z_euler = z_euler.clamp(VAL_MIN, VAL_MAX)
        z_euler = z_euler * mask + OCEAN_FILL * (1 - mask)

        x_input2 = _cfm_input(model, z_euler, x_t, x_tm1, x_tm2, spatial_c)
        v2 = model(x_input2, t_next.expand(1), vec_c, global_fields=global_fields)
        v2 = v2 * mask

        z = z + (v1 + v2) * 0.5 * dt
        z = z.clamp(VAL_MIN, VAL_MAX)
        z = z * mask + OCEAN_FILL * (1 - mask)

    return z[0, 0, :h, :w].cpu().numpy()


# =============================================================================
# AUTOREGRESSIVE ROLLOUT with return_all_steps support
# =============================================================================

@torch.inference_mode()
def generate_autoregressive_rollout(model, dataset, t_start, device, h, w, mask,
                                     deterministic=False, n_steps=50,
                                     rollout_steps=15, return_all_steps=False):
    """
    Autoregressive 1-day rollout for rollout_steps days.

    If return_all_steps is True, returns a list of numpy arrays (one per step).
    Otherwise returns only the final day's prediction.
    """
    model.eval()
    mask_device = mask.to(device=device, dtype=torch.float32)

    def load_hi(t_idx):
        raw = dataset.heat_index[:, :, t_idx]
        tensor = torch.from_numpy(raw.copy()).float()
        land = (torch.isfinite(tensor) & (tensor != 0.0)).float()
        doy_idx = int(dataset.doy_indices[t_idx])
        if not bool(dataset.valid_doys[doy_idx]):
            raise ValueError(f"DOY index {doy_idx} is outside populated MJJAS climatology.")
        clim = torch.from_numpy(dataset.clim_mean[doy_idx].copy()).float()
        cstd = torch.from_numpy(dataset.clim_std[doy_idx].copy()).float()
        normed = torch.zeros_like(tensor)
        valid = (land > 0.5) & torch.isfinite(clim) & torch.isfinite(cstd)
        normed[valid] = (tensor[valid] - clim[valid]) / (cstd[valid] + 1e-6)
        normed = normed.unsqueeze(0).unsqueeze(0)
        land = land.unsqueeze(0).unsqueeze(0)
        return (normed * land + OCEAN_FILL * (1 - land)).to(device)

    x_t   = load_hi(t_start)
    x_tm1 = load_hi(t_start - 1)
    x_tm2 = load_hi(t_start - 2)

    predictions = []

    for k in range(1, rollout_steps + 1):
        t_k = t_start + k - 1

        spatial_extra, vec_c, global_fields, _ = dataset.get_conditions_at(t_k)
        spatial_extra = spatial_extra.unsqueeze(0).to(device)
        vec_c = vec_c.unsqueeze(0).to(device)
        global_fields = global_fields.unsqueeze(0).to(device)

        if deterministic:
            x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_extra)
            dummy_t = torch.full((1,), 0.5, device=device)
            pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)
        else:
            z = x_t.clone()
            z = z * mask_device + OCEAN_FILL * (1 - mask_device)
            dt_val = 1.0 / n_steps

            for i in range(n_steps):
                t_i = torch.tensor([i * dt_val], device=device)
                t_next = torch.tensor([(i + 1) * dt_val], device=device).clamp(max=1.0)

                x_input = _cfm_input(model, z, x_t, x_tm1, x_tm2, spatial_extra)
                v1 = model(x_input, t_i.expand(1), vec_c, global_fields=global_fields)
                v1 = v1 * mask_device

                z_euler = (z + v1 * dt_val).clamp(-4.0, 4.0)
                z_euler = z_euler * mask_device + OCEAN_FILL * (1 - mask_device)

                x_input2 = _cfm_input(model, z_euler, x_t, x_tm1, x_tm2, spatial_extra)
                v2 = model(x_input2, t_next.expand(1), vec_c, global_fields=global_fields)
                v2 = v2 * mask_device

                z = (z + (v1 + v2) * 0.5 * dt_val).clamp(-4.0, 4.0)
                z = z * mask_device + OCEAN_FILL * (1 - mask_device)

            pred = z

        pred = pred * mask_device + OCEAN_FILL * (1 - mask_device)
        pred = pred.clamp(-4.0, 4.0)

        if return_all_steps:
            predictions.append(pred[0, 0, :h, :w].cpu().numpy())

        x_tm2 = x_tm1
        x_tm1 = x_t
        x_t = pred

    if return_all_steps:
        return predictions
    return pred[0, 0, :h, :w].cpu().numpy()
