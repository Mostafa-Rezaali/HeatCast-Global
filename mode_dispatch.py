"""
================================================================================
Mode Dispatch: Deterministic (GraphCast) vs Probabilistic (GenCast/CFM)
================================================================================

Import the appropriate loss and sampling functions based on Config.DETERMINISTIC.

Usage in training script:
    from mode_dispatch import compute_loss, generate_sample

    # Training:
    loss, components = compute_loss(model, fm, y, x_t, x_tm1, x_tm2,
                                     spatial_c, vec_c, global_fields, mask,
                                     deterministic=Config.DETERMINISTIC)
    # components['pred'] contains the prediction tensor (deterministic mode)

    # Inference (single step):
    pred = generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                           global_fields, device, h, w, mask,
                           deterministic=Config.DETERMINISTIC,
                           n_steps=Config.CFM_SAMPLING_STEPS)
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


def _predicts_persistence_residual(model):
    return bool(getattr(_raw_model(model), "predict_persistence_residual", False))


def _multi_lead_tube(model):
    return bool(getattr(_raw_model(model), "multi_lead_tube", False))


def _tube_center_index(model):
    raw = _raw_model(model)
    leads = tuple(int(x) for x in getattr(raw, "prediction_leads", (15,)))
    center = int(getattr(raw, "center_lead", 15 if 15 in leads else leads[len(leads) // 2]))
    return leads.index(center)


def _tube_loss_weights(model):
    raw = _raw_model(model)
    return tuple(float(x) for x in getattr(raw, "tube_loss_weights", (0.80, 0.10, 0.10)))


def _ensure_standard_input(model):
    mode = _input_mode(model)
    if mode != "standard":
        raise ValueError(
            f"Unsupported input_mode={mode!r}. JEPA inputs are disabled; use standard MeshFlowNet inputs."
        )


def _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c):
    _ensure_standard_input(model)
    return torch.cat([x_t, x_tm1, x_tm2, spatial_c], dim=1)


def _cfm_input(model, flow_state, x_t, x_tm1, x_tm2, spatial_c):
    _ensure_standard_input(model)
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
    Deterministic direct prediction loss.
    Returns prediction in components['pred'] for downstream use (e.g. extreme loss).
    """
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((y.shape[0],), 0.5, device=device)

    raw_pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)
    pred = x_t + raw_pred if _predicts_persistence_residual(model) else raw_pred

    if _multi_lead_tube(model):
        if y.ndim != 4:
            raise RuntimeError(f"Tube target must have shape (B,L,H,W), got {tuple(y.shape)}")
        mask_expanded = mask.expand_as(pred)
        valid = mask_expanded > 0.5
        if valid.any():
            daily_mse = ((pred - y).square() * mask_expanded).sum() / mask_expanded.sum().clamp_min(1.0)
            center_idx = _tube_center_index(model)
            center_pred = pred[:, center_idx:center_idx + 1]
            center_truth = y[:, center_idx:center_idx + 1]
            center_mask = mask.expand_as(center_pred)
            center_mse = ((center_pred - center_truth).square() * center_mask).sum() / center_mask.sum().clamp_min(1.0)
            weekly_pred = pred.mean(dim=1, keepdim=True)
            weekly_truth = y.mean(dim=1, keepdim=True)
            weekly_mask = mask.expand_as(weekly_pred)
            weekly_mse = ((weekly_pred - weekly_truth).square() * weekly_mask).sum() / weekly_mask.sum().clamp_min(1.0)
        else:
            daily_mse = pred.sum() * 0.0
            center_mse = daily_mse
            weekly_mse = daily_mse
        w_daily, w_center, w_weekly = _tube_loss_weights(model)
        total_loss = w_daily * daily_mse + w_center * center_mse + w_weekly * weekly_mse
        zero = torch.tensor(0.0, device=device)
        return total_loss, {
            "det_loss": total_loss.detach(),
            "recon_mse": daily_mse.detach(),
            "tube_daily_mse": daily_mse.detach(),
            "tube_center_mse": center_mse.detach(),
            "tube_weekly_mse": weekly_mse.detach(),
            "residual_abs": raw_pred.detach().abs().mean(),
            "loss_t<0.33": zero,
            "loss_0.33<t<0.67": zero,
            "loss_t>0.67": zero,
            "pred": pred,
        }

    mask_expanded = mask.expand_as(pred)
    valid = mask_expanded > 0.5
    if valid.any():
        total_loss = F.huber_loss(pred[valid], y[valid], delta=2.0)
        recon_mse = F.mse_loss(pred[valid], y[valid])
    else:
        total_loss = pred.sum() * 0.0
        recon_mse = total_loss.detach()

    zero = torch.tensor(0.0, device=device)
    return total_loss, {
        "det_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
        "residual_abs": raw_pred.detach().abs().mean(),
        "loss_t<0.33": zero,
        "loss_0.33<t<0.67": zero,
        "loss_t>0.67": zero,
        "pred": pred,  # prediction tensor for extreme loss
    }


@torch.inference_mode()
def generate_deterministic_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                                  global_fields, device, h, w, mask):
    model.eval()
    mask = mask.to(device=device, dtype=x_t.dtype)
    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((1,), 0.5, device=device)

    raw_hat = model(x_input, dummy_t, vec_c, global_fields=global_fields)
    y_hat = x_t + raw_hat if _predicts_persistence_residual(model) else raw_hat
    y_hat = y_hat * mask + OCEAN_FILL * (1 - mask)
    y_hat = y_hat.clamp(-4.0, 4.0)

    if _multi_lead_tube(model):
        center_idx = _tube_center_index(model)
        return y_hat[0, center_idx, :h, :w].cpu().numpy()

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
        "pred": None,  # CFM doesn't have a single prediction during training
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
