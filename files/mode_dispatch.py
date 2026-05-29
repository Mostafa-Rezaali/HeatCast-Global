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

    # Inference:
    pred = generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                           global_fields, device, h, w, pad_h, pad_w, mask,
                           deterministic=Config.DETERMINISTIC,
                           n_steps=Config.CFM_SAMPLING_STEPS)
================================================================================
"""

import torch
import numpy as np

OCEAN_FILL = 0


# =============================================================================
# UNIFIED DISPATCH
# =============================================================================

def compute_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                 global_fields, mask, deterministic=False):
    """
    Dispatch to the correct loss function.

    Args: same as cfm_loss / deterministic_loss
    Returns: (total_loss, components_dict)
    """
    if deterministic:
        return deterministic_loss(model, y, x_t, x_tm1, x_tm2,
                                  spatial_c, vec_c, global_fields, mask)
    else:
        return cfm_loss(model, fm, y, x_t, x_tm1, x_tm2,
                        spatial_c, vec_c, global_fields, mask)


def generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                    global_fields, device, h, w, pad_h, pad_w, mask,
                    deterministic=False, n_steps=50):
    """
    Dispatch to the correct sampling function.

    Returns: (h, w) numpy array of the predicted field (normalized z-score).
    """
    if deterministic:
        return generate_deterministic_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, pad_h, pad_w, mask)
    else:
        return generate_cfm_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, pad_h, pad_w, mask,
            n_steps=n_steps)


# =============================================================================
# DETERMINISTIC MODE (GraphCast)
# =============================================================================

def deterministic_loss(model, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                       global_fields, mask):
    """
    Direct regression loss for deterministic (GraphCast) mode.

    The model predicts the residual: pred = y - x_t
    Loss = masked MSE on the residual.

    Input to model: [x_t, x_tm1, x_tm2, spatial_c]  (no flow state)
    """
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    # Target: residual from current state to future state
    residual_target = y - x_t  # (B, 1, H, W)

    # Model input: conditions only, no flow state
    x_input = torch.cat([x_t, x_tm1, x_tm2, spatial_c], dim=1)

    # Dummy time (model ignores it in deterministic mode, sets t=0.5 internally)
    dummy_t = torch.full((y.shape[0],), 0.5, device=device)

    residual_pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)

    # Masked MSE
    loss_per_pixel = (residual_pred - residual_target) ** 2 * mask
    loss_per_sample = loss_per_pixel.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    total_loss = loss_per_sample.mean()

    # Reconstruction metrics (the actual prediction y_hat = x_t + residual_pred)
    y_hat = x_t + residual_pred
    recon_mse = ((y_hat - y) ** 2 * mask).sum(dim=(1, 2, 3))
    recon_mse = (recon_mse / (mask.sum(dim=(1, 2, 3)) + 1e-8)).mean()

    return total_loss, {
        "det_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
    }


@torch.inference_mode()
def generate_deterministic_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                                  global_fields, device, h, w, pad_h, pad_w, mask):
    """
    Single forward pass prediction for deterministic (GraphCast) mode.

    Model predicts residual, add it to x_t to get the forecast.
    No ODE integration, no stochastic sampling. One forward pass.
    """
    model.eval()

    x_input = torch.cat([x_t, x_tm1, x_tm2, spatial_c], dim=1)
    dummy_t = torch.full((1,), 0.5, device=device)

    residual = model(x_input, dummy_t, vec_c, global_fields=global_fields)

    # Forecast = current state + predicted residual
    y_hat = x_t + residual
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
    """
    Conditional Flow Matching loss for probabilistic (GenCast) mode.

    The model predicts velocity v_t along the interpolation path from x_t to y.
    """
    batch_size = y.shape[0]
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    times = sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0)
    times = times.clamp(1e-5, 1.0 - 1e-5)

    # Interpolate along flow path
    x_t_flow = fm.sample_xt(x_0=x_t, x_1=y, t=times)
    x_t_flow = x_t_flow * mask + OCEAN_FILL * (1 - mask)

    v_target = fm.velocity_target(x_0=x_t, x_1=y)

    # Model input: [flow_state, x_t, x_tm1, x_tm2, spatial_c]
    x_input = torch.cat([x_t_flow, x_t, x_tm1, x_tm2, spatial_c], dim=1)
    v_pred = model(x_input, times, vec_c, global_fields=global_fields)

    loss_per_pixel = (v_pred - v_target) ** 2 * mask
    loss_per_sample = loss_per_pixel.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    total_loss = loss_per_sample.mean()

    # ============ DIAGNOSTIC: velocity field analysis ============
    if not hasattr(cfm_loss, '_diag_counter'):
        cfm_loss._diag_counter = 0
    cfm_loss._diag_counter += 1
    if cfm_loss._diag_counter % 500 == 1:
        with torch.no_grad():
            land = mask > 0.5
            vp = v_pred[land]
            vt = v_target[land]
            zero_mse = (vt ** 2).mean().item()
            print(f"\n  [DIAG cfm_loss] step={cfm_loss._diag_counter}")
            print(f"    v_target  mean={vt.mean().item():.4f}  std={vt.std().item():.4f}  "
                  f"absmax={vt.abs().max().item():.4f}")
            print(f"    v_pred    mean={vp.mean().item():.4f}  std={vp.std().item():.4f}  "
                  f"absmax={vp.abs().max().item():.4f}")
            print(f"    loss={total_loss.item():.4f}  zero_baseline={zero_mse:.4f}  "
                  f"ratio={total_loss.item()/max(zero_mse,1e-8):.4f}")
            print(f"    times  min={times.min().item():.4f}  max={times.max().item():.4f}  "
                  f"mean={times.mean().item():.4f}")
            # Check if v_pred correlates with v_target at all
            if vp.numel() > 100:
                corr = torch.corrcoef(torch.stack([vp[:10000], vt[:10000]]))[0, 1].item()
                print(f"    v_pred vs v_target correlation (sample): {corr:.4f}")
    # ============ END DIAGNOSTIC ============

    # Reconstruction diagnostic
    t_view = times.view(-1, 1, 1, 1)
    y_recon = x_t_flow + (1 - t_view) * v_pred
    recon_mse = ((y_recon - y) ** 2 * mask).sum(dim=(1, 2, 3))
    recon_mse = (recon_mse / (mask.sum(dim=(1, 2, 3)) + 1e-8)).mean()

    return total_loss, {
        "cfm_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
    }


@torch.inference_mode()
def generate_cfm_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                        global_fields, device, h, w, pad_h, pad_w, mask,
                        n_steps=50):
    """
    ODE integration (Heun's method) for probabilistic (GenCast/CFM) mode.

    Integrates the learned velocity field from t=0 to t=1.
    Starting point is x_t (today's weather), endpoint is the forecast.
    """
    model.eval()

    z = x_t.clone()
    z = z * mask + OCEAN_FILL * (1 - mask)

    dt = 1.0 / n_steps
    VAL_MIN, VAL_MAX = -4.0, 4.0

    for i in range(n_steps):
        t_i = torch.tensor([i * dt], device=device)
        t_next = torch.tensor([(i + 1) * dt], device=device).clamp(max=1.0)

        # Heun step 1
        x_input = torch.cat([z, x_t, x_tm1, x_tm2, spatial_c], dim=1)
        v1 = model(x_input, t_i.expand(1), vec_c, global_fields=global_fields)
        v1 = v1 * mask

        z_euler = z + v1 * dt
        z_euler = z_euler.clamp(VAL_MIN, VAL_MAX)
        z_euler = z_euler * mask + OCEAN_FILL * (1 - mask)

        # Heun step 2
        x_input2 = torch.cat([z_euler, x_t, x_tm1, x_tm2, spatial_c], dim=1)
        v2 = model(x_input2, t_next.expand(1), vec_c, global_fields=global_fields)
        v2 = v2 * mask

        # Heun update
        z = z + (v1 + v2) * 0.5 * dt
        z = z.clamp(VAL_MIN, VAL_MAX)
        z = z * mask + OCEAN_FILL * (1 - mask)

    return z[0, 0, :h, :w].cpu().numpy()
