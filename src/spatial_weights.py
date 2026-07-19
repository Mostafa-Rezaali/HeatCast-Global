"""Latitude-area weights and masked spatial reductions for global metrics.

The public ``area_weights`` helper implements normalized spherical
``cos(latitude)`` weights for both NumPy arrays and PyTorch tensors. All new
global loss and evaluation paths use these helpers instead of unweighted
spatial means.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


def area_weights(lat):
    """Return non-negative ``cos(lat)`` weights normalized to sum to one.

    ``lat`` must be a one-dimensional NumPy array, sequence, or torch tensor in
    degrees. Tensor inputs preserve their device and floating dtype.
    """
    if torch.is_tensor(lat):
        if lat.ndim != 1:
            raise ValueError(f"Latitude must be one-dimensional, got shape {tuple(lat.shape)}.")
        dtype = lat.dtype if lat.dtype.is_floating_point else torch.float32
        values = lat.to(dtype=dtype)
        weights = torch.cos(torch.deg2rad(values)).clamp_min(0.0)
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total.detach().cpu()) <= 0.0:
            raise ValueError("Latitude weights have no positive finite area.")
        return weights / total

    values = np.asarray(lat, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Latitude must be one-dimensional, got shape {values.shape}.")
    weights = np.clip(np.cos(np.deg2rad(values)), 0.0, None)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Latitude weights have no positive finite area.")
    return weights / total


def weighted_spatial_mean(
    values,
    lat,
    mask: Optional[object] = None,
    spatial_dims: Sequence[int] = (-2, -1),
):
    """Compute a finite, area-weighted mean over latitude/longitude axes.

    The latitude and longitude dimensions must be the final two dimensions.
    ``spatial_dims`` is accepted explicitly to make call sites self-documenting
    and currently must equal ``(-2, -1)``.
    """
    if tuple(spatial_dims) != (-2, -1):
        raise ValueError("weighted_spatial_mean currently requires spatial_dims=(-2, -1).")
    if values.ndim < 2:
        raise ValueError("Values must include latitude and longitude dimensions.")

    if torch.is_tensor(values):
        if values.shape[-2] != len(lat):
            raise ValueError("Latitude length does not match the values latitude dimension.")
        weights_1d = area_weights(
            lat if torch.is_tensor(lat) else torch.as_tensor(lat, device=values.device, dtype=values.dtype)
        ).to(device=values.device, dtype=values.dtype)
        weights = weights_1d.reshape((1,) * (values.ndim - 2) + (values.shape[-2], 1))
        valid = torch.isfinite(values)
        if mask is not None:
            valid = valid & torch.as_tensor(mask, device=values.device, dtype=torch.bool)
        effective = torch.where(valid, weights, torch.zeros((), device=values.device, dtype=values.dtype))
        numerator = torch.where(valid, values, torch.zeros_like(values)).mul(effective).sum(dim=(-2, -1))
        denominator = effective.expand_as(values).sum(dim=(-2, -1))
        nan = torch.full_like(numerator, float("nan"))
        return torch.where(denominator > 0, numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny), nan)

    array = np.asarray(values)
    if array.shape[-2] != len(lat):
        raise ValueError("Latitude length does not match the values latitude dimension.")
    weights_1d = np.asarray(area_weights(lat), dtype=np.float64)
    weights = weights_1d.reshape((1,) * (array.ndim - 2) + (array.shape[-2], 1))
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    effective = np.where(valid, weights, 0.0)
    numerator = np.sum(np.where(valid, array, 0.0) * effective, axis=(-2, -1))
    denominator = np.sum(np.broadcast_to(effective, array.shape), axis=(-2, -1))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=np.float64), np.nan),
        where=denominator > 0,
    )
