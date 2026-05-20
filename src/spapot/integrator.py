from __future__ import annotations

import torch

from .fields import SpaPOTPotentialModel


def _spatial_smoothness(
    spatial_velocity: torch.Tensor,
    logw: torch.Tensor,
    neighbor_index: torch.Tensor | None,
) -> torch.Tensor:
    if neighbor_index is None or neighbor_index.numel() == 0:
        return torch.zeros_like(logw)
    center = spatial_velocity.unsqueeze(1)
    neighbor = spatial_velocity[neighbor_index]
    local = (center - neighbor).pow(2).sum(dim=2).mean(dim=1, keepdim=True)
    return local * torch.exp(logw)


def _rhs(
    model: SpaPOTPotentialModel,
    t: torch.Tensor,
    state: torch.Tensor,
    logw: torch.Tensor,
    *,
    alpha_exp: float,
    alpha_gro: float,
    neighbor_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity, growth, aux = model(t, state)
    spatial_v = aux["spatial_velocity"]
    gene_v = aux["gene_velocity"]
    action = (
        spatial_v.pow(2).sum(dim=1, keepdim=True)
        + float(alpha_exp) * gene_v.pow(2).sum(dim=1, keepdim=True)
        + float(alpha_gro) * growth.pow(2)
    ) * torch.exp(logw)
    ssp = _spatial_smoothness(spatial_v, logw, neighbor_index)
    return velocity, growth, action, ssp


def integrate_fixed(
    model: SpaPOTPotentialModel,
    state0: torch.Tensor,
    logw0: torch.Tensor,
    t0: float,
    t1: float,
    *,
    steps: int,
    method: str,
    alpha_exp: float,
    alpha_gro: float,
    neighbor_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    method = method.lower()
    state = state0
    logw = logw0
    action = torch.zeros_like(logw0)
    ssp = torch.zeros_like(logw0)
    dt_value = float(t1 - t0) / float(steps)
    dt = torch.tensor(dt_value, dtype=state0.dtype, device=state0.device)
    t = torch.tensor(float(t0), dtype=state0.dtype, device=state0.device)
    for _ in range(steps):
        if method == "euler":
            kx, kw, ka, ks = _rhs(
                model,
                t,
                state,
                logw,
                alpha_exp=alpha_exp,
                alpha_gro=alpha_gro,
                neighbor_index=neighbor_index,
            )
            state = state + dt * kx
            logw = logw + dt * kw
            action = action + dt.abs() * ka
            ssp = ssp + dt.abs() * ks
        elif method == "rk4":
            k1x, k1w, k1a, k1s = _rhs(model, t, state, logw, alpha_exp=alpha_exp, alpha_gro=alpha_gro, neighbor_index=neighbor_index)
            k2x, k2w, k2a, k2s = _rhs(
                model,
                t + dt / 2,
                state + dt * k1x / 2,
                logw + dt * k1w / 2,
                alpha_exp=alpha_exp,
                alpha_gro=alpha_gro,
                neighbor_index=neighbor_index,
            )
            k3x, k3w, k3a, k3s = _rhs(
                model,
                t + dt / 2,
                state + dt * k2x / 2,
                logw + dt * k2w / 2,
                alpha_exp=alpha_exp,
                alpha_gro=alpha_gro,
                neighbor_index=neighbor_index,
            )
            k4x, k4w, k4a, k4s = _rhs(
                model,
                t + dt,
                state + dt * k3x,
                logw + dt * k3w,
                alpha_exp=alpha_exp,
                alpha_gro=alpha_gro,
                neighbor_index=neighbor_index,
            )
            state = state + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            logw = logw + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6
            action = action + dt.abs() * (k1a + 2 * k2a + 2 * k3a + k4a) / 6
            ssp = ssp + dt.abs() * (k1s + 2 * k2s + 2 * k3s + k4s) / 6
        else:
            raise ValueError(f"Unsupported integrator: {method}")
        t = t + dt
    return state, logw, action, ssp
