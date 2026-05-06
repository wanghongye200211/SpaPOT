from __future__ import annotations

import torch

from .fields import FullPSGRUOTModel


def _rhs(
    model: FullPSGRUOTModel,
    t: torch.Tensor,
    state: torch.Tensor,
    logw: torch.Tensor,
    *,
    action_gene_weight: float,
    action_growth_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity, growth, aux = model(t, state)
    spatial_v = aux["spatial_velocity"]
    gene_v = aux["gene_velocity"]
    weighted_action = (
        spatial_v.pow(2).sum(dim=1, keepdim=True)
        + float(action_gene_weight) * gene_v.pow(2).sum(dim=1, keepdim=True)
        + float(action_growth_weight) * growth.pow(2)
    ) * torch.exp(logw)
    return velocity, growth, weighted_action


def integrate_fixed(
    model: FullPSGRUOTModel,
    state0: torch.Tensor,
    logw0: torch.Tensor,
    t0: float,
    t1: float,
    *,
    steps: int,
    method: str,
    action_gene_weight: float,
    action_growth_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    method = method.lower()
    state = state0
    logw = logw0
    action = torch.zeros_like(logw0)
    dt_value = float(t1 - t0) / float(steps)
    dt = torch.tensor(dt_value, dtype=state0.dtype, device=state0.device)
    t = torch.tensor(float(t0), dtype=state0.dtype, device=state0.device)
    for _ in range(steps):
        if method == "euler":
            kx, kw, ka = _rhs(
                model,
                t,
                state,
                logw,
                action_gene_weight=action_gene_weight,
                action_growth_weight=action_growth_weight,
            )
            state = state + dt * kx
            logw = logw + dt * kw
            action = action + dt.abs() * ka
        elif method == "rk4":
            k1x, k1w, k1a = _rhs(model, t, state, logw, action_gene_weight=action_gene_weight, action_growth_weight=action_growth_weight)
            k2x, k2w, k2a = _rhs(model, t + dt / 2, state + dt * k1x / 2, logw + dt * k1w / 2, action_gene_weight=action_gene_weight, action_growth_weight=action_growth_weight)
            k3x, k3w, k3a = _rhs(model, t + dt / 2, state + dt * k2x / 2, logw + dt * k2w / 2, action_gene_weight=action_gene_weight, action_growth_weight=action_growth_weight)
            k4x, k4w, k4a = _rhs(model, t + dt, state + dt * k3x, logw + dt * k3w, action_gene_weight=action_gene_weight, action_growth_weight=action_growth_weight)
            state = state + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            logw = logw + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6
            action = action + dt.abs() * (k1a + 2 * k2a + 2 * k3a + k4a) / 6
        else:
            raise ValueError(f"Unsupported integrator: {method}")
        t = t + dt
    return state, logw, action

