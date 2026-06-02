from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint

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


class SpaPOTPotentialODE(nn.Module):
    def __init__(
        self,
        model: SpaPOTPotentialModel,
        *,
        alpha_exp: float,
        alpha_gro: float,
        direction_sign: float,
        use_growth: bool,
        neighbor_index: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.alpha_exp = float(alpha_exp)
        self.alpha_gro = float(alpha_gro)
        self.direction_sign = float(direction_sign)
        self.use_growth = bool(use_growth)
        self.neighbor_index = neighbor_index

    def forward(
        self,
        t: torch.Tensor,
        y: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, logw, action, ssp = y
        del action, ssp
        velocity, growth, aux = self.model(t, state)
        spatial_v = aux["spatial_velocity"]
        gene_v = aux["gene_velocity"]
        kinetic_rate = spatial_v.pow(2).sum(dim=1, keepdim=True) + self.alpha_exp * gene_v.pow(2).sum(dim=1, keepdim=True)
        if self.use_growth:
            action_rate = (kinetic_rate + self.alpha_gro * growth.pow(2)) * torch.exp(logw)
        else:
            action_rate = kinetic_rate * torch.ones_like(logw)
        ssp_rate = _spatial_smoothness(spatial_v, logw, self.neighbor_index)
        signed_action_rate = self.direction_sign * action_rate
        signed_ssp_rate = self.direction_sign * ssp_rate
        return velocity, growth, signed_action_rate, signed_ssp_rate


def rollout_spapot_potential(
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
    use_growth: bool = True,
    neighbor_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    method = method.lower()
    if method not in {"dopri5", "rk4", "euler", "midpoint"}:
        raise ValueError(f"Unsupported torchdiffeq method: {method}")
    direction_sign = 1.0 if float(t1) >= float(t0) else -1.0
    action0 = torch.zeros_like(logw0)
    ssp0 = torch.zeros_like(logw0)
    t_eval = torch.linspace(float(t0), float(t1), int(steps) + 1, dtype=state0.dtype, device=state0.device)
    ode_func = SpaPOTPotentialODE(
        model,
        alpha_exp=alpha_exp,
        alpha_gro=alpha_gro,
        direction_sign=direction_sign,
        use_growth=use_growth,
        neighbor_index=neighbor_index,
    )
    options = None
    if method in {"rk4", "euler", "midpoint"}:
        step_size = abs(float(t1) - float(t0)) / float(steps)
        options = {"step_size": step_size}
    state_t, logw_t, action_t, ssp_t = odeint(
        ode_func,
        (state0, logw0, action0, ssp0),
        t_eval,
        method=method,
        options=options,
    )
    return state_t[-1], logw_t[-1], action_t[-1], ssp_t[-1]


# Backward-compatible names used by earlier SpaPOT scripts.
HybridPotentialODE = SpaPOTPotentialODE
rollout_hybrid_potential = rollout_spapot_potential
