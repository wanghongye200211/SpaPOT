from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "elu":
        return nn.ELU()
    if key == "leakyrelu":
        return nn.LeakyReLU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(cur, hidden_dim))
        layers.append(_activation(activation))
        cur = hidden_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class SpaPOTPotentialModel(nn.Module):
    """SpaPOT Hybrid dynamics.

    This is the fixed model family used by the current SpaPOT reconstruction
    path:

        ds/dt      = spatial_net(s, z, t)
        dz/dt      = -grad_z U(s, z, t)
        d log w/dt = growth_net(s, z, t)

    Spatial motion is a direct neural field, while gene-latent motion is
    constrained by a scalar potential landscape. The growth branch intentionally
    uses the SpaPOT Hybrid three-hidden-layer MLP; it does not follow
    ``config.n_hidden``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_dim = int(config.spatial_dim)
        self.latent_dim = int(config.latent_dim)
        state_dim = self.spatial_dim + self.latent_dim
        self.potential_net = make_mlp(state_dim + 1, 1, config.hidden_dim, config.n_hidden, config.activation)
        self.spatial_net = make_mlp(state_dim + 1, self.spatial_dim, config.hidden_dim, config.n_hidden, config.activation)
        self.growth_net = make_mlp(state_dim + 1, 1, config.hidden_dim, 3, config.activation)

    def _time_column(self, t: torch.Tensor | float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=dtype, device=device)
        return t.to(device=device, dtype=dtype).reshape(1, 1).expand(n, 1)

    def potential(self, t: torch.Tensor | float, z: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        t_col = self._time_column(t, z.shape[0], z.device, z.dtype)
        return self.potential_net(torch.cat([t_col, spatial, z], dim=1))

    def potential_and_gene_gradient(
        self,
        t: torch.Tensor | float,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = state[:, : self.spatial_dim]
        z = state[:, self.spatial_dim :]
        if not z.requires_grad:
            z = z.clone().requires_grad_(True)
        u = self.potential(t, z, s)
        grad_z = torch.autograd.grad(u.sum(), z, create_graph=True, retain_graph=True)[0]
        return u, grad_z

    def potential_derivatives(
        self,
        t: torch.Tensor | float,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = state[:, : self.spatial_dim]
        z = state[:, self.spatial_dim :].clone().requires_grad_(True)
        if torch.is_tensor(t):
            t_base = t.to(device=state.device, dtype=state.dtype)
        else:
            t_base = torch.tensor(float(t), dtype=state.dtype, device=state.device)
        t_col = t_base.reshape(1, 1).expand(state.shape[0], 1).clone().requires_grad_(True)
        u = self.potential_net(torch.cat([t_col, s, z], dim=1))
        grad_z = torch.autograd.grad(u.sum(), z, create_graph=True, retain_graph=True)[0]
        grad_t = torch.autograd.grad(u.sum(), t_col, create_graph=True, retain_graph=True)[0]
        return u, grad_z, grad_t

    def forward(self, t: torch.Tensor | float, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        s = state[:, : self.spatial_dim]
        z = state[:, self.spatial_dim :]
        t_col = self._time_column(t, state.shape[0], state.device, state.dtype)
        u, grad_z = self.potential_and_gene_gradient(t, state)
        dz = -grad_z
        sz = torch.cat([t_col, s, z], dim=1)
        ds = self.spatial_net(sz)
        growth = self.growth_net(sz)
        velocity = torch.cat([ds, dz], dim=1)
        aux = {
            "potential": u,
            "spatial_velocity": ds,
            "gene_velocity": dz,
            "growth": growth,
        }
        return velocity.float(), growth.float(), aux
