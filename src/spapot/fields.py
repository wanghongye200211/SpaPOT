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
        return nn.LeakyReLU(0.2)
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


class FullPSGRUOTModel(nn.Module):
    """Full potential-space-growth model.

    Spatial velocity and mass source are independent neural fields on `(s,z,t)`.
    The potential can be either `U(z,t)` for older checkpoints or `U(s,z,t)`;
    in both cases it only defines the gene latent velocity `dz/dt = -grad_z U`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_dim = int(config.spatial_dim)
        self.latent_dim = int(config.latent_dim)
        state_dim = self.spatial_dim + self.latent_dim
        self.potential_depends_on_spatial = bool(config.potential_depends_on_spatial)
        potential_input_dim = state_dim + 1 if self.potential_depends_on_spatial else self.latent_dim + 1
        self.potential_net = make_mlp(potential_input_dim, 1, config.hidden_dim, config.n_hidden, config.activation)
        self.spatial_net = make_mlp(state_dim + 1, self.spatial_dim, config.hidden_dim, config.n_hidden, config.activation)
        self.growth_net = make_mlp(state_dim + 1, 1, config.hidden_dim, config.n_hidden, config.activation)

    def _time_column(self, t: torch.Tensor | float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=dtype, device=device)
        return t.to(device=device, dtype=dtype).reshape(1, 1).expand(n, 1)

    def potential(self, t: torch.Tensor | float, z: torch.Tensor, spatial: torch.Tensor | None = None) -> torch.Tensor:
        t_col = self._time_column(t, z.shape[0], z.device, z.dtype)
        if self.potential_depends_on_spatial:
            if spatial is None:
                raise ValueError("This model uses U(s,z,t); pass spatial coordinates to potential().")
            return self.potential_net(torch.cat([t_col, spatial, z], dim=1))
        return self.potential_net(torch.cat([t_col, z], dim=1))

    def forward(self, t: torch.Tensor | float, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        s = state[:, : self.spatial_dim]
        z = state[:, self.spatial_dim :]
        z.requires_grad_(True)
        t_col = self._time_column(t, state.shape[0], state.device, state.dtype)
        u = self.potential(t, z, s)
        dz = -torch.autograd.grad(u.sum(), z, create_graph=True, retain_graph=True)[0]
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
