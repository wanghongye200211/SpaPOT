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


class HyperNetwork1(nn.Module):
    """stCTD direct spatial vector field."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hiddens: int, activation: str = "relu") -> None:
        super().__init__()
        layers = [int(in_dim) + 1]
        for _ in range(int(n_hiddens)):
            layers.append(int(hidden_dim))
        layers.append(int(out_dim))
        self.activation = _activation(activation)
        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    self.activation,
                )
                for i in range(len(layers) - 2)
            ]
        )
        self.out = nn.Linear(layers[-2], layers[-1])

    def _time_column(self, t: torch.Tensor | float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=dtype, device=device)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.repeat(n).reshape(n, 1)
        if t.ndim == 1:
            return t.reshape(n, 1)
        return t

    def forward(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        state = torch.cat((self._time_column(t, x.shape[0], x.device, x.dtype), x), dim=1)
        hidden = state
        for layer in self.net:
            hidden = layer(hidden)
        return self.out(hidden)


class PotentialGradientNetwork(nn.Module):
    """stCTD scalar potential with velocity = -grad U."""

    def __init__(self, in_dim: int, out_slice: slice, hidden_dim: int, n_hiddens: int, activation: str = "relu") -> None:
        super().__init__()
        self.out_slice = out_slice
        layers = [int(in_dim) + 1]
        for _ in range(int(n_hiddens)):
            layers.append(int(hidden_dim))
        layers.append(1)
        self.activation = _activation(activation)
        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    self.activation,
                )
                for i in range(len(layers) - 2)
            ]
        )
        self.out = nn.Linear(layers[-2], layers[-1])

    def _prepare_state(self, t: torch.Tensor | float, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if x.requires_grad:
                state_x = x
            else:
                state_x = x.clone().requires_grad_(True)

            batchsize = state_x.shape[0]
            t = torch.as_tensor(t, dtype=state_x.dtype, device=state_x.device)
            if t.ndim == 0:
                t_batch = t.repeat(batchsize).reshape(batchsize, 1)
            elif t.ndim == 1:
                t_batch = t.reshape(batchsize, 1)
            else:
                t_batch = t

            if t_batch.requires_grad:
                state_t = t_batch
            else:
                state_t = t_batch.clone().requires_grad_(True)

            state = torch.cat((state_t, state_x), dim=1)
            hidden = state
            for layer in self.net:
                hidden = layer(hidden)
            potential = self.out(hidden)
            return potential, state_x, state_t

    def potential(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        potential, _, _ = self._prepare_state(t, x)
        return potential

    def potential_and_gradient(self, t: torch.Tensor | float, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        potential, state_x, state_t = self._prepare_state(t, x)
        grad = torch.autograd.grad(potential.sum(), state_x, create_graph=True, retain_graph=True)[0]
        return potential, grad, state_t

    def potential_gradients_with_time(
        self,
        t: torch.Tensor | float,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        potential, state_x, state_t = self._prepare_state(t, x)
        grad_x = torch.autograd.grad(potential.sum(), state_x, create_graph=True, retain_graph=True)[0]
        grad_t = torch.autograd.grad(potential.sum(), state_t, create_graph=True, retain_graph=True)[0]
        return potential, grad_x, grad_t

    def forward(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        _, grad, _ = self.potential_and_gradient(t, x)
        return -grad[:, self.out_slice]


class GenePotentialGradient(PotentialGradientNetwork):
    """stCTD gene-latent dynamics: dz/dt = -grad_z U(s,z,t)."""

    def __init__(self, full_dim: int, spatial_dim: int, hidden_dim: int, n_hiddens: int, activation: str = "relu") -> None:
        super().__init__(
            in_dim=int(full_dim),
            out_slice=slice(int(spatial_dim), int(full_dim)),
            hidden_dim=int(hidden_dim),
            n_hiddens=int(n_hiddens),
            activation=activation,
        )


class HyperNetwork2(nn.Module):
    """stCTD growth field with three hidden layers."""

    def __init__(self, in_out_dim: int, hidden_dim: int, activation: str = "relu") -> None:
        super().__init__()
        self.activation = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(int(in_out_dim) + 1, int(hidden_dim)),
            self.activation,
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            self.activation,
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            self.activation,
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.repeat(batchsize).reshape(batchsize, 1)
        elif t.ndim == 1:
            t = t.reshape(batchsize, 1)
        t = t.clone().requires_grad_(True)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class STCTDModel(nn.Module):
    """Coupled spatial, molecular-potential, and source-sink dynamics.

    The manuscript model follows this structure:

        ds/dt      = spatial_net(s, z, t)
        dz/dt      = -grad_z U(s, z, t)
        d log w/dt = growth_net(s, z, t)

    Spatial motion is a direct neural field, while gene-latent motion is
    constrained by a scalar potential landscape. The growth branch intentionally
    uses a three-hidden-layer MLP; it does not follow ``config.n_hidden``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_dim = int(config.spatial_dim)
        self.latent_dim = int(config.latent_dim)
        self.velocity_parameterization = str(getattr(config, "velocity_parameterization", "potential")).lower()
        if self.velocity_parameterization not in {"potential", "vector"}:
            raise ValueError("velocity_parameterization must be 'potential' or 'vector'.")
        state_dim = self.spatial_dim + self.latent_dim
        self.spatial_velocity_net = HyperNetwork1(
            in_dim=state_dim,
            out_dim=self.spatial_dim,
            hidden_dim=config.hidden_dim,
            n_hiddens=config.n_hidden,
            activation=config.activation,
        )
        if self.velocity_parameterization == "potential":
            self.gene_velocity_net = GenePotentialGradient(
                full_dim=state_dim,
                spatial_dim=self.spatial_dim,
                hidden_dim=config.hidden_dim,
                n_hiddens=config.n_hidden,
                activation=config.activation,
            )
        else:
            self.gene_velocity_net = HyperNetwork1(
                in_dim=state_dim,
                out_dim=self.latent_dim,
                hidden_dim=config.hidden_dim,
                n_hiddens=config.n_hidden,
                activation=config.activation,
            )
        self.growth_rate_net = HyperNetwork2(state_dim, config.hidden_dim, config.activation)

    @property
    def spatial_net(self) -> nn.Module:
        return self.spatial_velocity_net

    @property
    def potential_net(self) -> nn.Module:
        return self.gene_velocity_net

    @property
    def growth_net(self) -> nn.Module:
        return self.growth_rate_net

    def _time_column(self, t: torch.Tensor | float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=dtype, device=device)
        return t.to(device=device, dtype=dtype).reshape(1, 1).expand(n, 1)

    def potential(self, t: torch.Tensor | float, z: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        if self.velocity_parameterization != "potential":
            raise ValueError("Direct-vector stCTD has no scalar gene potential.")
        state = torch.cat([spatial, z], dim=1)
        return self.gene_velocity_net.potential(t, state)

    def potential_and_gene_gradient(
        self,
        t: torch.Tensor | float,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.velocity_parameterization != "potential":
            raise ValueError("Direct-vector stCTD has no scalar gene potential.")
        u, grad_state, _ = self.gene_velocity_net.potential_and_gradient(t, state)
        return u, grad_state[:, self.spatial_dim :]

    def potential_derivatives(
        self,
        t: torch.Tensor | float,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.velocity_parameterization != "potential":
            raise ValueError("Direct-vector stCTD has no scalar gene potential.")
        u, grad_state, grad_t = self.gene_velocity_net.potential_gradients_with_time(t, state)
        return u, grad_state[:, self.spatial_dim :], grad_t

    def forward(self, t: torch.Tensor | float, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.velocity_parameterization == "potential":
            u, grad_z = self.potential_and_gene_gradient(t, state)
            dz = -grad_z
        else:
            dz = self.gene_velocity_net(t, state)
            u = torch.zeros((state.shape[0], 1), dtype=state.dtype, device=state.device)
        ds = self.spatial_velocity_net(t, state)
        growth = self.growth_rate_net(t, state)
        velocity = torch.cat([ds, dz], dim=1)
        aux = {
            "potential": u,
            "spatial_velocity": ds,
            "gene_velocity": dz,
            "growth": growth,
        }
        return velocity.float(), growth.float(), aux
