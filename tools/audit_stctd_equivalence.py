#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stctd.config import ModelConfig  # noqa: E402
from stctd.fields import STCTDModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit functional equivalence between a historical trajectory checkpoint and STCTDModel."
    )
    parser.add_argument("--reference-src", type=Path, required=True, help="Path to the reference source directory needed to load the checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Torch-saved historical trajectory-model checkpoint.")
    parser.add_argument("--spatial-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=6)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--time", type=float, default=7.5)
    parser.add_argument("--n-probe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _copy_modulelist(old_branch: torch.nn.Module, new_branch: torch.nn.Module, n_hidden: int) -> None:
    for i in range(n_hidden):
        new_branch.net[i][0].weight.data.copy_(old_branch.net[i][0].weight.data)
        new_branch.net[i][0].bias.data.copy_(old_branch.net[i][0].bias.data)
    new_branch.out.weight.data.copy_(old_branch.out.weight.data)
    new_branch.out.bias.data.copy_(old_branch.out.bias.data)


def _copy_growth(old_seq: torch.nn.Sequential, new_branch: torch.nn.Module) -> None:
    new_seq = new_branch.net
    for old_i, new_i in zip([0, 2, 4, 6], [0, 2, 4, 6]):
        new_seq[new_i].weight.data.copy_(old_seq[old_i].weight.data)
        new_seq[new_i].bias.data.copy_(old_seq[old_i].bias.data)


def main() -> None:
    args = parse_args()
    reference_src = args.reference_src.resolve()
    if str(reference_src) not in sys.path:
        sys.path.insert(0, str(reference_src))

    old_model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_model.eval()
    new_model = STCTDModel(
        ModelConfig(
            spatial_dim=int(args.spatial_dim),
            latent_dim=int(args.latent_dim),
            hidden_dim=int(args.hidden_dim),
            n_hidden=int(args.n_hidden),
            activation=str(args.activation),
        )
    ).cpu().eval()

    _copy_modulelist(old_model.spatial_velocity_net, new_model.spatial_net, int(args.n_hidden))
    _copy_modulelist(old_model.gene_velocity_net, new_model.potential_net, int(args.n_hidden))
    _copy_growth(old_model.growth_rate_net.net, new_model.growth_net)

    state_dim = int(args.spatial_dim) + int(args.latent_dim)
    torch.manual_seed(int(args.seed))
    state = torch.randn(int(args.n_probe), state_dim, dtype=torch.float32)
    time_value = torch.tensor(float(args.time), dtype=torch.float32)
    with torch.enable_grad():
        old_velocity, old_growth = old_model(time_value, (state.clone(), torch.zeros(state.shape[0], 1)))
        new_velocity, new_growth, aux = new_model(time_value, state.clone())

    payload = {
        "checkpoint": str(args.checkpoint),
        "reference_src": str(reference_src),
        "max_abs_velocity_diff": float((old_velocity - new_velocity).abs().max().detach()),
        "max_abs_growth_diff": float((old_growth - new_growth).abs().max().detach()),
        "spatial_diff": float((old_velocity[:, : args.spatial_dim] - aux["spatial_velocity"]).abs().max().detach()),
        "gene_diff": float((old_velocity[:, args.spatial_dim :] - aux["gene_velocity"]).abs().max().detach()),
        "old_velocity_shape": list(old_velocity.shape),
        "new_velocity_shape": list(new_velocity.shape),
        "old_growth_shape": list(old_growth.shape),
        "new_growth_shape": list(new_growth.shape),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    worst = max(payload["max_abs_velocity_diff"], payload["max_abs_growth_diff"])
    if worst > float(args.tolerance):
        raise SystemExit(f"Equivalence audit failed: worst diff {worst:g} exceeds tolerance {args.tolerance:g}")


if __name__ == "__main__":
    main()
