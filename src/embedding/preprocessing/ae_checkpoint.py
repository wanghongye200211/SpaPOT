from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from embedding.preprocessing.gene_prior_gatae import GenePriorGATConfig, load_gene_prior_gatae_decoder


@dataclass(frozen=True)
class FrozenDecoderBundle:
    checkpoint_path: Path
    latent_dim: int
    output_dim: int
    decoder: nn.Module


def _load_full_model(checkpoint_path: Path, device: torch.device | str):
    """Load a full model object across pre/post PyTorch 2.6 behavior."""

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_frozen_decoder(checkpoint_path: str | Path, device: torch.device | str) -> FrozenDecoderBundle:
    """Load the decoder from a pretrained embedding AE checkpoint and freeze it.

    The local preprocessing pipeline saves the whole model object with `torch.save(model, path)`.
    This helper keeps that contract and exposes only the decoder branch for the future
    joint-AE training loop.
    """

    checkpoint_path = Path(checkpoint_path)
    model = _load_full_model(checkpoint_path, device)

    if isinstance(model, dict) and model.get("model_type") == "gene_prior_gatae":
        decoder = load_gene_prior_gatae_decoder(checkpoint_path, device)
        config = GenePriorGATConfig(**model["config"])
        return FrozenDecoderBundle(
            checkpoint_path=checkpoint_path,
            latent_dim=int(config.latent_dim),
            output_dim=int(len(model["gene_names"])),
            decoder=decoder,
        )

    if not hasattr(model, "decode_mlp1"):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not expose `decode_mlp1`; "
            "it does not look like a compatible embedding AE model."
        )

    decoder = model.decode_mlp1.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False

    latent_dim = int(decoder[0].in_features)
    output_dim = None
    for module in reversed(list(decoder)):
        if hasattr(module, "out_features"):
            output_dim = int(module.out_features)
            break
    if output_dim is None:
        raise ValueError(f"Could not infer decoder output dimension from checkpoint {checkpoint_path}.")
    return FrozenDecoderBundle(
        checkpoint_path=checkpoint_path,
        latent_dim=latent_dim,
        output_dim=output_dim,
        decoder=decoder,
    )


def decode_gene_latent(bundle: FrozenDecoderBundle, latent: torch.Tensor) -> torch.Tensor:
    """Decode predicted gene-latent states into expression space."""

    if latent.shape[-1] != bundle.latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: got {latent.shape[-1]}, expected {bundle.latent_dim}."
        )
    if hasattr(bundle.decoder, "num_genes"):
        chunk_size = 1 if latent.device.type == "mps" else 32
        if latent.shape[0] <= chunk_size:
            if latent.device.type == "mps" and latent.requires_grad:
                return checkpoint(lambda part: bundle.decoder(part), latent, use_reentrant=False)
            return bundle.decoder(latent)
        chunks = [
            checkpoint(lambda part: bundle.decoder(part), part, use_reentrant=False)
            if latent.device.type == "mps" and part.requires_grad
            else bundle.decoder(part)
            for part in torch.split(latent, chunk_size, dim=0)
        ]
        return torch.cat(chunks, dim=0)
    return bundle.decoder(latent)
