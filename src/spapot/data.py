from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from .config import DataConfig
from .utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_SRC = PROJECT_ROOT / "src"
if str(EMBEDDING_SRC) not in sys.path:
    sys.path.insert(0, str(EMBEDDING_SRC))

from embedding.preprocessing.gene_prior_gatae import (  # noqa: E402
    GenePriorGATConfig,
    train_gene_prior_gatae,
)


def _to_dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def ensure_csr(adata: ad.AnnData) -> ad.AnnData:
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    for key in list(adata.layers.keys()):
        if sp.issparse(adata.layers[key]):
            adata.layers[key] = adata.layers[key].tocsr()
    return adata


@dataclass
class FeatureScaler:
    spatial_mean: np.ndarray
    spatial_std: np.ndarray
    latent_mean: np.ndarray
    latent_std: np.ndarray
    spatial_weight: float

    def transform(self, spatial: np.ndarray, latent: np.ndarray) -> np.ndarray:
        s = (spatial - self.spatial_mean) / self.spatial_std
        z = (latent - self.latent_mean) / self.latent_std
        return np.concatenate([s * self.spatial_weight, z], axis=1).astype(np.float32)

    def inverse(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = state[:, :2] / self.spatial_weight
        z = state[:, 2:]
        spatial = s * self.spatial_std + self.spatial_mean
        latent = z * self.latent_std + self.latent_mean
        return spatial.astype(np.float32), latent.astype(np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "spatial_mean": self.spatial_mean.tolist(),
            "spatial_std": self.spatial_std.tolist(),
            "latent_mean": self.latent_mean.tolist(),
            "latent_std": self.latent_std.tolist(),
            "spatial_weight": float(self.spatial_weight),
        }


@dataclass
class PreparedData:
    adata: ad.AnnData
    annotation_key: str
    time_values: list[float]
    raw_time_values: list[float]
    state_by_time: list[torch.Tensor]
    expression_by_time: list[torch.Tensor]
    labels_by_time: list[np.ndarray]
    raw_indices_by_time: list[np.ndarray]
    scaler: FeatureScaler
    spatial_dim: int
    latent_dim: int
    expression_dim: int


@dataclass
class SampledSlice:
    state: torch.Tensor
    expression: torch.Tensor
    labels: np.ndarray
    raw_indices: np.ndarray
    time_index: int
    time_value: float


def ensure_gat_embedding(config: DataConfig) -> tuple[Path, Path]:
    config.gat_out_dir.mkdir(parents=True, exist_ok=True)
    latent_h5ad = config.gat_out_dir / f"dorsal_midbrain_gene_prior_gatae_{config.gat_latent_dim}d_latent.h5ad"
    ready_h5ad = config.gat_out_dir / f"dorsal_midbrain_gene_prior_gatae_{config.gat_latent_dim}d_spapot_ready.h5ad"
    checkpoint = config.gat_out_dir / f"gene_prior_gatae_{config.gat_latent_dim}d.pt"
    summary = config.gat_out_dir / f"dorsal_midbrain_gene_prior_gatae_{config.gat_latent_dim}d.summary.json"
    trace = config.gat_out_dir / f"dorsal_midbrain_gene_prior_gatae_{config.gat_latent_dim}d.training_trace.jsonl"
    if ready_h5ad.exists() and checkpoint.exists() and not config.force_rebuild_gat:
        return ready_h5ad, checkpoint

    adata = ensure_csr(ad.read_h5ad(config.input_h5ad))
    adata.obs_names_make_unique()
    adata.obs[config.annotation_key] = adata.obs[config.annotation_key].astype("category")
    gat_config = GenePriorGATConfig(
        latent_dim=config.gat_latent_dim,
        max_epochs=config.gat_max_epochs,
        batch_size=config.gat_batch_size,
        max_prior_neighbors_per_gene=8,
        max_train_cells=1500,
        latent_var_weight=1e-3,
    )
    train_summary = train_gene_prior_gatae(
        adata,
        network_path=config.network_tsv,
        output_h5ad=latent_h5ad,
        checkpoint_path=checkpoint,
        summary_path=summary,
        trace_path=trace,
        layer_key=config.expression_layer_key,
        config=gat_config,
        device=config.gat_device,
    )
    latent = ensure_csr(ad.read_h5ad(latent_h5ad))
    latent_key = str(train_summary["latent_key"])
    z = np.asarray(latent.obsm[latent_key], dtype=np.float32)
    latent.obsm["X_gene_input"] = z
    latent.obsm["X_ae"] = z
    if "X_spatial_input" not in latent.obsm:
        latent.obsm["X_spatial_input"] = np.asarray(latent.obsm["spatial"], dtype=np.float32)
    latent.uns["full_psg_ruot_gat_embedding"] = {
        "gene_input_key": "X_gene_input",
        "latent_key": latent_key,
        "decoder_checkpoint_path": str(checkpoint),
        "source_input": str(config.input_h5ad),
    }
    latent.write_h5ad(ready_h5ad)
    write_json(
        config.gat_out_dir / "manifest.json",
        {
            "status": "DONE",
            "ready_h5ad": str(ready_h5ad),
            "checkpoint": str(checkpoint),
            "summary": str(summary),
            "trace": str(trace),
            "train_summary": train_summary,
        },
    )
    return ready_h5ad, checkpoint


def load_prepared_data(config: DataConfig, device: torch.device) -> tuple[PreparedData, Path | None]:
    if config.use_precomputed_embedding:
        ready_h5ad = config.input_h5ad
        decoder_checkpoint = config.decoder_checkpoint
    else:
        ready_h5ad, decoder_checkpoint = ensure_gat_embedding(config)
    adata = ensure_csr(ad.read_h5ad(ready_h5ad))
    adata.obs_names_make_unique()
    adata.obs[config.annotation_key] = adata.obs[config.annotation_key].astype("category")

    spatial = np.asarray(adata.obsm[config.spatial_key], dtype=np.float32)
    latent = np.asarray(adata.obsm[config.latent_key], dtype=np.float32)
    expression_matrix = adata.layers[config.expression_layer_key] if config.expression_layer_key in adata.layers else adata.X
    expression = _to_dense_float32(expression_matrix)

    spatial_std = np.where(spatial.std(axis=0) < 1e-6, 1.0, spatial.std(axis=0))
    latent_std = np.where(latent.std(axis=0) < 1e-6, 1.0, latent.std(axis=0))
    scaler = FeatureScaler(
        spatial_mean=spatial.mean(axis=0),
        spatial_std=spatial_std,
        latent_mean=latent.mean(axis=0),
        latent_std=latent_std,
        spatial_weight=float(config.spatial_weight),
    )
    state = scaler.transform(spatial, latent)
    time_values = sorted(float(v) for v in np.unique(np.asarray(adata.obs[config.time_key], dtype=np.float32)))
    raw_time_values: list[float] = []
    state_by_time: list[torch.Tensor] = []
    expression_by_time: list[torch.Tensor] = []
    labels_by_time: list[np.ndarray] = []
    raw_indices_by_time: list[np.ndarray] = []
    time_array = np.asarray(adata.obs[config.time_key], dtype=np.float32)
    raw_time_array = np.asarray(adata.obs[config.raw_time_key], dtype=np.float32)
    all_labels = adata.obs[config.annotation_key].astype(str).to_numpy()
    for time_value in time_values:
        mask = np.isclose(time_array, time_value)
        indices = np.flatnonzero(mask)
        raw_indices_by_time.append(indices)
        labels_by_time.append(all_labels[indices])
        raw_time_values.append(float(np.median(raw_time_array[indices])))
        state_by_time.append(torch.as_tensor(state[indices], dtype=torch.float32, device=device))
        expression_by_time.append(torch.as_tensor(expression[indices], dtype=torch.float32, device=device))
    return (
        PreparedData(
            adata=adata,
            annotation_key=config.annotation_key,
            time_values=time_values,
            raw_time_values=raw_time_values,
            state_by_time=state_by_time,
            expression_by_time=expression_by_time,
            labels_by_time=labels_by_time,
            raw_indices_by_time=raw_indices_by_time,
            scaler=scaler,
            spatial_dim=2,
            latent_dim=int(latent.shape[1]),
            expression_dim=int(expression.shape[1]),
        ),
        decoder_checkpoint,
    )


def sample_slice(data: PreparedData, time_index: int, sample_size: int) -> SampledSlice:
    state = data.state_by_time[time_index]
    expression = data.expression_by_time[time_index]
    labels = data.labels_by_time[time_index]
    raw_indices = data.raw_indices_by_time[time_index]
    n_obs = state.shape[0]
    if n_obs > sample_size:
        chosen = torch.randperm(n_obs, device=state.device)[:sample_size]
    else:
        chosen = torch.arange(n_obs, dtype=torch.long, device=state.device)
    chosen_cpu = chosen.detach().cpu().numpy()
    return SampledSlice(
        state=state[chosen],
        expression=expression[chosen],
        labels=labels[chosen_cpu],
        raw_indices=raw_indices[chosen_cpu],
        time_index=time_index,
        time_value=float(data.time_values[time_index]),
    )
