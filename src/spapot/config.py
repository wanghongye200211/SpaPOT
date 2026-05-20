from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    input_h5ad: Path
    network_tsv: Path
    gat_out_dir: Path
    use_precomputed_embedding: bool = False
    decoder_checkpoint: Path | None = None
    spatial_key: str = "X_spatial_input"
    latent_key: str = "X_gene_input"
    time_key: str = "time_input"
    raw_time_key: str = "time"
    annotation_key: str = "Annotation"
    expression_layer_key: str = "lognorm"
    spatial_weight: float = 3.0
    force_rebuild_gat: bool = False
    gat_latent_dim: int = 10
    gat_max_epochs: int = 160
    gat_batch_size: int = 96
    gat_device: str = "cpu"

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True)
class ModelConfig:
    spatial_dim: int = 2
    latent_dim: int = 10
    hidden_dim: int = 128
    n_hidden: int = 6
    activation: str = "relu"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    device: str = "mps"
    seed: int = 19491001
    epochs: int = 1000
    sample_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    integrator: str = "rk4"
    steps_per_interval: int = 8
    use_bidirectional: bool = True
    use_growth: bool = True
    lambda_match: float = 1.0
    lambda_action: float = 1.0
    alpha_exp: float = 0.01
    alpha_gro: float = 0.0002
    kappa_exp: float = 0.02
    kappa_gro: float = 0.1
    lambda_ssp: float = 0.0
    ssp_neighbors: int = 30
    lambda_hjb: float = 0.0
    hjb_start_epoch: int = 0
    hjb_ramp_epochs: int = 0
    use_cell_type_prior: bool = False
    cell_type_prior_min_count: int = 3
    trace_interval: int = 25
    grad_clip: float = 5.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
