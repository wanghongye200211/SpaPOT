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
    hidden_dim: int = 256
    n_hidden: int = 3
    activation: str = "silu"
    potential_depends_on_spatial: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    device: str = "mps"
    seed: int = 19491001
    epochs: int = 500
    sample_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    integrator: str = "rk4"
    steps_per_interval: int = 8
    use_bidirectional: bool = True
    use_growth: bool = True
    lambda_state_ot: float = 1.0
    lambda_spatial_ot: float = 0.5
    lambda_expr: float = 0.05
    lambda_mass_global: float = 0.25
    lambda_mass_local: float = 1.0
    lambda_rollout_spatial_ot: float = 0.0
    lambda_rollout_mass_global: float = 0.0
    rollout_start_epoch: int = 0
    rollout_ramp_epochs: int = 0
    lambda_spatial_deform: float = 0.0
    lambda_spatial_coverage: float = 0.0
    spatial_deform_neighbors: int = 8
    spatial_coverage_bandwidth: float = 0.35
    spatial_coverage_anchor_count: int = 256
    lambda_action: float = 1e-3
    state_spatial_cost_weight: float = 0.7
    state_gene_cost_weight: float = 0.3
    action_gene_weight: float = 0.02
    action_growth_weight: float = 0.1
    use_cell_type_prior: bool = False
    cell_type_prior_min_count: int = 3
    local_mass_bandwidth: float = 0.75
    local_mass_anchor_count: int = 256
    decoder_chunk_size: int = 1
    detach_transport_plan: bool = True
    trace_interval: int = 25
    grad_clip: float = 5.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
