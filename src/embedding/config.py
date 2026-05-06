from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DecoderPolicy = Literal["frozen", "finetune"]
ReconTargetMode = Literal["transport_plan", "matched_indices"]


@dataclass(frozen=True)
class DatasetContract:
    source_h5ad: Path
    expression_layer_key: str = "lognorm"
    gene_input_key: str = "X_gene_input"
    spatial_input_key: str = "X_spatial_input"
    time_input_key: str = "time_input"


@dataclass(frozen=True)
class DecoderContract:
    checkpoint_path: Path
    latent_key: str = "X_ae"
    latent_dim: int = 10
    policy: DecoderPolicy = "frozen"


@dataclass(frozen=True)
class JointAEObjective:
    enabled: bool = True
    lambda_ae: float = 0.1
    target_mode: ReconTargetMode = "transport_plan"
    use_forward_terms: bool = True
    use_backward_terms: bool = True
    normalize_by_plan_mass: bool = True
    detach_transport_plan: bool = True


@dataclass(frozen=True)
class ForkRuntime:
    device: str = "cpu"
    seed: int = 666
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class JointAEExperimentConfig:
    name: str
    dataset: DatasetContract
    decoder: DecoderContract
    objective: JointAEObjective
    runtime: ForkRuntime = field(default_factory=ForkRuntime)
