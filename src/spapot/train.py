from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .data import PreparedData, SampledSlice, sample_slice
from .fields import SpaPOTPotentialModel
from .integrator import integrate_fixed
from .losses import (
    global_mass_ratio_loss,
    local_soft_mass_loss,
    spatial_undercoverage_loss,
    weighted_emd_plan,
    weighted_expression_reconstruction_loss,
    weighted_spatial_emd_loss,
)
from .utils import append_jsonl, clear_cache, seed_all


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_SRC = PROJECT_ROOT / "src"
if str(EMBEDDING_SRC) not in sys.path:
    sys.path.insert(0, str(EMBEDDING_SRC))

from embedding.preprocessing.ae_checkpoint import decode_gene_latent, load_frozen_decoder  # noqa: E402


@dataclass
class DecoderRuntime:
    bundle: Any
    latent_mean: torch.Tensor | None
    latent_std: torch.Tensor | None

    def to_decoder_latent(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_std is None:
            return standardized_latent
        return standardized_latent * self.latent_std + self.latent_mean


def load_decoder_runtime(checkpoint_path: Path, device: torch.device) -> DecoderRuntime:
    bundle = load_frozen_decoder(checkpoint_path, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latent_mean = None
    latent_std = None
    if isinstance(payload, dict) and payload.get("model_type") == "gene_prior_gatae":
        latent_mean = torch.as_tensor(payload["latent_mean"], dtype=torch.float32, device=device).reshape(1, -1)
        latent_std = torch.as_tensor(payload["latent_std"], dtype=torch.float32, device=device).reshape(1, -1)
    return DecoderRuntime(bundle=bundle, latent_mean=latent_mean, latent_std=latent_std)


def _decode_pred_expression(
    pred_state: torch.Tensor,
    data: PreparedData,
    decoder: DecoderRuntime,
) -> torch.Tensor:
    scaled_latent = pred_state[:, data.spatial_dim :]
    mean = torch.as_tensor(data.scaler.latent_mean, dtype=scaled_latent.dtype, device=scaled_latent.device).reshape(1, -1)
    std = torch.as_tensor(data.scaler.latent_std, dtype=scaled_latent.dtype, device=scaled_latent.device).reshape(1, -1)
    standardized_latent = scaled_latent * std + mean
    decoder_latent = decoder.to_decoder_latent(standardized_latent)
    return decode_gene_latent(decoder.bundle, decoder_latent)


def _rollout(
    model: SpaPOTPotentialModel,
    source: SampledSlice,
    target_time: float,
    train_config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logw0 = torch.zeros(source.state.shape[0], 1, dtype=source.state.dtype, device=source.state.device)
    return integrate_fixed(
        model,
        source.state,
        logw0,
        source.time_value,
        target_time,
        steps=train_config.steps_per_interval,
        method=train_config.integrator,
        action_gene_weight=train_config.action_gene_weight,
        action_growth_weight=train_config.action_growth_weight,
    )


def _spatial_deformation_loss(
    state: torch.Tensor,
    spatial_velocity: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    spatial_dim: int,
    n_neighbors: int,
) -> torch.Tensor:
    n_obs = int(state.shape[0])
    if n_obs <= 2 or n_neighbors <= 0:
        return state.new_zeros(())
    k = min(int(n_neighbors), n_obs - 1)
    spatial = state[:, :spatial_dim]
    dist = torch.cdist(spatial, spatial)
    _, nn_idx = torch.topk(dist, k=k + 1, largest=False)
    nn_idx = nn_idx[:, 1:]
    center_s = spatial.unsqueeze(1)
    neigh_s = spatial[nn_idx]
    center_v = spatial_velocity.unsqueeze(1)
    neigh_v = spatial_velocity[nn_idx]
    direction = center_s - neigh_s
    direction = direction / direction.norm(dim=2, keepdim=True).clamp_min(1e-6)
    radial_rel_velocity = ((center_v - neigh_v) * direction).sum(dim=2).pow(2)
    if weights is not None:
        radial_rel_velocity = radial_rel_velocity * weights.reshape(-1, 1)
    return radial_rel_velocity.mean()


def _source_spatial_deformation_loss(
    model: SpaPOTPotentialModel,
    source: SampledSlice,
    data: PreparedData,
    train_config: TrainConfig,
) -> torch.Tensor:
    if train_config.lambda_spatial_deform <= 0:
        return source.state.new_zeros(())
    _, _, aux = model(source.time_value, source.state)
    return _spatial_deformation_loss(
        source.state,
        aux["spatial_velocity"],
        None,
        spatial_dim=data.spatial_dim,
        n_neighbors=train_config.spatial_deform_neighbors,
    )


def _pred_spatial_deformation_loss(
    model: SpaPOTPotentialModel,
    pred_state: torch.Tensor,
    pred_weights: torch.Tensor,
    target_time: float,
    data: PreparedData,
    train_config: TrainConfig,
) -> torch.Tensor:
    if train_config.lambda_spatial_deform <= 0:
        return pred_state.new_zeros(())
    _, _, aux = model(target_time, pred_state)
    return _spatial_deformation_loss(
        pred_state,
        aux["spatial_velocity"],
        pred_weights,
        spatial_dim=data.spatial_dim,
        n_neighbors=train_config.spatial_deform_neighbors,
    )


def _cell_type_prior_weighted_emd(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    pred_weights: torch.Tensor,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    spatial_dim: int,
    spatial_cost_weight: float,
    gene_cost_weight: float,
    min_count: int,
) -> torch.Tensor:
    total = pred_state.new_zeros(())
    total_group_weight = pred_state.new_zeros(())
    labels = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    n_target = max(int(target_labels.shape[0]), 1)
    for label in labels:
        source_mask_np = source_labels == label
        target_mask_np = target_labels == label
        if int(source_mask_np.sum()) < min_count or int(target_mask_np.sum()) < min_count:
            continue
        source_mask = torch.as_tensor(source_mask_np, dtype=torch.bool, device=pred_state.device)
        target_mask = torch.as_tensor(target_mask_np, dtype=torch.bool, device=target_state.device)
        group_loss, _, _ = weighted_emd_plan(
            pred_state[source_mask],
            target_state[target_mask],
            pred_weights[source_mask],
            None,
            spatial_dim=spatial_dim,
            spatial_cost_weight=spatial_cost_weight,
            gene_cost_weight=gene_cost_weight,
        )
        group_weight = torch.as_tensor(float(target_mask_np.sum()) / float(n_target), dtype=pred_state.dtype, device=pred_state.device)
        total = total + group_weight * group_loss
        total_group_weight = total_group_weight + group_weight
    if float(total_group_weight.detach().cpu()) <= 0:
        fallback, _, _ = weighted_emd_plan(
            pred_state,
            target_state,
            pred_weights,
            None,
            spatial_dim=spatial_dim,
            spatial_cost_weight=spatial_cost_weight,
            gene_cost_weight=gene_cost_weight,
        )
        return fallback
    return total / total_group_weight.clamp_min(1e-8)


def _interval_loss(
    model: SpaPOTPotentialModel,
    data: PreparedData,
    decoder: DecoderRuntime | None,
    source_index: int,
    target_index: int,
    train_config: TrainConfig,
) -> dict[str, torch.Tensor]:
    source = sample_slice(data, source_index, train_config.sample_size)
    target = sample_slice(data, target_index, train_config.sample_size)
    pred_state, pred_logw, action = _rollout(model, source, target.time_value, train_config)
    pred_weights = torch.exp(pred_logw).reshape(-1)
    if not train_config.use_growth:
        pred_weights = torch.ones_like(pred_weights)
    spatial_deform = 0.5 * (
        _source_spatial_deformation_loss(model, source, data, train_config)
        + _pred_spatial_deformation_loss(model, pred_state, pred_weights, target.time_value, data, train_config)
    )

    expected_ratio = data.state_by_time[target_index].shape[0] / data.state_by_time[source_index].shape[0]
    plan = None
    if train_config.use_cell_type_prior:
        state_ot = _cell_type_prior_weighted_emd(
            pred_state,
            target.state,
            pred_weights,
            source.labels,
            target.labels,
            spatial_dim=data.spatial_dim,
            spatial_cost_weight=train_config.state_spatial_cost_weight,
            gene_cost_weight=train_config.state_gene_cost_weight,
            min_count=train_config.cell_type_prior_min_count,
        )
    else:
        state_ot, plan, _ = weighted_emd_plan(
            pred_state,
            target.state,
            pred_weights,
            None,
            spatial_dim=data.spatial_dim,
            spatial_cost_weight=train_config.state_spatial_cost_weight,
            gene_cost_weight=train_config.state_gene_cost_weight,
        )
    mass_global = global_mass_ratio_loss(pred_weights, expected_ratio) if train_config.use_growth else pred_weights.new_zeros(())
    matching = state_ot + train_config.lambda_mass_global * mass_global
    if train_config.lambda_spatial_ot > 0:
        spatial_ot = weighted_spatial_emd_loss(
            pred_state,
            target.state,
            pred_weights,
            None,
            spatial_dim=data.spatial_dim,
        )
    else:
        spatial_ot = pred_state.new_zeros(())
    spatial_coverage = (
        spatial_undercoverage_loss(
            pred_state,
            target.state,
            pred_weights,
            spatial_dim=data.spatial_dim,
            bandwidth=train_config.spatial_coverage_bandwidth,
            anchor_count=train_config.spatial_coverage_anchor_count,
        )
        if train_config.lambda_spatial_coverage > 0
        else pred_state.new_zeros(())
    )
    mass_local = (
        local_soft_mass_loss(
            pred_state,
            target.state,
            pred_weights,
            expected_ratio,
            bandwidth=train_config.local_mass_bandwidth,
            anchor_count=train_config.local_mass_anchor_count,
        )
        if train_config.use_growth and train_config.lambda_mass_local > 0
        else pred_weights.new_zeros(())
    )
    if train_config.lambda_expr > 0:
        if decoder is None:
            raise ValueError("lambda_expr > 0 requires a decoder checkpoint.")
        if plan is None:
            _, plan, _ = weighted_emd_plan(
                pred_state,
                target.state,
                pred_weights,
                None,
                spatial_dim=data.spatial_dim,
                spatial_cost_weight=train_config.state_spatial_cost_weight,
                gene_cost_weight=train_config.state_gene_cost_weight,
            )
        decoded = _decode_pred_expression(pred_state, data, decoder)
        expr = weighted_expression_reconstruction_loss(
            decoded,
            target.expression,
            plan,
            detach_plan=train_config.detach_transport_plan,
        )
    else:
        expr = pred_state.new_zeros(())
    action_loss = action.mean()
    total = (
        train_config.lambda_state_ot * matching
        + train_config.lambda_spatial_ot * spatial_ot
        + train_config.lambda_expr * expr
        + train_config.lambda_mass_local * mass_local
        + train_config.lambda_spatial_deform * spatial_deform
        + train_config.lambda_spatial_coverage * spatial_coverage
        + train_config.lambda_action * action_loss
    )
    return {
        "total": total,
        "matching": matching.detach(),
        "state_ot": state_ot.detach(),
        "spatial_ot": spatial_ot.detach(),
        "expr": expr.detach(),
        "mass_global": mass_global.detach(),
        "mass_local": mass_local.detach(),
        "spatial_deform": spatial_deform.detach(),
        "spatial_coverage": spatial_coverage.detach(),
        "action": action_loss.detach(),
        "pred_ratio": pred_weights.mean().detach(),
        "expected_ratio": torch.as_tensor(float(expected_ratio), dtype=pred_weights.dtype, device=pred_weights.device),
    }


def _rollout_loss(
    model: SpaPOTPotentialModel,
    data: PreparedData,
    target_index: int,
    train_config: TrainConfig,
) -> dict[str, torch.Tensor]:
    source = sample_slice(data, 0, train_config.sample_size)
    target = sample_slice(data, target_index, train_config.sample_size)
    pred_state, pred_logw, action = _rollout(model, source, target.time_value, train_config)
    pred_weights = torch.exp(pred_logw).reshape(-1)
    if not train_config.use_growth:
        pred_weights = torch.ones_like(pred_weights)

    expected_ratio = data.state_by_time[target_index].shape[0] / data.state_by_time[0].shape[0]
    spatial_ot = (
        weighted_spatial_emd_loss(
            pred_state,
            target.state,
            pred_weights,
            None,
            spatial_dim=data.spatial_dim,
        )
        if train_config.lambda_rollout_spatial_ot > 0
        else pred_state.new_zeros(())
    )
    mass_global = (
        global_mass_ratio_loss(pred_weights, expected_ratio)
        if train_config.use_growth and train_config.lambda_rollout_mass_global > 0
        else pred_state.new_zeros(())
    )
    action_loss = action.mean()
    total = (
        train_config.lambda_rollout_spatial_ot * spatial_ot
        + train_config.lambda_rollout_mass_global * mass_global
        + train_config.lambda_action * action_loss
    )
    zero = pred_state.new_zeros(())
    return {
        "total": total,
        "matching": zero.detach(),
        "state_ot": zero.detach(),
        "spatial_ot": spatial_ot.detach(),
        "expr": zero.detach(),
        "mass_global": mass_global.detach(),
        "mass_local": zero.detach(),
        "spatial_deform": zero.detach(),
        "spatial_coverage": zero.detach(),
        "action": action_loss.detach(),
        "pred_ratio": pred_weights.mean().detach(),
        "expected_ratio": torch.as_tensor(float(expected_ratio), dtype=pred_weights.dtype, device=pred_weights.device),
    }


def _rollout_scale(epoch: int, train_config: TrainConfig) -> float:
    if train_config.lambda_rollout_spatial_ot <= 0 and train_config.lambda_rollout_mass_global <= 0:
        return 0.0
    if epoch < train_config.rollout_start_epoch:
        return 0.0
    if train_config.rollout_ramp_epochs <= 0:
        return 1.0
    progress = (epoch - train_config.rollout_start_epoch + 1) / float(train_config.rollout_ramp_epochs)
    return float(max(0.0, min(1.0, progress)))


def train_spapot_model(
    data: PreparedData,
    decoder_checkpoint: Path | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    output_dir: Path,
) -> tuple[SpaPOTPotentialModel, dict[str, Any]]:
    seed_all(train_config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    device = data.state_by_time[0].device
    model = SpaPOTPotentialModel(model_config).to(device)
    decoder = load_decoder_runtime(decoder_checkpoint, device) if decoder_checkpoint is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    started = time.time()
    n_intervals = len(data.time_values) - 1
    if n_intervals <= 0:
        raise ValueError("Need at least two time points.")

    for epoch in range(train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rows = []
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        for idx in range(n_intervals):
            row = _interval_loss(model, data, decoder, idx, idx + 1, train_config)
            total_loss = total_loss + row["total"]
            rows.append(("forward", idx, idx + 1, row))
            if train_config.use_bidirectional:
                back = _interval_loss(model, data, decoder, idx + 1, idx, train_config)
                total_loss = total_loss + back["total"]
                rows.append(("backward", idx + 1, idx, back))
        rollout_scale = _rollout_scale(epoch, train_config)
        if rollout_scale > 0:
            for target_idx in range(1, len(data.time_values)):
                rollout = _rollout_loss(model, data, target_idx, train_config)
                total_loss = total_loss + rollout_scale * rollout["total"]
                rows.append(("rollout", 0, target_idx, rollout))
        total_loss = total_loss / max(len(rows), 1)
        total_loss.backward()
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if epoch % train_config.trace_interval == 0 or epoch == train_config.epochs - 1:
            append_jsonl(
                trace_path,
                {
                    "epoch": int(epoch),
                    "seconds": round(time.time() - started, 3),
                    "total_loss": float(total_loss.detach().cpu()),
                    "rollout_scale": float(rollout_scale),
                    "intervals": [
                        {
                            "direction": direction,
                            "source_time": float(data.time_values[source]),
                            "target_time": float(data.time_values[target]),
                            "matching": float(row["matching"].detach().cpu()),
                            "state_ot": float(row["state_ot"].detach().cpu()),
                            "spatial_ot": float(row["spatial_ot"].detach().cpu()),
                            "expr": float(row["expr"].detach().cpu()),
                            "mass_global": float(row["mass_global"].detach().cpu()),
                            "mass_local": float(row["mass_local"].detach().cpu()),
                            "spatial_deform": float(row["spatial_deform"].detach().cpu()),
                            "spatial_coverage": float(row["spatial_coverage"].detach().cpu()),
                            "action": float(row["action"].detach().cpu()),
                            "pred_ratio": float(row["pred_ratio"].detach().cpu()),
                            "expected_ratio": float(row["expected_ratio"].detach().cpu()),
                        }
                        for direction, source, target, row in rows
                    ],
                },
            )
        if device.type in {"mps", "cuda"} and (epoch + 1) % 10 == 0:
            clear_cache(device)

    checkpoint_path = output_dir / "model.pt"
    checkpoint = {
        "model_type": "spapot_potential",
        "model_config": model_config.to_json_dict(),
        "train_config": train_config.to_json_dict(),
        "state_dict": model.state_dict(),
        "time_values": data.time_values,
        "raw_time_values": data.raw_time_values,
        "scaler": data.scaler.to_json_dict(),
        "decoder_checkpoint": str(decoder_checkpoint) if decoder_checkpoint is not None else None,
        "seconds": round(time.time() - started, 2),
    }
    torch.save(checkpoint, checkpoint_path)
    return model, {
        "checkpoint": str(checkpoint_path),
        "training_trace": str(trace_path),
        "seconds": checkpoint["seconds"],
        "epochs": int(train_config.epochs),
    }
