from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TrainConfig
from .data import PreparedData, SampledSlice, sample_slice
from .fields import SpaPOTPotentialModel
from .integrator import rollout_hybrid_potential
from .losses import grouped_joint_emd_loss, growth_ratio_penalty, weighted_joint_emd_loss
from .utils import append_jsonl, clear_cache, seed_all


def _neighbor_index(source: SampledSlice, spatial_dim: int, n_neighbors: int) -> torch.Tensor | None:
    if n_neighbors <= 0 or source.state.shape[0] <= 2:
        return None
    k = min(int(n_neighbors), int(source.state.shape[0]) - 1)
    dist = torch.cdist(source.state[:, :spatial_dim], source.state[:, :spatial_dim])
    _, idx = torch.topk(dist, k=k + 1, largest=False)
    return idx[:, 1:].contiguous()


def _rollout(
    model: SpaPOTPotentialModel,
    source: SampledSlice,
    target_time: float,
    train_config: TrainConfig,
    *,
    steps: int,
    neighbor_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logw0 = torch.zeros(source.state.shape[0], 1, dtype=source.state.dtype, device=source.state.device)
    return rollout_hybrid_potential(
        model,
        source.state,
        logw0,
        source.time_value,
        target_time,
        steps=max(int(steps), 1),
        method=train_config.integrator,
        alpha_exp=train_config.alpha_exp,
        alpha_gro=train_config.alpha_gro if train_config.use_growth else 0.0,
        neighbor_index=neighbor_index,
    )


def _endpoint_term(
    model: SpaPOTPotentialModel,
    data: PreparedData,
    source: SampledSlice,
    source_index: int,
    target_index: int,
    train_config: TrainConfig,
    *,
    direction: str,
    neighbor_index: torch.Tensor | None,
) -> dict[str, torch.Tensor | float | str]:
    target = sample_slice(data, target_index, train_config.sample_size)
    n_steps = max(1, train_config.steps_per_interval * abs(int(target_index) - int(source_index)))
    pred_state, pred_logw, action, ssp = _rollout(
        model,
        source,
        target.time_value,
        train_config,
        steps=n_steps,
        neighbor_index=neighbor_index,
    )
    pred_weights = torch.exp(pred_logw).reshape(-1)
    if not train_config.use_growth:
        pred_weights = torch.ones_like(pred_weights)

    if train_config.use_cell_type_prior:
        ot_loss = grouped_joint_emd_loss(
            pred_state,
            target.state,
            pred_weights,
            source.labels,
            target.labels,
            spatial_dim=data.spatial_dim,
            kappa_exp=train_config.kappa_exp,
            min_count=train_config.cell_type_prior_min_count,
        )
    else:
        ot_loss = weighted_joint_emd_loss(
            pred_state,
            target.state,
            pred_weights,
            spatial_dim=data.spatial_dim,
            kappa_exp=train_config.kappa_exp,
        )

    expected_ratio = data.state_by_time[target_index].shape[0] / data.state_by_time[source_index].shape[0]
    growth_ratio = growth_ratio_penalty(pred_weights, expected_ratio) if train_config.use_growth else pred_state.new_zeros(())
    match_loss = ot_loss + float(train_config.kappa_gro) * growth_ratio
    return {
        "direction": direction,
        "source_time": float(data.time_values[source_index]),
        "target_time": float(data.time_values[target_index]),
        "match_loss": match_loss,
        "ot_loss": ot_loss,
        "growth_ratio_loss": growth_ratio,
        "action_loss": action.mean(),
        "ssp_loss": ssp.mean(),
        "pred_ratio": pred_weights.mean().detach(),
        "expected_ratio": torch.as_tensor(float(expected_ratio), dtype=pred_weights.dtype, device=pred_weights.device),
    }


def _ramped_weight(base_weight: float, epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if base_weight <= 0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    progress = (epoch - start_epoch + 1) / float(ramp_epochs)
    return float(base_weight) * float(max(0.0, min(1.0, progress)))


def _hjb_loss(model: SpaPOTPotentialModel, data: PreparedData, train_config: TrainConfig) -> torch.Tensor:
    rows = []
    for time_index, time_value in enumerate(data.time_values):
        sampled = sample_slice(data, time_index, train_config.sample_size)
        _, grad_z, d_u_dt = model.potential_derivatives(time_value, sampled.state)
        residual = d_u_dt.reshape(-1) - 0.5 * grad_z.pow(2).sum(dim=1)
        rows.append(residual.pow(2).mean())
    if not rows:
        return data.state_by_time[0].new_zeros(())
    return torch.stack(rows).mean()


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    started = time.time()
    n_times = len(data.time_values)
    if n_times <= 1:
        raise ValueError("Need at least two time points.")

    first_index = 0
    last_index = n_times - 1
    for epoch in range(train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        terms: list[dict[str, torch.Tensor | float | str]] = []
        first_source = sample_slice(data, first_index, train_config.sample_size)
        first_neighbors = _neighbor_index(first_source, data.spatial_dim, train_config.ssp_neighbors) if train_config.lambda_ssp > 0 else None
        for target_index in range(1, n_times):
            terms.append(
                _endpoint_term(
                    model,
                    data,
                    first_source,
                    first_index,
                    target_index,
                    train_config,
                    direction="forward",
                    neighbor_index=first_neighbors,
                )
            )

        if train_config.use_bidirectional:
            last_source = sample_slice(data, last_index, train_config.sample_size)
            last_neighbors = _neighbor_index(last_source, data.spatial_dim, train_config.ssp_neighbors) if train_config.lambda_ssp > 0 else None
            for target_index in range(0, last_index):
                terms.append(
                    _endpoint_term(
                        model,
                        data,
                        last_source,
                        last_index,
                        target_index,
                        train_config,
                        direction="backward",
                        neighbor_index=last_neighbors,
                    )
                )

        match_loss = torch.stack([term["match_loss"] for term in terms if torch.is_tensor(term["match_loss"])]).mean()
        action_loss = torch.stack([term["action_loss"] for term in terms if torch.is_tensor(term["action_loss"])]).mean()
        growth_ratio_loss = torch.stack([term["growth_ratio_loss"] for term in terms if torch.is_tensor(term["growth_ratio_loss"])]).mean()
        ot_loss = torch.stack([term["ot_loss"] for term in terms if torch.is_tensor(term["ot_loss"])]).mean()
        ssp_loss = torch.stack([term["ssp_loss"] for term in terms if torch.is_tensor(term["ssp_loss"])]).mean()
        hjb_scale = _ramped_weight(train_config.lambda_hjb, epoch, train_config.hjb_start_epoch, train_config.hjb_ramp_epochs)
        hjb_loss = _hjb_loss(model, data, train_config) if hjb_scale > 0 else data.state_by_time[0].new_zeros(())
        total_loss = (
            float(train_config.lambda_match) * match_loss
            + float(train_config.lambda_action) * action_loss
            + float(train_config.lambda_ssp) * ssp_loss
            + float(hjb_scale) * hjb_loss
        )
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
                    "match_loss": float(match_loss.detach().cpu()),
                    "ot_loss": float(ot_loss.detach().cpu()),
                    "growth_ratio_loss": float(growth_ratio_loss.detach().cpu()),
                    "action_loss": float(action_loss.detach().cpu()),
                    "ssp_loss": float(ssp_loss.detach().cpu()),
                    "hjb_loss": float(hjb_loss.detach().cpu()),
                    "hjb_scale": float(hjb_scale),
                    "terms": [
                        {
                            "direction": str(term["direction"]),
                            "source_time": float(term["source_time"]),
                            "target_time": float(term["target_time"]),
                            "match_loss": float(term["match_loss"].detach().cpu()) if torch.is_tensor(term["match_loss"]) else float(term["match_loss"]),
                            "ot_loss": float(term["ot_loss"].detach().cpu()) if torch.is_tensor(term["ot_loss"]) else float(term["ot_loss"]),
                            "growth_ratio_loss": float(term["growth_ratio_loss"].detach().cpu()) if torch.is_tensor(term["growth_ratio_loss"]) else float(term["growth_ratio_loss"]),
                            "action_loss": float(term["action_loss"].detach().cpu()) if torch.is_tensor(term["action_loss"]) else float(term["action_loss"]),
                            "ssp_loss": float(term["ssp_loss"].detach().cpu()) if torch.is_tensor(term["ssp_loss"]) else float(term["ssp_loss"]),
                            "pred_ratio": float(term["pred_ratio"].detach().cpu()) if torch.is_tensor(term["pred_ratio"]) else float(term["pred_ratio"]),
                            "expected_ratio": float(term["expected_ratio"].detach().cpu()) if torch.is_tensor(term["expected_ratio"]) else float(term["expected_ratio"]),
                        }
                        for term in terms
                    ],
                },
            )
        if device.type in {"mps", "cuda"} and (epoch + 1) % 10 == 0:
            clear_cache(device)

    checkpoint_path = output_dir / "model.pt"
    checkpoint = {
        "model_type": "spapot_hybrid_potential",
        "dynamics_equation": "dsdt=v(s,z,t); dzdt=-grad_z Phi(s,z,t); dlogwdt=g(s,z,t)",
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
