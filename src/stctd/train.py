from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torchdiffeq import odeint

from .config import ModelConfig, TrainConfig
from .data import PreparedData, SampledSlice, sample_slice
from .fields import STCTDModel
from .integrator import STCTDODE, rollout_stctd
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
    model: STCTDModel,
    source: SampledSlice,
    target_time: float,
    train_config: TrainConfig,
    *,
    steps: int,
    neighbor_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logw0 = torch.zeros(source.state.shape[0], 1, dtype=source.state.dtype, device=source.state.device)
    return rollout_stctd(
        model,
        source.state,
        logw0,
        source.time_value,
        target_time,
        steps=max(int(steps), 1),
        method=train_config.integrator,
        alpha_exp=train_config.alpha_exp,
        alpha_gro=train_config.alpha_gro if train_config.use_growth else 0.0,
        use_growth=bool(train_config.use_growth),
        neighbor_index=neighbor_index,
    )


def _endpoint_term(
    model: STCTDModel,
    data: PreparedData,
    source: SampledSlice,
    source_index: int,
    target_index: int,
    train_config: TrainConfig,
    *,
    direction: str,
    neighbor_index: torch.Tensor | None,
    sample_size: int | None = None,
) -> dict[str, torch.Tensor | float | str]:
    target = sample_slice(data, target_index, int(sample_size or train_config.sample_size))
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


def _hj_config(train_config: TrainConfig) -> tuple[float, int, int]:
    lambda_hj = getattr(train_config, "lambda_hj", 0.0)
    alias_lambda_hj = getattr(train_config, "lambda_hjb", 0.0)
    use_alias = lambda_hj <= 0 and alias_lambda_hj > 0
    if use_alias:
        lambda_hj = alias_lambda_hj
        hj_start_epoch = getattr(train_config, "hjb_start_epoch", getattr(train_config, "hj_start_epoch", 0))
        hj_ramp_epochs = getattr(train_config, "hjb_ramp_epochs", getattr(train_config, "hj_ramp_epochs", 0))
    else:
        hj_start_epoch = getattr(train_config, "hj_start_epoch", getattr(train_config, "hjb_start_epoch", 0))
        hj_ramp_epochs = getattr(train_config, "hj_ramp_epochs", getattr(train_config, "hjb_ramp_epochs", 0))
    return float(lambda_hj), int(hj_start_epoch), int(hj_ramp_epochs)


def _hj_loss(model: STCTDModel, data: PreparedData, train_config: TrainConfig, sample_size: int | None = None) -> torch.Tensor:
    if str(getattr(model, "velocity_parameterization", "potential")).lower() != "potential":
        raise ValueError("HJ loss requires potential-driven molecular dynamics; disable lambda_hj for vector ablations.")
    rows = []
    for time_index, time_value in enumerate(data.time_values):
        sampled = sample_slice(data, time_index, int(sample_size or train_config.sample_size))
        _, grad_z, d_u_dt = model.potential_derivatives(time_value, sampled.state)
        # With dz/dt = -grad_z U, this is a first-order HJ residual rather than
        # a diffusion-control HJB residual.
        residual = d_u_dt.reshape(-1) - 0.5 * grad_z.pow(2).sum(dim=1)
        rows.append(residual.pow(2).mean())
    if not rows:
        return data.state_by_time[0].new_zeros(())
    return torch.stack(rows).mean()


def _make_optimizer(model: STCTDModel, train_config: TrainConfig) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(train_config, "optimizer", "adam")).lower()
    kwargs = {"lr": train_config.lr, "weight_decay": train_config.weight_decay}
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {train_config.optimizer}")


def _rollout_time_grid(
    model: STCTDModel,
    state0: torch.Tensor,
    time_values: list[float],
    train_config: TrainConfig,
    *,
    neighbor_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logw0 = torch.zeros(state0.shape[0], 1, dtype=state0.dtype, device=state0.device)
    action0 = torch.zeros_like(logw0)
    ssp0 = torch.zeros_like(logw0)
    method = str(train_config.integrator).lower()
    if method not in {"dopri5", "rk4", "euler", "midpoint"}:
        raise ValueError(f"Unsupported torchdiffeq method: {method}")
    direction_sign = 1.0 if float(time_values[-1]) >= float(time_values[0]) else -1.0
    ode_func = STCTDODE(
        model,
        alpha_exp=train_config.alpha_exp,
        alpha_gro=train_config.alpha_gro if train_config.use_growth else 0.0,
        direction_sign=direction_sign,
        use_growth=bool(train_config.use_growth),
        neighbor_index=neighbor_index,
    )
    options = None
    if method in {"rk4", "euler", "midpoint"}:
        options = {"step_size": float(train_config.ode_step_size)}
    t_eval = torch.as_tensor(time_values, dtype=state0.dtype, device=state0.device)
    return odeint(
        ode_func,
        (state0, logw0, action0, ssp0),
        t=t_eval,
        atol=1e-5,
        rtol=1e-5,
        method=method,
        options=options,
    )


def _stctd_weighted_match(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    pred_logw: torch.Tensor,
    expected_ratio: float,
    data: PreparedData,
    train_config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_weights = torch.exp(pred_logw).reshape(-1)
    if not train_config.use_growth:
        pred_weights = torch.ones_like(pred_weights)
    ot_loss = weighted_joint_emd_loss(
        pred_state,
        target_state,
        pred_weights,
        spatial_dim=data.spatial_dim,
        kappa_exp=train_config.kappa_exp,
    )
    growth_ratio = growth_ratio_penalty(pred_weights, expected_ratio) if train_config.use_growth else pred_state.new_zeros(())
    match_loss = ot_loss + float(train_config.kappa_gro) * growth_ratio
    return match_loss, ot_loss, growth_ratio, pred_weights.mean().detach()


def _stctd_fullgrid_loss(
    model: STCTDModel,
    data: PreparedData,
    train_config: TrainConfig,
    *,
    sample_size: int,
) -> dict[str, Any]:
    if train_config.use_cell_type_prior:
        raise ValueError("stctd_fullgrid currently supports the no-prior training path only.")
    if float(train_config.lambda_ssp) > 0:
        raise ValueError("stctd_fullgrid does not support SSP loss; use loss_mode='endpoint' for SSP ablations.")

    n_times = len(data.time_values)
    first_index = 0
    last_index = n_times - 1
    first_source = sample_slice(data, first_index, sample_size)
    last_source = sample_slice(data, last_index, sample_size)
    forward_state, forward_logw, forward_action, _ = _rollout_time_grid(
        model,
        first_source.state,
        data.time_values,
        train_config,
    )
    backward_times = list(reversed(data.time_values))
    backward_state, backward_logw, backward_action, _ = _rollout_time_grid(
        model,
        last_source.state,
        backward_times,
        train_config,
    )

    total_match = data.state_by_time[0].new_zeros(())
    ot_terms = []
    growth_terms = []
    term_rows: list[dict[str, Any]] = []
    first_count = int(data.state_by_time[first_index].shape[0])
    last_count = int(data.state_by_time[last_index].shape[0])

    for target_index, target_time in enumerate(data.time_values):
        target = sample_slice(data, target_index, sample_size)
        target_count = int(data.state_by_time[target_index].shape[0])
        if target_index > first_index:
            expected_ratio = target_count / max(first_count, 1)
            match_loss, ot_loss, growth_loss, pred_ratio = _stctd_weighted_match(
                forward_state[target_index],
                target.state,
                forward_logw[target_index],
                expected_ratio,
                data,
                train_config,
            )
            total_match = total_match + match_loss
            ot_terms.append(ot_loss)
            growth_terms.append(growth_loss)
            term_rows.append(
                {
                    "direction": "forward",
                    "source_time": float(data.time_values[first_index]),
                    "target_time": float(target_time),
                    "match_loss": match_loss,
                    "ot_loss": ot_loss,
                    "growth_ratio_loss": growth_loss,
                    "pred_ratio": pred_ratio,
                    "expected_ratio": expected_ratio,
                }
            )
        if target_index < last_index and train_config.use_bidirectional:
            back_index = last_index - target_index
            expected_ratio = target_count / max(last_count, 1)
            match_loss, ot_loss, growth_loss, pred_ratio = _stctd_weighted_match(
                backward_state[back_index],
                target.state,
                backward_logw[back_index],
                expected_ratio,
                data,
                train_config,
            )
            total_match = total_match + match_loss
            ot_terms.append(ot_loss)
            growth_terms.append(growth_loss)
            term_rows.append(
                {
                    "direction": "backward",
                    "source_time": float(data.time_values[last_index]),
                    "target_time": float(target_time),
                    "match_loss": match_loss,
                    "ot_loss": ot_loss,
                    "growth_ratio_loss": growth_loss,
                    "pred_ratio": pred_ratio,
                    "expected_ratio": expected_ratio,
                }
            )

    action_loss = forward_action[-1].mean()
    if train_config.use_bidirectional:
        action_loss = action_loss + backward_action[-1].mean()
    match_terms = torch.stack([row["match_loss"] for row in term_rows])
    ot_loss_mean = torch.stack(ot_terms).mean() if ot_terms else total_match.new_zeros(())
    growth_loss_mean = torch.stack(growth_terms).mean() if growth_terms else total_match.new_zeros(())
    return {
        "match_loss": total_match,
        "match_loss_mean": match_terms.mean() if len(term_rows) else total_match.new_zeros(()),
        "ot_loss": ot_loss_mean,
        "growth_ratio_loss": growth_loss_mean,
        "action_loss": action_loss,
        "ssp_loss": total_match.new_zeros(()),
        "terms": term_rows,
    }


def train_stctd_model(
    data: PreparedData,
    decoder_checkpoint: Path | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    output_dir: Path,
) -> tuple[STCTDModel, dict[str, Any]]:
    seed_all(train_config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    device = data.state_by_time[0].device
    model = STCTDModel(model_config).to(device)
    optimizer = _make_optimizer(model, train_config)
    started = time.time()
    n_times = len(data.time_values)
    if n_times <= 1:
        raise ValueError("Need at least two time points.")

    loss_mode = str(getattr(train_config, "loss_mode", "stctd_fullgrid")).lower()
    if loss_mode not in {"stctd_fullgrid", "endpoint"}:
        raise ValueError("loss_mode must be 'stctd_fullgrid' or 'endpoint'.")
    effective_sample_size = int(train_config.sample_size)
    first_index = 0
    last_index = n_times - 1
    for epoch in range(train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if loss_mode == "stctd_fullgrid":
            loss_payload = _stctd_fullgrid_loss(
                model,
                data,
                train_config,
                sample_size=effective_sample_size,
            )
            terms = loss_payload["terms"]
            match_loss = loss_payload["match_loss"]
            match_loss_mean = loss_payload["match_loss_mean"]
            action_loss = loss_payload["action_loss"]
            growth_ratio_loss = loss_payload["growth_ratio_loss"]
            ot_loss = loss_payload["ot_loss"]
            ssp_loss = loss_payload["ssp_loss"]
        else:
            terms: list[dict[str, torch.Tensor | float | str]] = []
            first_source = sample_slice(data, first_index, effective_sample_size)
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
                        sample_size=effective_sample_size,
                    )
                )

            if train_config.use_bidirectional:
                last_source = sample_slice(data, last_index, effective_sample_size)
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
                            sample_size=effective_sample_size,
                        )
                    )

            match_loss = torch.stack([term["match_loss"] for term in terms if torch.is_tensor(term["match_loss"])]).mean()
            match_loss_mean = match_loss
            action_loss = torch.stack([term["action_loss"] for term in terms if torch.is_tensor(term["action_loss"])]).mean()
            growth_ratio_loss = torch.stack([term["growth_ratio_loss"] for term in terms if torch.is_tensor(term["growth_ratio_loss"])]).mean()
            ot_loss = torch.stack([term["ot_loss"] for term in terms if torch.is_tensor(term["ot_loss"])]).mean()
            ssp_loss = torch.stack([term["ssp_loss"] for term in terms if torch.is_tensor(term["ssp_loss"])]).mean()

        lambda_hj, hj_start_epoch, hj_ramp_epochs = _hj_config(train_config)
        hj_scale = _ramped_weight(lambda_hj, epoch, hj_start_epoch, hj_ramp_epochs)
        hj_loss = _hj_loss(model, data, train_config, effective_sample_size) if hj_scale > 0 else data.state_by_time[0].new_zeros(())
        if loss_mode == "stctd_fullgrid":
            total_loss = float(train_config.lambda_match) * match_loss + action_loss + float(hj_scale) * hj_loss
        else:
            total_loss = (
                float(train_config.lambda_match) * match_loss
                + float(train_config.lambda_action) * action_loss
                + float(train_config.lambda_ssp) * ssp_loss
                + float(hj_scale) * hj_loss
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
                    "loss_mode": loss_mode,
                    "effective_sample_size": int(effective_sample_size),
                    "total_loss": float(total_loss.detach().cpu()),
                    "match_loss": float(match_loss.detach().cpu()),
                    "match_loss_mean": float(match_loss_mean.detach().cpu()),
                    "ot_loss": float(ot_loss.detach().cpu()),
                    "growth_ratio_loss": float(growth_ratio_loss.detach().cpu()),
                    "action_loss": float(action_loss.detach().cpu()),
                    "ssp_loss": float(ssp_loss.detach().cpu()),
                    "hj_loss": float(hj_loss.detach().cpu()),
                    "hj_scale": float(hj_scale),
                    "terms": [
                        {
                            "direction": str(term["direction"]),
                            "source_time": float(term["source_time"]),
                            "target_time": float(term["target_time"]),
                            "match_loss": float(term["match_loss"].detach().cpu()) if torch.is_tensor(term["match_loss"]) else float(term["match_loss"]),
                            "ot_loss": float(term["ot_loss"].detach().cpu()) if torch.is_tensor(term["ot_loss"]) else float(term["ot_loss"]),
                            "growth_ratio_loss": float(term["growth_ratio_loss"].detach().cpu()) if torch.is_tensor(term["growth_ratio_loss"]) else float(term["growth_ratio_loss"]),
                            "action_loss": float(term["action_loss"].detach().cpu()) if torch.is_tensor(term.get("action_loss")) else float(term.get("action_loss", 0.0)),
                            "ssp_loss": float(term["ssp_loss"].detach().cpu()) if torch.is_tensor(term.get("ssp_loss")) else float(term.get("ssp_loss", 0.0)),
                            "pred_ratio": float(term["pred_ratio"].detach().cpu()) if torch.is_tensor(term["pred_ratio"]) else float(term["pred_ratio"]),
                            "expected_ratio": float(term["expected_ratio"].detach().cpu()) if torch.is_tensor(term["expected_ratio"]) else float(term["expected_ratio"]),
                        }
                        for term in terms
                    ],
                },
            )
        if device.type in {"mps", "cuda"} and (epoch + 1) % 10 == 0:
            clear_cache(device)
        if (
            bool(getattr(train_config, "increase_sample_size", False))
            and (epoch + 1) < int(train_config.epochs)
            and epoch > 0
            and epoch % int(train_config.sample_growth_interval) == 0
        ):
            effective_sample_size += int(train_config.sample_growth_step)

    checkpoint_path = output_dir / "model.pt"
    checkpoint = {
        "model_type": "stctd",
        "dynamics_equation": "dsdt=v(s,z,t); dzdt=-grad_z Phi(s,z,t); dlogwdt=g(s,z,t)",
        "model_config": model_config.to_json_dict(),
        "train_config": train_config.to_json_dict(),
        "state_dict": model.state_dict(),
        "time_values": data.time_values,
        "raw_time_values": data.raw_time_values,
        "scaler": data.scaler.to_json_dict(),
        "final_effective_sample_size": int(effective_sample_size),
        "decoder_checkpoint": str(decoder_checkpoint) if decoder_checkpoint is not None else None,
        "seconds": round(time.time() - started, 2),
    }
    torch.save(checkpoint, checkpoint_path)
    return model, {
        "checkpoint": str(checkpoint_path),
        "training_trace": str(trace_path),
        "seconds": checkpoint["seconds"],
        "epochs": int(train_config.epochs),
        "loss_mode": loss_mode,
        "final_effective_sample_size": int(effective_sample_size),
    }
