from __future__ import annotations

import numpy as np
import ot
import torch


def squared_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, None, :] - b[None, :, :]).pow(2).sum(dim=2)


def joint_cost_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    spatial_dim: int,
    kappa_exp: float,
) -> torch.Tensor:
    kappa = float(kappa_exp)
    spatial_cost = squared_cdist(pred[:, :spatial_dim], target[:, :spatial_dim])
    gene_cost = squared_cdist(pred[:, spatial_dim:], target[:, spatial_dim:])
    return (1.0 - kappa) * spatial_cost + kappa * gene_cost


def _normalized_weights(weights: torch.Tensor, n_obs: int, like: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 0:
        out = torch.ones(n_obs, dtype=like.dtype, device=like.device)
    else:
        out = weights.reshape(-1).to(dtype=like.dtype, device=like.device).clamp_min(1e-8)
    return out / out.sum().clamp_min(1e-8)


def weighted_joint_emd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    *,
    spatial_dim: int,
    kappa_exp: float,
) -> torch.Tensor:
    cost = joint_cost_matrix(pred, target, spatial_dim=spatial_dim, kappa_exp=kappa_exp)
    a = _normalized_weights(pred_weights, pred.shape[0], pred)
    b = torch.full((target.shape[0],), 1.0 / max(int(target.shape[0]), 1), dtype=pred.dtype, device=pred.device)
    plan_np = ot.emd(
        a.detach().cpu().numpy().astype(np.float64),
        b.detach().cpu().numpy().astype(np.float64),
        cost.detach().cpu().numpy().astype(np.float64),
    )
    plan = torch.as_tensor(plan_np, dtype=cost.dtype, device=cost.device)
    return torch.sum(plan * cost)


def grouped_joint_emd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    spatial_dim: int,
    kappa_exp: float,
    min_count: int,
) -> torch.Tensor:
    labels = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    total = pred.new_zeros(())
    total_weight = pred.new_zeros(())
    n_target = max(int(target_labels.shape[0]), 1)
    for label in labels:
        source_mask_np = source_labels == label
        target_mask_np = target_labels == label
        if int(source_mask_np.sum()) < int(min_count) or int(target_mask_np.sum()) < int(min_count):
            continue
        source_mask = torch.as_tensor(source_mask_np, dtype=torch.bool, device=pred.device)
        target_mask = torch.as_tensor(target_mask_np, dtype=torch.bool, device=target.device)
        group_loss = weighted_joint_emd_loss(
            pred[source_mask],
            target[target_mask],
            pred_weights[source_mask],
            spatial_dim=spatial_dim,
            kappa_exp=kappa_exp,
        )
        group_weight = torch.as_tensor(float(target_mask_np.sum()) / float(n_target), dtype=pred.dtype, device=pred.device)
        total = total + group_weight * group_loss
        total_weight = total_weight + group_weight
    if float(total_weight.detach().cpu()) <= 0:
        return weighted_joint_emd_loss(
            pred,
            target,
            pred_weights,
            spatial_dim=spatial_dim,
            kappa_exp=kappa_exp,
        )
    return total / total_weight.clamp_min(1e-8)


def growth_ratio_penalty(pred_weights: torch.Tensor, expected_ratio: float) -> torch.Tensor:
    target = torch.as_tensor(float(expected_ratio), dtype=pred_weights.dtype, device=pred_weights.device).clamp_min(1e-8)
    pred_ratio = pred_weights.reshape(-1).mean()
    return torch.abs(pred_ratio - target) / target
