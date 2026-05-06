from __future__ import annotations

import numpy as np
import ot
import torch


def squared_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, None, :] - b[None, :, :]).pow(2).sum(dim=2)


def weighted_emd_plan(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    target_weights: torch.Tensor | None = None,
    *,
    spatial_dim: int,
    spatial_cost_weight: float,
    gene_cost_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial_cost = squared_cdist(pred[:, :spatial_dim], target[:, :spatial_dim])
    gene_cost = squared_cdist(pred[:, spatial_dim:], target[:, spatial_dim:])
    cost = spatial_cost_weight * spatial_cost + gene_cost_weight * gene_cost
    a = pred_weights.reshape(-1).clamp_min(1e-8)
    if target_weights is None:
        b = torch.ones(target.shape[0], dtype=pred.dtype, device=pred.device)
    else:
        b = target_weights.reshape(-1).clamp_min(1e-8)
    a = a / a.sum()
    b = b / b.sum()
    plan_np = ot.emd(
        a.detach().cpu().numpy().astype(np.float64),
        b.detach().cpu().numpy().astype(np.float64),
        cost.detach().cpu().numpy().astype(np.float64),
    )
    plan = torch.as_tensor(plan_np, dtype=cost.dtype, device=cost.device)
    loss = torch.sum(plan * cost)
    return loss, plan, cost


def weighted_spatial_emd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    target_weights: torch.Tensor | None,
    *,
    spatial_dim: int,
) -> torch.Tensor:
    cost = squared_cdist(pred[:, :spatial_dim], target[:, :spatial_dim])
    a = pred_weights.reshape(-1).clamp_min(1e-8)
    b = torch.ones(target.shape[0], dtype=pred.dtype, device=pred.device) if target_weights is None else target_weights.reshape(-1).clamp_min(1e-8)
    a = a / a.sum()
    b = b / b.sum()
    plan_np = ot.emd(
        a.detach().cpu().numpy().astype(np.float64),
        b.detach().cpu().numpy().astype(np.float64),
        cost.detach().cpu().numpy().astype(np.float64),
    )
    plan = torch.as_tensor(plan_np, dtype=cost.dtype, device=cost.device)
    return torch.sum(plan * cost)


def spatial_undercoverage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    *,
    spatial_dim: int,
    bandwidth: float,
    anchor_count: int,
) -> torch.Tensor:
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros(())
    if anchor_count > 0 and target.shape[0] > anchor_count:
        idx = torch.randperm(target.shape[0], device=target.device)[:anchor_count]
        anchors = target[idx, :spatial_dim]
    else:
        anchors = target[:, :spatial_dim]
    h2 = torch.as_tensor(float(bandwidth) ** 2, dtype=pred.dtype, device=pred.device).clamp_min(1e-8)
    pred_spatial = pred[:, :spatial_dim]
    target_spatial = target[:, :spatial_dim]
    k_pred = torch.exp(-squared_cdist(anchors, pred_spatial) / (2.0 * h2))
    k_target = torch.exp(-squared_cdist(anchors, target_spatial) / (2.0 * h2))
    pred_mass = pred_weights.reshape(-1).clamp_min(1e-8)
    pred_mass = pred_mass / pred_mass.sum().clamp_min(1e-8)
    target_mass = torch.full((target.shape[0],), 1.0 / max(int(target.shape[0]), 1), dtype=target.dtype, device=target.device)
    pred_density = k_pred @ pred_mass
    target_density = k_target @ target_mass
    deficit = torch.relu(torch.log(target_density.detach() + 1e-6) - torch.log(pred_density + 1e-6))
    return deficit.pow(2).mean()


def global_mass_ratio_loss(pred_weights: torch.Tensor, expected_ratio: float) -> torch.Tensor:
    pred_ratio = pred_weights.reshape(-1).mean()
    target = torch.as_tensor(float(expected_ratio), dtype=pred_weights.dtype, device=pred_weights.device)
    return ((pred_ratio - target) / target.clamp_min(1e-6)).pow(2)


def local_soft_mass_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_weights: torch.Tensor,
    expected_ratio: float,
    *,
    bandwidth: float,
    anchor_count: int,
) -> torch.Tensor:
    if anchor_count > 0 and target.shape[0] > anchor_count:
        idx = torch.randperm(target.shape[0], device=target.device)[:anchor_count]
        anchors = target[idx]
    else:
        anchors = target
    h2 = torch.as_tensor(float(bandwidth) ** 2, dtype=pred.dtype, device=pred.device).clamp_min(1e-8)
    k_pred = torch.exp(-squared_cdist(anchors, pred) / (2.0 * h2))
    k_target = torch.exp(-squared_cdist(anchors, target) / (2.0 * h2))
    rho_pred = k_pred @ pred_weights.reshape(-1)
    target_weights = torch.full((target.shape[0],), float(expected_ratio), dtype=pred.dtype, device=pred.device)
    rho_target = k_target @ target_weights
    rho_pred = rho_pred / max(pred.shape[0], 1)
    rho_target = rho_target / max(target.shape[0], 1)
    return torch.mean((torch.log1p(rho_pred) - torch.log1p(rho_target.detach())) ** 2)


def pairwise_expression_mse(decoded_pred: torch.Tensor, observed_expression: torch.Tensor) -> torch.Tensor:
    n_genes = decoded_pred.shape[1]
    pred_sq = decoded_pred.pow(2).sum(dim=1, keepdim=True)
    obs_sq = observed_expression.pow(2).sum(dim=1).unsqueeze(0)
    cross = decoded_pred @ observed_expression.T
    return torch.clamp((pred_sq + obs_sq - 2.0 * cross) / float(n_genes), min=0.0)


def weighted_expression_reconstruction_loss(
    decoded_pred: torch.Tensor,
    target_expression: torch.Tensor,
    plan: torch.Tensor,
    *,
    detach_plan: bool,
) -> torch.Tensor:
    if detach_plan:
        plan = plan.detach()
    cost = pairwise_expression_mse(decoded_pred, target_expression)
    return torch.sum(plan * cost) / plan.sum().clamp_min(1e-12)
