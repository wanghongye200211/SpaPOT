from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PreparedData
from .utils import append_jsonl, seed_all


@dataclass(frozen=True)
class LatentClassifierConfig:
    hidden_dim: int = 128
    n_hidden: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 1000
    batch_size: int = 1024
    valid_fraction: float = 0.2
    patience: int = 80
    seed: int = 19491001
    trace_interval: int = 25
    class_balance: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatentMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: LatentClassifierConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = input_dim
        for _ in range(config.n_hidden):
            layers.append(nn.Linear(cur, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(config.dropout))
            cur = config.hidden_dim
        layers.append(nn.Linear(cur, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedLatentClassifier:
    model: LatentMLPClassifier
    categories: list[str]
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    config: LatentClassifierConfig
    checkpoint_path: Path
    trace_path: Path


def _build_latent_training_set(data: PreparedData, annotation_key: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    latent_blocks = []
    label_blocks = []
    categories = list(data.adata.obs[annotation_key].cat.categories)
    label_map = {category: idx for idx, category in enumerate(categories)}
    all_labels = data.adata.obs[annotation_key].astype(str).to_numpy()
    device = data.state_by_time[0].device
    latent_mean = torch.as_tensor(data.scaler.latent_mean, dtype=torch.float32, device=device).reshape(1, -1)
    latent_std = torch.as_tensor(data.scaler.latent_std, dtype=torch.float32, device=device).reshape(1, -1)
    for time_idx, state in enumerate(data.state_by_time):
        scaled_latent = state[:, data.spatial_dim :]
        latent_blocks.append(scaled_latent * latent_std + latent_mean)
        raw_idx = data.raw_indices_by_time[time_idx]
        label_blocks.append(torch.as_tensor([label_map[value] for value in all_labels[raw_idx]], dtype=torch.long, device=device))
    return torch.cat(latent_blocks, dim=0), torch.cat(label_blocks, dim=0), categories


def train_latent_classifier(
    data: PreparedData,
    annotation_key: str,
    *,
    output_dir: Path,
    config: LatentClassifierConfig | None = None,
) -> TrainedLatentClassifier:
    config = config or LatentClassifierConfig()
    seed_all(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "latent_classifier_trace.jsonl"
    checkpoint_path = output_dir / "latent_classifier.pt"
    if trace_path.exists():
        trace_path.unlink()

    x_raw, y, categories = _build_latent_training_set(data, annotation_key)
    latent_mean = x_raw.mean(dim=0, keepdim=True)
    latent_std = x_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x_raw - latent_mean) / latent_std

    generator = torch.Generator(device=x.device)
    generator.manual_seed(config.seed)
    perm = torch.randperm(x.shape[0], device=x.device, generator=generator)
    valid_n = max(1, int(round(x.shape[0] * config.valid_fraction)))
    valid_idx = perm[:valid_n]
    train_idx = perm[valid_n:]

    model = LatentMLPClassifier(x.shape[1], len(categories), config).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    class_weight = None
    if config.class_balance:
        counts = torch.bincount(y, minlength=len(categories)).float()
        class_weight = (counts.sum() / (counts.clamp_min(1.0) * len(categories))).to(x.device)

    best_state = None
    best_valid = math.inf
    bad_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        train_perm = train_idx[torch.randperm(train_idx.shape[0], device=x.device, generator=generator)]
        train_loss_total = 0.0
        train_correct = 0
        train_count = 0
        for start in range(0, train_perm.shape[0], config.batch_size):
            batch_idx = train_perm[start : start + config.batch_size]
            logits = model(x[batch_idx])
            loss = F.cross_entropy(logits, y[batch_idx], weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n_batch = int(batch_idx.shape[0])
            train_loss_total += float(loss.detach().cpu()) * n_batch
            train_correct += int((logits.argmax(dim=1) == y[batch_idx]).sum().detach().cpu())
            train_count += n_batch

        model.eval()
        valid_loss_total = 0.0
        valid_correct = 0
        valid_count = 0
        with torch.no_grad():
            for start in range(0, valid_idx.shape[0], config.batch_size):
                batch_idx = valid_idx[start : start + config.batch_size]
                logits = model(x[batch_idx])
                loss = F.cross_entropy(logits, y[batch_idx], weight=class_weight)
                n_batch = int(batch_idx.shape[0])
                valid_loss_total += float(loss.detach().cpu()) * n_batch
                valid_correct += int((logits.argmax(dim=1) == y[batch_idx]).sum().detach().cpu())
                valid_count += n_batch
        valid_loss = valid_loss_total / max(valid_count, 1)
        valid_acc = valid_correct / max(valid_count, 1)
        if epoch % config.trace_interval == 0 or epoch == config.epochs - 1:
            append_jsonl(
                trace_path,
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss_total / max(train_count, 1),
                    "train_acc": train_correct / max(train_count, 1),
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                },
            )
        if valid_loss < best_valid - 1e-5:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_type": "stctd_latent_classifier",
            "config": config.to_json_dict(),
            "categories": categories,
            "input_dim": int(x.shape[1]),
            "latent_mean": latent_mean.detach().cpu(),
            "latent_std": latent_std.detach().cpu(),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return TrainedLatentClassifier(
        model=model,
        categories=categories,
        latent_mean=latent_mean,
        latent_std=latent_std,
        config=config,
        checkpoint_path=checkpoint_path,
        trace_path=trace_path,
    )


@torch.no_grad()
def predict_latent_labels(
    classifier: TrainedLatentClassifier,
    latent: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    classifier.model.eval()
    labels = []
    mean = classifier.latent_mean.to(device)
    std = classifier.latent_std.to(device)
    for start in range(0, latent.shape[0], batch_size):
        batch = torch.as_tensor(latent[start : start + batch_size], dtype=torch.float32, device=device)
        logits = classifier.model((batch - mean) / std)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        labels.extend(classifier.categories[int(idx)] for idx in pred)
    return np.asarray(labels, dtype=object)
