from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


class MLPNetWork(nn.Module):
    """stCTD classifier over [spatial, latent, time]."""

    def __init__(self, spatial_dim: int = 2, input_gene_dim: int = 10, output_dim: int = 24) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=spatial_dim + input_gene_dim + 1, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
        )
        self.out = nn.Linear(in_features=128, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(x))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, spatial_data: torch.Tensor, exp_data: torch.Tensor, time: torch.Tensor, label: torch.Tensor) -> None:
        self.x = torch.cat((spatial_data, exp_data, time.unsqueeze(1)), dim=1)
        self.label = label

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index, :], self.label[index]


def seed_all(seed: int = 19491001) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set: Dataset, valid_ratio: float, seed: int) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    valid_set_size = int(valid_ratio * len(data_set))
    if len(data_set) > 1:
        valid_set_size = max(1, valid_set_size)
    valid_set_size = min(valid_set_size, max(0, len(data_set) - 1))
    train_set_size = len(data_set) - valid_set_size
    return random_split(
        data_set,
        [train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(seed),
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def train_st_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    best_loss = math.inf
    early_stop_count = 0
    trace_path = config.get("trace_path")
    trace_interval = int(config.get("trace_interval", 50))

    for cur_epoch in range(int(config["n_epochs"])):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True
            outputs = model(inputs)
            loss_entropy = criterion(outputs, labels)
            grad_outputs = torch.ones_like(outputs)
            loss_time_l1norm = torch.mean(
                torch.abs(torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True)[0][:, -1])
            )
            loss = loss_entropy + float(config["weight_time_l1_norm"]) * loss_time_l1norm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_record: list[float] = []
        eval_loader = valid_loader if len(valid_loader.dataset) else train_loader
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True
            outputs = model(inputs)
            loss_entropy = criterion(outputs, labels)
            grad_outputs = torch.ones_like(outputs)
            loss_time_l1norm = torch.mean(
                torch.abs(torch.autograd.grad(outputs, inputs, grad_outputs, create_graph=True)[0][:, -1])
            )
            loss = loss_entropy + float(config["weight_time_l1_norm"]) * loss_time_l1norm
            loss_record.append(float(loss.detach().cpu()))
        mean_valid_loss = _mean(loss_record)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            early_stop_count = 0
            Path(config["save_path"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model, config["save_path"])
        else:
            early_stop_count += 1

        if trace_path and (cur_epoch % trace_interval == 0 or cur_epoch == int(config["n_epochs"]) - 1):
            target = Path(trace_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": int(cur_epoch),
                            "valid_loss": float(mean_valid_loss),
                            "best_loss": float(best_loss),
                            "early_stop_count": int(early_stop_count),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if early_stop_count >= int(config["early_stop"]):
            break


def create_spatiotemporal_classifier(
    adata: Any,
    st_classifier_save_path: str | Path,
    annotation_key: str = "Annotation",
    device: torch.device | None = None,
    *,
    spatial_key: str = "X_spatial_input",
    latent_key: str = "X_gene_input",
    time_key: str = "time_input",
    trace_path: str | Path | None = None,
    n_epochs: int = 1000,
    batch_size: int = 1000,
    early_stop: int = 50,
    seed: int = 19491001,
) -> dict[int, str]:
    """Train the stCTD spatiotemporal cell-type readout."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_types = adata.obs[annotation_key].astype("category")
    unique_types = np.array(list(cell_types.cat.categories))
    cell_type_to_label_map = {cell_type: i for i, cell_type in enumerate(unique_types)}
    label_to_cell_type_map = {i: str(cell_type) for i, cell_type in enumerate(unique_types)}
    label = torch.tensor([cell_type_to_label_map[cell_type] for cell_type in cell_types], dtype=torch.long)

    config = {
        "seed": int(seed),
        "learning_rate": 1e-3,
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size),
        "early_stop": int(early_stop),
        "valid_ratio": 0.1,
        "weight_time_l1_norm": 10.0,
        "save_path": str(st_classifier_save_path),
        "trace_path": str(trace_path) if trace_path is not None else None,
    }
    seed_all(config["seed"])

    exp_data = torch.tensor(np.asarray(adata.obsm[latent_key], dtype=np.float32), dtype=torch.float32)
    spatial_data = torch.tensor(np.asarray(adata.obsm[spatial_key], dtype=np.float32), dtype=torch.float32)
    time = torch.tensor(np.asarray(adata.obs[time_key], dtype=np.float32), dtype=torch.float32)
    data_set = Dataset(spatial_data=spatial_data, exp_data=exp_data, time=time, label=label)
    train_data, valid_data = train_valid_split(data_set, float(config["valid_ratio"]), int(config["seed"]))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=int(config["batch_size"]))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=int(config["batch_size"]))

    model = MLPNetWork(
        spatial_dim=int(spatial_data.shape[1]),
        input_gene_dim=int(exp_data.shape[1]),
        output_dim=int(unique_types.shape[0]),
    ).to(device)
    train_st_classifier(model, train_loader, valid_loader, config, device)
    return label_to_cell_type_map
