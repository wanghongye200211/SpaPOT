from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GenePriorGraph:
    gene_names: list[str]
    edge_index: torch.Tensor
    retained_prior_edges: int
    retained_prior_genes: int
    connected_components: int
    largest_component_size: int
    network_path: str
    used_self_loops: bool
    used_symmetric_edges: bool


@dataclass(frozen=True)
class GenePriorGATConfig:
    latent_dim: int = 10
    h0: int = 64
    h1: int = 128
    h2: int = 128
    h3: int = 64
    heads: int = 4
    dropout: float = 0.1
    mask_prob: float = 0.1
    batch_size: int = 96
    max_epochs: int = 160
    patience: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.12
    latent_var_weight: float = 1e-3
    max_prior_neighbors_per_gene: int = 8
    max_train_cells: int | None = 1500
    store_reconstruction: bool = False
    seed: int = 19491001


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _dense_rows_float32(matrix: Any, indices: np.ndarray | slice) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix[indices].toarray().astype(np.float32)
    return np.asarray(matrix[indices], dtype=np.float32)


def _sample_training_matrix(matrix: Any, config: GenePriorGATConfig) -> np.ndarray:
    n_obs = int(matrix.shape[0])
    indices = np.arange(n_obs)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)
    if config.max_train_cells is not None and config.max_train_cells > 0 and config.max_train_cells < len(indices):
        indices = indices[: config.max_train_cells]
    indices = np.sort(indices)
    return _dense_rows_float32(matrix, indices)


def _connected_component_stats(num_nodes: int, undirected_edges: set[tuple[int, int]]) -> tuple[int, int]:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    touched: set[int] = set()
    for a, b in undirected_edges:
        if a == b:
            continue
        touched.add(a)
        touched.add(b)
        union(a, b)
    if not touched:
        return 0, 0
    counts: dict[int, int] = {}
    for node in touched:
        root = find(node)
        counts[root] = counts.get(root, 0) + 1
    return len(counts), max(counts.values())


def build_gene_prior_graph(
    gene_names: list[str],
    network_path: str | Path,
    *,
    add_symmetric_edges: bool = True,
    add_self_loops: bool = True,
    max_prior_neighbors_per_gene: int | None = None,
) -> GenePriorGraph:
    network_path = Path(network_path)
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_names)}
    directed_edges: set[tuple[int, int]] = set()
    prior_undirected: set[tuple[int, int]] = set()
    retained_prior_edges = 0

    with network_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src_gene, dst_gene = parts[0], parts[1]
            src = gene_to_idx.get(src_gene)
            dst = gene_to_idx.get(dst_gene)
            if src is None or dst is None:
                continue
            retained_prior_edges += 1
            directed_edges.add((src, dst))
            if add_symmetric_edges:
                directed_edges.add((dst, src))
            prior_undirected.add((min(src, dst), max(src, dst)))

    if max_prior_neighbors_per_gene is not None and max_prior_neighbors_per_gene > 0:
        by_dst: dict[int, list[int]] = {}
        for src, dst in directed_edges:
            if src == dst:
                continue
            by_dst.setdefault(dst, []).append(src)
        capped_edges: set[tuple[int, int]] = set()
        for dst, src_values in by_dst.items():
            # Deterministic cap: keep a reproducible spread of prior neighbors
            # instead of favoring file order in very high-degree genes.
            unique_src = sorted(set(src_values))
            if len(unique_src) > max_prior_neighbors_per_gene:
                positions = np.linspace(0, len(unique_src) - 1, max_prior_neighbors_per_gene)
                unique_src = [unique_src[int(round(pos))] for pos in positions]
            capped_edges.update((src, dst) for src in unique_src)
        directed_edges = capped_edges

    if add_self_loops:
        for idx in range(len(gene_names)):
            directed_edges.add((idx, idx))

    if not directed_edges:
        raise ValueError(f"No prior edges from {network_path} overlap the provided gene list.")

    ordered_edges = sorted(directed_edges)
    edge_index = torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()
    retained_prior_genes = len({idx for edge in prior_undirected for idx in edge})
    n_components, largest = _connected_component_stats(len(gene_names), prior_undirected)
    return GenePriorGraph(
        gene_names=list(map(str, gene_names)),
        edge_index=edge_index,
        retained_prior_edges=retained_prior_edges,
        retained_prior_genes=retained_prior_genes,
        connected_components=n_components,
        largest_component_size=largest,
        network_path=str(network_path),
        used_self_loops=add_self_loops,
        used_symmetric_edges=add_symmetric_edges,
    )


class StaticSparseGeneGATLayer(nn.Module):
    """Sparse multi-head GAT layer over a fixed gene graph.

    The attention coefficients are learned per prior edge from gene identity
    embeddings. Messages remain expression-dependent, so every cell still
    propagates its own gene-node states through the same prior graph. This keeps
    the layer usable for 2k genes without PyG while preserving an explicit
    attention mechanism in both encoder and decoder.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        edge_index: torch.Tensor,
    ) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}.")
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.head_dim = out_dim // heads
        self.dropout = float(dropout)
        self.register_buffer("edge_index", edge_index.long().contiguous())

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_gene = nn.Parameter(torch.empty(num_nodes, heads, self.head_dim))
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)
        nn.init.xavier_uniform_(self.attn_gene)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def _attention_values(self) -> torch.Tensor:
        src, dst = self.edge_index[0], self.edge_index[1]
        src_score = (self.attn_gene[src] * self.attn_src.unsqueeze(0)).sum(dim=-1)
        dst_score = (self.attn_gene[dst] * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        logits = F.leaky_relu(src_score + dst_score, negative_slope=0.2).t().contiguous()
        # Softmax over incoming edges for each destination gene, independently per head.
        alpha = torch.zeros_like(logits)
        for head in range(self.heads):
            per_head = logits[head]
            max_per_dst = torch.full(
                (self.num_nodes,),
                -torch.inf,
                dtype=per_head.dtype,
                device=per_head.device,
            )
            max_per_dst.scatter_reduce_(0, dst, per_head, reduce="amax", include_self=True)
            exp = torch.exp(per_head - max_per_dst[dst])
            denom = torch.zeros(self.num_nodes, dtype=per_head.dtype, device=per_head.device)
            denom.scatter_add_(0, dst, exp)
            alpha[head] = exp / denom[dst].clamp_min(1e-12)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        src, dst = self.edge_index[0], self.edge_index[1]
        h = self.proj(x).view(batch_size, self.num_nodes, self.heads, self.head_dim)
        alpha = self._attention_values()
        out_heads = torch.zeros(
            batch_size,
            self.num_nodes,
            self.heads,
            self.head_dim,
            dtype=h.dtype,
            device=h.device,
        )
        messages = h[:, src, :, :] * alpha.t().unsqueeze(0).unsqueeze(-1)
        out_heads.index_add_(1, dst, messages)
        out = out_heads.reshape(batch_size, self.num_nodes, self.out_dim)
        out = self.norm(out + self.residual(x))
        return F.elu(F.dropout(out, p=self.dropout, training=self.training))


class GenePriorGATEncoder(nn.Module):
    def __init__(self, num_genes: int, edge_index: torch.Tensor, config: GenePriorGATConfig) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        widths = [config.h0, config.h1, config.h2, config.h3]
        self.expr_projection = nn.Linear(1, config.h0)
        self.gene_embedding = nn.Embedding(num_genes, config.h0)
        self.layers = nn.ModuleList(
            [
                StaticSparseGeneGATLayer(num_genes, widths[0], widths[1], config.heads, config.dropout, edge_index),
                StaticSparseGeneGATLayer(num_genes, widths[1], widths[2], config.heads, config.dropout, edge_index),
                StaticSparseGeneGATLayer(num_genes, widths[2], widths[3], config.heads, config.dropout, edge_index),
            ]
        )
        self.pool_gate = nn.Linear(config.h3, 1)
        self.pool_value = nn.Linear(config.h3, config.h3)
        self.latent = nn.Sequential(nn.LayerNorm(config.h3), nn.Linear(config.h3, config.latent_dim))

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        gene_ids = torch.arange(self.num_genes, device=expression.device)
        x = self.expr_projection(expression.unsqueeze(-1)) + self.gene_embedding(gene_ids).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        gate = torch.softmax(self.pool_gate(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (gate * self.pool_value(x)).sum(dim=1)
        return self.latent(pooled)


class SymmetricGenePriorGATDecoder(nn.Module):
    def __init__(self, num_genes: int, edge_index: torch.Tensor, config: GenePriorGATConfig) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        widths = [config.h3, config.h2, config.h1, config.h0]
        self.latent_projection = nn.Linear(config.latent_dim, config.h3)
        self.gene_embedding = nn.Embedding(num_genes, config.h3)
        self.layers = nn.ModuleList(
            [
                StaticSparseGeneGATLayer(num_genes, widths[0], widths[1], config.heads, config.dropout, edge_index),
                StaticSparseGeneGATLayer(num_genes, widths[1], widths[2], config.heads, config.dropout, edge_index),
                StaticSparseGeneGATLayer(num_genes, widths[2], widths[3], config.heads, config.dropout, edge_index),
            ]
        )
        self.reconstruction_head = nn.Sequential(nn.LayerNorm(config.h0), nn.Linear(config.h0, 1))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        gene_ids = torch.arange(self.num_genes, device=latent.device)
        z = self.latent_projection(latent).unsqueeze(1)
        x = z + self.gene_embedding(gene_ids).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.reconstruction_head(x).squeeze(-1)


class GenePriorGATAutoEncoder(nn.Module):
    def __init__(self, num_genes: int, edge_index: torch.Tensor, config: GenePriorGATConfig) -> None:
        super().__init__()
        self.encoder = GenePriorGATEncoder(num_genes, edge_index, config)
        self.decoder = SymmetricGenePriorGATDecoder(num_genes, edge_index, config)

    def forward(self, expression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(expression)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def latent_variance_guard(latent: torch.Tensor) -> torch.Tensor:
    variance = latent.var(dim=0, unbiased=False)
    return F.relu(0.05 - variance).mean()


def edge_hidden_smoothness(latent: np.ndarray) -> float:
    if latent.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.var(latent, axis=0)))


def _make_loaders(x: np.ndarray, config: GenePriorGATConfig) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    rng = np.random.default_rng(config.seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    if config.max_train_cells is not None and config.max_train_cells > 0 and config.max_train_cells < len(indices):
        indices = indices[: config.max_train_cells]
    valid_n = max(1, int(round(x.shape[0] * config.val_fraction)))
    valid_n = min(valid_n, max(1, len(indices) // 3))
    valid_idx = indices[:valid_n]
    train_idx = indices[valid_n:]
    train = torch.utils.data.TensorDataset(torch.from_numpy(x[train_idx]))
    valid = torch.utils.data.TensorDataset(torch.from_numpy(x[valid_idx]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader


def _run_epoch(
    model: GenePriorGATAutoEncoder,
    loader: torch.utils.data.DataLoader,
    config: GenePriorGATConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_recon = 0.0
    total_var = 0.0
    n_obs = 0
    for (batch,) in loader:
        batch = batch.to(device)
        if training:
            mask = torch.rand_like(batch).lt(config.mask_prob)
            model_input = batch.masked_fill(mask, 0.0)
        else:
            model_input = batch
        with torch.set_grad_enabled(training):
            recon, latent = model(model_input)
            recon_loss = F.mse_loss(recon, batch)
            var_loss = latent_variance_guard(latent)
            loss = recon_loss + config.latent_var_weight * var_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_n = int(batch.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_n
        total_recon += float(recon_loss.detach().cpu()) * batch_n
        total_var += float(var_loss.detach().cpu()) * batch_n
        n_obs += batch_n
    return {
        "loss": total_loss / max(n_obs, 1),
        "reconstruction_mse": total_recon / max(n_obs, 1),
        "latent_var_penalty": total_var / max(n_obs, 1),
    }


@torch.no_grad()
def encode_expression(
    model: GenePriorGATAutoEncoder,
    x: Any,
    device: torch.device,
    batch_size: int,
    *,
    store_reconstruction: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    latents = []
    recon = [] if store_reconstruction else None
    n_obs = int(x.shape[0])
    for start in range(0, n_obs, batch_size):
        stop = min(start + batch_size, n_obs)
        batch_np = _dense_rows_float32(x, slice(start, stop))
        batch = torch.from_numpy(batch_np)
        batch = batch.to(device)
        if recon is None:
            latent = model.encoder(batch)
            decoded = None
        else:
            decoded, latent = model(batch)
        latents.append(latent.cpu().numpy())
        if recon is not None:
            recon.append(decoded.cpu().numpy())
    reconstruction = np.vstack(recon).astype(np.float32) if recon is not None else None
    return np.vstack(latents).astype(np.float32), reconstruction


def train_gene_prior_gatae(
    adata: ad.AnnData,
    *,
    network_path: str | Path,
    output_h5ad: str | Path,
    checkpoint_path: str | Path,
    summary_path: str | Path,
    trace_path: str | Path,
    layer_key: str = "lognorm",
    config: GenePriorGATConfig | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    config = config or GenePriorGATConfig()
    seed_all(config.seed)
    device = torch.device(device)
    output_h5ad = Path(output_h5ad)
    checkpoint_path = Path(checkpoint_path)
    summary_path = Path(summary_path)
    trace_path = Path(trace_path)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    gene_names = list(map(str, adata.var_names))
    graph = build_gene_prior_graph(
        gene_names,
        network_path,
        max_prior_neighbors_per_gene=config.max_prior_neighbors_per_gene,
    )
    expression_matrix = adata.layers[layer_key] if layer_key in adata.layers else adata.X
    x_train = _sample_training_matrix(expression_matrix, config)
    train_loader, valid_loader = _make_loaders(x_train, config)
    model = GenePriorGATAutoEncoder(len(gene_names), graph.edge_index.to(device), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_valid = math.inf
    bad_epochs = 0
    started = time.time()
    with trace_path.open("w", encoding="utf-8") as trace:
        for epoch in range(config.max_epochs):
            train_metrics = _run_epoch(model, train_loader, config, device, optimizer)
            valid_metrics = _run_epoch(model, valid_loader, config, device, None)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
                "seconds": round(time.time() - started, 3),
            }
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            if valid_metrics["loss"] < best_valid - 1e-5:
                best_valid = valid_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    latent, reconstruction = encode_expression(
        model,
        expression_matrix,
        device,
        config.batch_size,
        store_reconstruction=bool(config.store_reconstruction),
    )
    latent_mean = latent.mean(axis=0, keepdims=True)
    latent_std = latent.std(axis=0, keepdims=True)
    latent_std = np.where(latent_std < 1e-6, 1.0, latent_std)
    latent_standardized = ((latent - latent_mean) / latent_std).astype(np.float32)

    latent_key = f"X_gene_prior_gatae_{config.latent_dim}d"
    out = adata.copy()
    out.obsm[latent_key] = latent_standardized
    out.obsm["X_gene_prior_gatae_raw"] = latent
    if reconstruction is not None:
        out.layers["gene_prior_gatae_reconstruction"] = reconstruction.astype(np.float32)
    out.uns["gene_prior_gatae"] = {
        "latent_key": latent_key,
        "checkpoint_path": str(checkpoint_path),
        "network_path": str(network_path),
        "config": asdict(config),
        "graph": {
            "retained_prior_edges": graph.retained_prior_edges,
            "retained_prior_genes": graph.retained_prior_genes,
            "connected_components": graph.connected_components,
            "largest_component_size": graph.largest_component_size,
            "edge_index_edges_with_symmetry_and_selfloops": int(graph.edge_index.shape[1]),
        },
    }
    out.write_h5ad(output_h5ad)

    checkpoint = {
        "model_type": "gene_prior_gatae",
        "config": asdict(config),
        "gene_names": gene_names,
        "edge_index": graph.edge_index.cpu(),
        "state_dict": model.state_dict(),
        "latent_mean": latent_mean.squeeze(0).astype(np.float32),
        "latent_std": latent_std.squeeze(0).astype(np.float32),
        "latent_key": latent_key,
        "layer_key": layer_key,
    }
    torch.save(checkpoint, checkpoint_path)

    final_train = _run_epoch(model, train_loader, config, device, None)
    final_valid = _run_epoch(model, valid_loader, config, device, None)
    summary = {
        "status": "DONE",
        "output_h5ad": str(output_h5ad),
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "latent_key": latent_key,
        "input_shape": [int(adata.n_obs), int(adata.n_vars)],
        "layer_key": layer_key,
        "config": asdict(config),
        "graph": {
            "network_path": str(network_path),
            "retained_prior_edges": graph.retained_prior_edges,
            "retained_prior_genes": graph.retained_prior_genes,
            "connected_components": graph.connected_components,
            "largest_component_size": graph.largest_component_size,
            "edge_index_edges_with_symmetry_and_selfloops": int(graph.edge_index.shape[1]),
        },
        "final_train": final_train,
        "final_valid": final_valid,
        "reconstruction_mse_all": None
        if reconstruction is None
        else float(np.mean((reconstruction - _dense_float32(expression_matrix)) ** 2)),
        "latent_mean_abs_max": float(np.max(np.abs(latent_standardized.mean(axis=0)))),
        "latent_std_min": float(latent_standardized.std(axis=0).min()),
        "latent_std_max": float(latent_standardized.std(axis=0).max()),
        "latent_raw_variance_mean": edge_hidden_smoothness(latent),
        "seconds": round(time.time() - started, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_gene_prior_gatae_decoder(checkpoint_path: str | Path, device: str | torch.device) -> SymmetricGenePriorGATDecoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "gene_prior_gatae":
        raise ValueError(f"{checkpoint_path} is not a gene_prior_gatae checkpoint.")
    config = GenePriorGATConfig(**checkpoint["config"])
    edge_index = checkpoint["edge_index"].to(device)
    num_genes = len(checkpoint["gene_names"])
    model = GenePriorGATAutoEncoder(num_genes, edge_index, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    decoder = model.decoder
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    return decoder
