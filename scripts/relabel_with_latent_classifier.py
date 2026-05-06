#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spapot.config import DataConfig  # noqa: E402
from spapot.data import load_prepared_data  # noqa: E402
from spapot.evaluate import _label_metrics, _plot_real_pred, _spatial_metrics  # noqa: E402
from spapot.latent_classifier import (  # noqa: E402
    LatentClassifierConfig,
    predict_latent_labels,
    train_latent_classifier,
)
from spapot.utils import resolve_device, write_json  # noqa: E402


DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "spapot_hybrid"
DEFAULT_NETWORK = PROJECT_ROOT / "references" / "network_mouse.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel SpaPOT predictions with a latent-z classifier.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-h5ad", type=Path, default=None)
    parser.add_argument("--output-subdir", default="latent_classifier_eval")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--spatial-key", default="X_spatial_input")
    parser.add_argument("--latent-key", default="X_gene_input")
    parser.add_argument("--time-key", default="time_input")
    parser.add_argument("--raw-time-key", default="time")
    parser.add_argument("--annotation-key", default="cell_type")
    parser.add_argument("--expression-layer-key", default="X")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-class-balance", action="store_true")
    return parser.parse_args()


def _path_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path_keys = {"input_h5ad", "network_tsv", "gat_out_dir"}
    return {key: Path(value) if key in path_keys else value for key, value in payload.items()}


def _suffix(raw_time: float) -> str:
    return str(raw_time).replace(".", "p")


def _load_data_config(run_dir: Path, args: argparse.Namespace) -> DataConfig:
    config_path = run_dir / "config.json"
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        return DataConfig(**_path_payload(config_payload["data"]))

    summary_path = run_dir / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    input_h5ad = args.input_h5ad or Path(summary_payload.get("input_h5ad", ""))
    if not input_h5ad:
        raise FileNotFoundError(f"No config.json and no input_h5ad found for {run_dir}")
    return DataConfig(
        input_h5ad=input_h5ad.expanduser().resolve(),
        network_tsv=DEFAULT_NETWORK,
        gat_out_dir=run_dir / "latent_classifier_eval" / "unused_gat",
        use_precomputed_embedding=True,
        spatial_key=args.spatial_key,
        latent_key=args.latent_key,
        time_key=args.time_key,
        raw_time_key=args.raw_time_key,
        annotation_key=args.annotation_key,
        expression_layer_key=args.expression_layer_key,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = run_dir / args.output_subdir
    pred_out = output_dir / "predictions"
    comparison_dir = output_dir / "comparisons"
    pred_out.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    data_config = _load_data_config(run_dir, args)
    device = resolve_device(args.device)
    data, _ = load_prepared_data(data_config, device)
    data.adata.obs[data_config.annotation_key] = data.adata.obs[data_config.annotation_key].astype("category")

    classifier_config = LatentClassifierConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        hidden_dim=int(args.hidden_dim),
        n_hidden=int(args.n_hidden),
        dropout=float(args.dropout),
        class_balance=not bool(args.no_class_balance),
    )
    classifier = train_latent_classifier(
        data,
        data_config.annotation_key,
        output_dir=output_dir / "classifier",
        config=classifier_config,
    )

    observed = data.adata.copy()
    observed.obs[data_config.annotation_key] = observed.obs[data_config.annotation_key].astype("category")
    observed.obsm["X_spatial_aligned"] = np.asarray(observed.obsm[data_config.spatial_key], dtype=np.float32)

    metrics = []
    pred_in_dir = run_dir / "predictions"
    for target_index, raw_time in enumerate(data.raw_time_values):
        suffix = _suffix(raw_time)
        pred_path = pred_in_dir / f"predict_{suffix}.h5ad"
        pred = ad.read_h5ad(pred_path)
        latent = np.asarray(pred.obsm[data_config.latent_key], dtype=np.float32)
        labels = predict_latent_labels(classifier, latent, device=device)
        pred.obs[data_config.annotation_key] = pd.Categorical(
            labels,
            categories=data.adata.obs[data_config.annotation_key].cat.categories,
        )
        pred.uns["Annotation_colors"] = data.adata.uns.get("Annotation_colors", [])
        pred.uns[f"{data_config.annotation_key}_colors"] = data.adata.uns.get(
            f"{data_config.annotation_key}_colors",
            pred.uns["Annotation_colors"],
        )
        out_pred_path = pred_out / f"predict_{suffix}.h5ad"
        pred.write_h5ad(out_pred_path)

        real = observed[np.isclose(observed.obs[data_config.raw_time_key].astype(float), raw_time)].copy()
        real.obs[data_config.annotation_key] = real.obs[data_config.annotation_key].astype("category")
        row: dict[str, Any] = {
            "time": float(raw_time),
            "time_input": float(data.time_values[target_index]),
            "pred_h5ad": str(out_pred_path),
        }
        row.update(_label_metrics(real, pred, data_config.annotation_key))
        row.update(_spatial_metrics(real, pred, data_config.annotation_key))
        plot_path = comparison_dir / f"{suffix}_real_vs_pred.png"
        _plot_real_pred(real, pred, plot_path, f"SpaPOT latent classifier E{raw_time:g}", data_config.annotation_key)
        row["compare_png"] = str(plot_path)
        metrics.append(row)

    metrics_csv = output_dir / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    summary = {
        "status": "DONE",
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "classifier": {
            "checkpoint": str(classifier.checkpoint_path),
            "trace": str(classifier.trace_path),
            "config": classifier_config.to_json_dict(),
            "categories": classifier.categories,
        },
        "metrics_csv": str(metrics_csv),
        "final": metrics[-1],
        "metrics": metrics,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
