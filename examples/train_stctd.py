#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stctd.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from stctd.data import PreparedData, load_prepared_data  # noqa: E402
from stctd.evaluate import evaluate_stctd_model  # noqa: E402
from stctd.train import train_stctd_model  # noqa: E402
from stctd.utils import resolve_device, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stCTD model with the full time-grid objective.")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "stctd")
    parser.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--spatial-key", default="X_spatial_input")
    parser.add_argument("--latent-key", default="X_gene_input")
    parser.add_argument("--time-key", default="time_input")
    parser.add_argument("--raw-time-key", default="time")
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--expression-layer-key", default="lognorm")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None)
    parser.add_argument("--require-obs-value", nargs=2, action="append", metavar=("KEY", "VALUE"), default=[])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=6)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--velocity-parameterization", choices=["potential", "vector"], default="potential")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    parser.add_argument("--loss-mode", default="stctd_fullgrid", choices=["stctd_fullgrid", "endpoint"])
    parser.add_argument("--ode-step-size", type=float, default=0.25)
    parser.add_argument("--steps-per-interval", type=int, default=8)
    parser.add_argument("--lambda-match", type=float, default=4e5)
    parser.add_argument("--lambda-action", type=float, default=0.0)
    parser.add_argument("--lambda-ssp", type=float, default=0.0)
    parser.add_argument("--lambda-hj", type=float, default=0.0)
    parser.add_argument("--spatial-weight", type=float, default=3.0)
    parser.add_argument("--scale-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-exp", type=float, default=0.01)
    parser.add_argument("--alpha-gro", type=float, default=0.0002)
    parser.add_argument("--kappa-exp", type=float, default=0.02)
    parser.add_argument("--kappa-gro", type=float, default=0.1)
    parser.add_argument("--trace-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--no-sample-growth", action="store_true")
    parser.add_argument("--sample-growth-interval", type=int, default=100)
    parser.add_argument("--sample-growth-step", type=int, default=20)
    parser.add_argument("--use-cell-type-prior", action="store_true")
    parser.add_argument("--baseline-summary", type=Path, action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _obs_counts(data: PreparedData, key: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in data.adata.obs[key].astype(str).value_counts().items()}


def _check_required_obs_values(data: PreparedData, required: list[list[str]]) -> dict[str, dict[str, int]]:
    observed_counts: dict[str, dict[str, int]] = {}
    for key, value in required:
        if key not in data.adata.obs:
            raise ValueError(f"Required obs key is missing: {key}")
        counts = _obs_counts(data, key)
        observed_counts[key] = counts
        invalid = {label: count for label, count in counts.items() if label != value and count > 0}
        if invalid:
            raise ValueError(f"Expected obs[{key!r}] to contain only {value!r}, but found {invalid}")
    return observed_counts


def _metrics_rows(summary: dict[str, Any], baseline_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_metrics = summary["evaluation"]["metrics"]
    method_name = str(summary["config"].get("method", "stCTD"))
    rows.append(
        {
            "method": method_name,
            "run_dir": summary["run_dir"],
            "avg_label_prop_corr": float(np.mean([m["label_prop_corr"] for m in eval_metrics])),
            "avg_label_prop_l1": float(np.mean([m["label_prop_l1"] for m in eval_metrics])),
            "avg_spatial_chamfer_norm": float(np.mean([m["spatial_chamfer_norm"] for m in eval_metrics])),
            "avg_spatial_grid_iou": float(np.mean([m["spatial_grid_iou"] for m in eval_metrics])),
            "final_label_prop_corr": float(eval_metrics[-1]["label_prop_corr"]),
            "final_label_prop_l1": float(eval_metrics[-1]["label_prop_l1"]),
            "final_spatial_chamfer_norm": float(eval_metrics[-1]["spatial_chamfer_norm"]),
            "final_spatial_grid_iou": float(eval_metrics[-1]["spatial_grid_iou"]),
            "summary_json": str(Path(summary["run_dir"]) / "summary.json"),
        }
    )
    for path in baseline_paths:
        if not path.exists():
            continue
        baseline = _load_json(path)
        metrics = baseline.get("metrics", [])
        if not metrics:
            continue
        cfg = baseline.get("config", {})
        rows.append(
            {
                "method": f"baseline lambda_hj={float(cfg.get('lambda_hj', cfg.get('lambda_hjb', 0.0))):g}",
                "run_dir": baseline.get("run_dir", str(path.parent)),
                "avg_label_prop_corr": float(np.mean([m["label_prop_corr"] for m in metrics])),
                "avg_label_prop_l1": float(np.mean([m["label_prop_l1"] for m in metrics])),
                "avg_spatial_chamfer_norm": float(np.mean([m["spatial_chamfer_norm"] for m in metrics])),
                "avg_spatial_grid_iou": float(np.mean([m["spatial_grid_iou"] for m in metrics])),
                "final_label_prop_corr": float(metrics[-1]["label_prop_corr"]),
                "final_label_prop_l1": float(metrics[-1]["label_prop_l1"]),
                "final_spatial_chamfer_norm": float(metrics[-1]["spatial_chamfer_norm"]),
                "final_spatial_grid_iou": float(metrics[-1]["spatial_grid_iou"]),
                "summary_json": str(path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    data_config = DataConfig(
        input_h5ad=args.input_h5ad,
        network_tsv=REPO_ROOT / "references" / "gene_networks" / "network_mouse.tsv",
        gat_out_dir=args.out_dir / "unused_gat",
        use_precomputed_embedding=True,
        decoder_checkpoint=args.decoder_checkpoint,
        spatial_key=args.spatial_key,
        latent_key=args.latent_key,
        time_key=args.time_key,
        raw_time_key=args.raw_time_key,
        annotation_key=args.annotation_key,
        expression_layer_key=args.expression_layer_key,
        spatial_weight=float(args.spatial_weight),
        scale_features=bool(args.scale_features),
    )
    data, decoder_checkpoint = load_prepared_data(data_config, device)
    required_obs_counts = _check_required_obs_values(data, list(args.require_obs_value))

    model_config = ModelConfig(
        spatial_dim=data.spatial_dim,
        latent_dim=data.latent_dim,
        hidden_dim=int(args.hidden_dim),
        n_hidden=int(args.n_hidden),
        activation=str(args.activation),
        velocity_parameterization=str(args.velocity_parameterization),
    )
    train_config = TrainConfig(
        device=str(device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        sample_size=int(args.sample_size),
        lr=float(args.lr),
        optimizer=str(args.optimizer),
        loss_mode=str(args.loss_mode),
        ode_step_size=float(args.ode_step_size),
        steps_per_interval=int(args.steps_per_interval),
        use_bidirectional=True,
        use_growth=True,
        lambda_match=float(args.lambda_match),
        lambda_action=float(args.lambda_action),
        lambda_ssp=float(args.lambda_ssp),
        lambda_hj=float(args.lambda_hj),
        alpha_exp=float(args.alpha_exp),
        alpha_gro=float(args.alpha_gro),
        kappa_exp=float(args.kappa_exp),
        kappa_gro=float(args.kappa_gro),
        use_cell_type_prior=bool(args.use_cell_type_prior),
        cell_type_prior_min_count=5,
        increase_sample_size=not bool(args.no_sample_growth),
        sample_growth_interval=int(args.sample_growth_interval),
        sample_growth_step=int(args.sample_growth_step),
        trace_interval=int(args.trace_interval),
    )
    method_name = "stCTD" if args.velocity_parameterization == "potential" else "stCTD vector ablation"
    config_payload = {
        "method": method_name,
        "data": data_config.to_json_dict(),
        "model": model_config.to_json_dict(),
        "train": train_config.to_json_dict(),
        "required_obs_counts": required_obs_counts,
        "observed_shape": list(data.adata.shape),
        "note": "Dataset filtering is controlled only by the explicit input h5ad and optional obs guards.",
    }
    write_json(args.out_dir / "config.json", config_payload)

    model, train_summary = train_stctd_model(
        data,
        decoder_checkpoint,
        model_config,
        train_config,
        output_dir=args.out_dir / "checkpoints",
    )
    eval_summary = evaluate_stctd_model(
        model,
        data,
        decoder_checkpoint,
        train_config,
        data_config,
        output_dir=args.out_dir,
    )
    summary = {
        "status": "DONE",
        "run_dir": str(args.out_dir),
        "seconds": round(time.time() - started, 2),
        "train_summary": train_summary,
        "evaluation": eval_summary,
        "config": config_payload,
    }
    rows = _metrics_rows(summary, list(args.baseline_summary))
    comparison_csv = args.out_dir / "stctd_vs_baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)
    summary["comparison_csv"] = str(comparison_csv)
    summary["comparison_rows"] = rows
    write_json(args.out_dir / "summary.json", summary)

    if args.quiet:
        compact = {
            "status": summary["status"],
            "run_dir": summary["run_dir"],
            "seconds": summary["seconds"],
            "required_obs_counts": required_obs_counts,
            "stctd": rows[0],
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
