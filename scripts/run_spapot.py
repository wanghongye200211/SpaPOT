#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spapot.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from spapot.data import load_prepared_data  # noqa: E402
from spapot.evaluate import evaluate_full_model  # noqa: E402
from spapot.train import train_full_model  # noqa: E402
from spapot.utils import resolve_device, write_json  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "prepared_input.h5ad"
DEFAULT_NETWORK = PROJECT_ROOT / "references" / "network_mouse.tsv"
DEFAULT_GAT_DIR = PROJECT_ROOT / "intermediates" / "gene_prior_gatae"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "spapot_hybrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpaPOT on a prepared spatial transcriptomics trajectory.")
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--network-tsv", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--gat-out-dir", type=Path, default=DEFAULT_GAT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--use-precomputed-embedding", action="store_true")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None)
    parser.add_argument("--spatial-key", type=str, default="X_spatial_input")
    parser.add_argument("--latent-key", type=str, default="X_gene_input")
    parser.add_argument("--time-key", type=str, default="time_input")
    parser.add_argument("--raw-time-key", type=str, default="time")
    parser.add_argument("--annotation-key", type=str, default="Annotation")
    parser.add_argument("--expression-layer-key", type=str, default="lognorm")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--gat-device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--force-rebuild-gat", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps-per-interval", type=int, default=8)
    parser.add_argument("--spatial-weight", type=float, default=3.0)
    parser.add_argument("--lambda-state-ot", type=float, default=1.0)
    parser.add_argument("--lambda-spatial-ot", type=float, default=0.5)
    parser.add_argument("--lambda-expr", type=float, default=0.05)
    parser.add_argument("--lambda-mass-global", type=float, default=0.25)
    parser.add_argument("--lambda-mass-local", type=float, default=1.0)
    parser.add_argument("--lambda-rollout-spatial-ot", type=float, default=0.0)
    parser.add_argument("--lambda-rollout-mass-global", type=float, default=0.0)
    parser.add_argument("--rollout-start-epoch", type=int, default=0)
    parser.add_argument("--rollout-ramp-epochs", type=int, default=0)
    parser.add_argument("--lambda-spatial-deform", type=float, default=0.0)
    parser.add_argument("--lambda-spatial-coverage", type=float, default=0.0)
    parser.add_argument("--spatial-deform-neighbors", type=int, default=8)
    parser.add_argument("--spatial-coverage-bandwidth", type=float, default=0.35)
    parser.add_argument("--spatial-coverage-anchor-count", type=int, default=256)
    parser.add_argument("--lambda-action", type=float, default=1e-3)
    parser.add_argument("--state-spatial-cost-weight", type=float, default=0.7)
    parser.add_argument("--state-gene-cost-weight", type=float, default=0.3)
    parser.add_argument("--potential-z-only", action="store_true")
    parser.add_argument("--use-cell-type-prior", action="store_true")
    parser.add_argument("--cell-type-prior-min-count", type=int, default=3)
    parser.add_argument("--local-mass-bandwidth", type=float, default=0.75)
    parser.add_argument("--local-mass-anchor-count", type=int, default=256)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--no-growth", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    data_config = DataConfig(
        input_h5ad=args.input_h5ad.expanduser().resolve(),
        network_tsv=args.network_tsv.expanduser().resolve(),
        gat_out_dir=args.gat_out_dir.expanduser().resolve(),
        use_precomputed_embedding=bool(args.use_precomputed_embedding),
        decoder_checkpoint=args.decoder_checkpoint.expanduser().resolve() if args.decoder_checkpoint is not None else None,
        spatial_key=args.spatial_key,
        latent_key=args.latent_key,
        time_key=args.time_key,
        raw_time_key=args.raw_time_key,
        annotation_key=args.annotation_key,
        expression_layer_key=args.expression_layer_key,
        spatial_weight=float(args.spatial_weight),
        force_rebuild_gat=bool(args.force_rebuild_gat),
        gat_device=args.gat_device,
    )
    data, decoder_checkpoint = load_prepared_data(data_config, device)
    model_config = ModelConfig(
        spatial_dim=data.spatial_dim,
        latent_dim=data.latent_dim,
        potential_depends_on_spatial=not bool(args.potential_z_only),
    )
    train_config = TrainConfig(
        device=str(device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        sample_size=int(args.sample_size),
        lr=float(args.lr),
        steps_per_interval=int(args.steps_per_interval),
        use_bidirectional=not bool(args.no_bidirectional),
        use_growth=not bool(args.no_growth),
        lambda_state_ot=float(args.lambda_state_ot),
        lambda_spatial_ot=float(args.lambda_spatial_ot),
        lambda_expr=float(args.lambda_expr),
        lambda_mass_global=float(args.lambda_mass_global),
        lambda_mass_local=float(args.lambda_mass_local),
        lambda_rollout_spatial_ot=float(args.lambda_rollout_spatial_ot),
        lambda_rollout_mass_global=float(args.lambda_rollout_mass_global),
        rollout_start_epoch=int(args.rollout_start_epoch),
        rollout_ramp_epochs=int(args.rollout_ramp_epochs),
        lambda_spatial_deform=float(args.lambda_spatial_deform),
        lambda_spatial_coverage=float(args.lambda_spatial_coverage),
        spatial_deform_neighbors=int(args.spatial_deform_neighbors),
        spatial_coverage_bandwidth=float(args.spatial_coverage_bandwidth),
        spatial_coverage_anchor_count=int(args.spatial_coverage_anchor_count),
        lambda_action=float(args.lambda_action),
        state_spatial_cost_weight=float(args.state_spatial_cost_weight),
        state_gene_cost_weight=float(args.state_gene_cost_weight),
        use_cell_type_prior=bool(args.use_cell_type_prior),
        cell_type_prior_min_count=int(args.cell_type_prior_min_count),
        local_mass_bandwidth=float(args.local_mass_bandwidth),
        local_mass_anchor_count=int(args.local_mass_anchor_count),
    )
    config_payload = {
        "data": data_config.to_json_dict(),
        "model": model_config.to_json_dict(),
        "train": train_config.to_json_dict(),
        "decoder_checkpoint": str(decoder_checkpoint) if decoder_checkpoint is not None else None,
    }
    write_json(run_dir / "config.json", config_payload)
    model, train_summary = train_full_model(
        data,
        decoder_checkpoint,
        model_config,
        train_config,
        output_dir=run_dir / "checkpoints",
    )
    eval_summary = None
    if not args.skip_eval:
        eval_summary = evaluate_full_model(
            model,
            data,
            decoder_checkpoint,
            train_config,
            data_config,
            output_dir=run_dir,
        )
    summary = {
        "status": "DONE",
        "run_dir": str(run_dir),
        "data": data_config.to_json_dict(),
        "model": model_config.to_json_dict(),
        "train": train_config.to_json_dict(),
        "decoder_checkpoint": str(decoder_checkpoint) if decoder_checkpoint is not None else None,
        "train_summary": train_summary,
        "evaluation": eval_summary,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
