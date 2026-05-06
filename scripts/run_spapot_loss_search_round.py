#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_spapot.py"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "prepared_input.h5ad"
FALLBACK_INPUT = PROJECT_ROOT / "data" / "sim_data_heart2duck.h5ad"
DEFAULT_NETWORK = PROJECT_ROOT / "references" / "network_mouse.tsv"
DEFAULT_GAT_DIR = PROJECT_ROOT / "intermediates" / "loss_search_unused_gat"


@dataclass(frozen=True)
class Variant:
    name: str
    hypothesis: str
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one visual-first SpaPOT loss-search round.")
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "runs" / "spapot_loss_search")
    parser.add_argument("--round-name", default="round01_rollout_prior_deform")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--sample-size", type=int, default=384)
    parser.add_argument("--steps-per-interval", type=int, default=8)
    parser.add_argument("--annotation-key", default="cell_type")
    parser.add_argument("--spatial-key", default="X_spatial_input")
    parser.add_argument("--latent-key", default="X_gene_input")
    parser.add_argument("--time-key", default="time_input")
    parser.add_argument("--raw-time-key", default="time")
    parser.add_argument("--expression-layer-key", default="X")
    parser.add_argument("--preset", choices=["r01", "r02", "r03", "r04", "r05", "r06", "r07", "r08", "r09", "r10", "r11", "r12", "r13", "r14"], default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def r01_variants() -> list[Variant]:
    return [
        Variant(
            name="r01_spatial_rollout_deform",
            hypothesis=(
                "Increase spatial consistency by adding multi-step rollout spatial OT and a small local "
                "spatial deformation penalty; keep mass terms weak."
            ),
            args=(
                "--state-spatial-cost-weight",
                "0.90",
                "--state-gene-cost-weight",
                "0.10",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.75",
                "--lambda-rollout-spatial-ot",
                "0.30",
                "--lambda-rollout-mass-global",
                "0.03",
                "--rollout-start-epoch",
                "40",
                "--rollout-ramp-epochs",
                "40",
                "--lambda-mass-global",
                "0.03",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.05",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.0005",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r01_typeprior_rollout",
            hypothesis=(
                "Use cell-type prior inside matching so minority/source labels are not swallowed by the dominant "
                "geometry; add weaker rollout spatial OT."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.85",
                "--state-gene-cost-weight",
                "0.15",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.65",
                "--lambda-rollout-spatial-ot",
                "0.25",
                "--lambda-rollout-mass-global",
                "0.02",
                "--rollout-start-epoch",
                "40",
                "--rollout-ramp-epochs",
                "40",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.0005",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r01_gene_preserve_typeprior",
            hypothesis=(
                "Keep more gene-latent weight while using cell-type prior; test whether visual failures are caused "
                "by over-spatial matching erasing latent identity."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.70",
                "--state-gene-cost-weight",
                "0.30",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.35",
                "--lambda-rollout-spatial-ot",
                "0.15",
                "--lambda-rollout-mass-global",
                "0.02",
                "--rollout-start-epoch",
                "40",
                "--rollout-ramp-epochs",
                "40",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.0002",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r02_variants() -> list[Variant]:
    return [
        Variant(
            name="r02_geneprior_softspatial",
            hypothesis=(
                "Visual follow-up from round01: keep type prior and more gene cost, reduce extra spatial OT, "
                "and remove rollout spatial OT to avoid the large final-shape drift."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r02_typeprior_action_smooth",
            hypothesis=(
                "Keep type prior and moderately spatial matching, but increase action regularization and add "
                "a small spatial deformation penalty so the final cloud does not rotate/scatter."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.75",
                "--state-gene-cost-weight",
                "0.25",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.30",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.02",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.002",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r02_no_typeprior_soft_rollout",
            hypothesis=(
                "Control for whether type-prior grouped OT itself over-constrains the geometry: no type prior, "
                "balanced gene/spatial cost, very weak late rollout only."
            ),
            args=(
                "--state-spatial-cost-weight",
                "0.70",
                "--state-gene-cost-weight",
                "0.30",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.08",
                "--lambda-rollout-mass-global",
                "0.01",
                "--rollout-start-epoch",
                "70",
                "--rollout-ramp-epochs",
                "30",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r03_variants() -> list[Variant]:
    return [
        Variant(
            name="r03_geneprior_spatial015",
            hypothesis=(
                "Start from the best visual round02 setting and reduce the extra spatial OT from 0.20 to 0.15. "
                "Goal: keep the lower pale branch while reducing horizontal smearing."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.15",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r03_geneprior_action003",
            hypothesis=(
                "Keep the round02 visual baseline but increase action regularization to 0.003. "
                "Goal: smooth the migration enough to reduce scattered points without collapsing the branch."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-action",
                "0.003",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r03_geneprior_deform005",
            hypothesis=(
                "Keep the round02 visual baseline and add a very small spatial deformation penalty. "
                "Goal: preserve local spatial neighborhoods without imposing the stronger r01 deformation drift."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.005",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r04_variants() -> list[Variant]:
    return [
        Variant(
            name="r04_deform002",
            hypothesis=(
                "Fine-tune around r03_deform005 with a smaller deformation weight 0.002. "
                "Goal: keep mass and label gains while reducing over-smoothing of the lower branch."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.002",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r04_deform010",
            hypothesis=(
                "Fine-tune around r03_deform005 with a larger deformation weight 0.01. "
                "Goal: test whether stronger local neighborhood preservation improves the visible silhouette."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r04_deform005_late_rollout003",
            hypothesis=(
                "Keep the best r03 deformation setting and add only a tiny late rollout spatial OT. "
                "Goal: see if multi-step consistency can improve final placement without the r01 drift."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.03",
                "--lambda-rollout-mass-global",
                "0.005",
                "--rollout-start-epoch",
                "80",
                "--rollout-ramp-epochs",
                "30",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.005",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r05_variants() -> list[Variant]:
    return [
        Variant(
            name="r05_deform0075",
            hypothesis=(
                "Fine-tune the visual best r04_deform010 downward to 0.0075. "
                "Goal: retain label/mass stability while making the lower branch less sparse."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.0075",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r05_deform015",
            hypothesis=(
                "Fine-tune the visual best r04_deform010 upward to 0.015. "
                "Goal: test if stronger local smoothness further stabilizes minority label and branch shape."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.20",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.015",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r05_deform010_spatial025",
            hypothesis=(
                "Keep deformation at the visual best 0.01 but slightly increase extra spatial OT to 0.25. "
                "Goal: improve silhouette occupancy without using rollout."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r06_variants() -> list[Variant]:
    return [
        Variant(
            name="r06_spatial022",
            hypothesis=(
                "Start from r05_deform010_spatial025 and reduce extra spatial OT to 0.22. "
                "Goal: keep good mass while reducing possible over-spatial smoothing of the lower branch."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.22",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r06_spatial028",
            hypothesis=(
                "Start from r05_deform010_spatial025 and increase extra spatial OT to 0.28. "
                "Goal: test if a little more occupancy pressure improves the upper body silhouette."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.28",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r06_spatial030",
            hypothesis=(
                "Probe the upper end of extra spatial OT without rollout. "
                "Goal: determine whether the spatial pressure improvement saturates or begins to damage labels/mass."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.30",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r07_variants() -> list[Variant]:
    return [
        Variant(
            name="r07_state6040",
            hypothesis=(
                "Use the current visual baseline but shift state matching toward gene identity: spatial/gene 0.60/0.40. "
                "Goal: preserve cell identity and lower branch without losing too much silhouette."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.60",
                "--state-gene-cost-weight",
                "0.40",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r07_state7030",
            hypothesis=(
                "Use the current visual baseline but shift state matching toward spatial structure: spatial/gene 0.70/0.30. "
                "Goal: improve body occupancy while checking whether labels/mass degrade."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.70",
                "--state-gene-cost-weight",
                "0.30",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r07_state6545_action002",
            hypothesis=(
                "Intermediate gene-preserving balance with slightly higher action regularization. "
                "Goal: reduce scatter while preserving the lower branch and minority region."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.65",
                "--state-gene-cost-weight",
                "0.35",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.002",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r08_variants() -> list[Variant]:
    return [
        Variant(
            name="r08_state5842",
            hypothesis=(
                "Fine-tune around r07_state6040 with slightly more gene identity weight: spatial/gene 0.58/0.42. "
                "Goal: preserve the lower branch and minority type while keeping the improved silhouette."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.58",
                "--state-gene-cost-weight",
                "0.42",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r08_state6238",
            hypothesis=(
                "Fine-tune around r07_state6040 with slightly more spatial weight: spatial/gene 0.62/0.38. "
                "Goal: test if the fuller body can be retained with better centroid/mass."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.62",
                "--state-gene-cost-weight",
                "0.38",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.02",
                "--lambda-mass-local",
                "0.02",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
        Variant(
            name="r08_state6040_mass010",
            hypothesis=(
                "Keep r07_state6040 but reduce global/local mass regularization from 0.02 to 0.01. "
                "Goal: check whether mass terms are slightly biasing geometry while preserving acceptable growth."
            ),
            args=(
                "--use-cell-type-prior",
                "--cell-type-prior-min-count",
                "5",
                "--state-spatial-cost-weight",
                "0.60",
                "--state-gene-cost-weight",
                "0.40",
                "--lambda-state-ot",
                "1.0",
                "--lambda-spatial-ot",
                "0.25",
                "--lambda-rollout-spatial-ot",
                "0.0",
                "--lambda-rollout-mass-global",
                "0.0",
                "--lambda-mass-global",
                "0.01",
                "--lambda-mass-local",
                "0.01",
                "--lambda-spatial-deform",
                "0.01",
                "--spatial-deform-neighbors",
                "8",
                "--lambda-action",
                "0.001",
                "--lambda-expr",
                "0.0",
            ),
        ),
    ]


def r09_variants() -> list[Variant]:
    base = (
        "--use-cell-type-prior",
        "--cell-type-prior-min-count",
        "5",
        "--state-spatial-cost-weight",
        "0.60",
        "--state-gene-cost-weight",
        "0.40",
        "--lambda-state-ot",
        "1.0",
        "--lambda-spatial-ot",
        "0.25",
        "--lambda-rollout-spatial-ot",
        "0.0",
        "--lambda-rollout-mass-global",
        "0.0",
        "--lambda-mass-global",
        "0.02",
        "--lambda-mass-local",
        "0.02",
        "--lambda-spatial-deform",
        "0.01",
        "--spatial-deform-neighbors",
        "8",
        "--lambda-action",
        "0.001",
        "--lambda-expr",
        "0.0",
    )
    return [
        Variant(
            name="r09_state6040_seed19491002",
            hypothesis="Seed stability check for the current best r07_state6040 setting.",
            args=base + ("--seed", "19491002"),
        ),
        Variant(
            name="r09_state6040_seed19491003",
            hypothesis="Second seed stability check for the current best r07_state6040 setting.",
            args=base + ("--seed", "19491003"),
        ),
        Variant(
            name="r09_state6040_seed19491004",
            hypothesis="Third seed stability check for the current best r07_state6040 setting.",
            args=base + ("--seed", "19491004"),
        ),
    ]


def r10_variants() -> list[Variant]:
    base = (
        "--use-cell-type-prior",
        "--cell-type-prior-min-count",
        "5",
        "--state-spatial-cost-weight",
        "0.60",
        "--state-gene-cost-weight",
        "0.40",
        "--lambda-state-ot",
        "1.0",
        "--lambda-spatial-ot",
        "0.25",
        "--lambda-rollout-spatial-ot",
        "0.0",
        "--lambda-rollout-mass-global",
        "0.0",
        "--lambda-mass-global",
        "0.02",
        "--lambda-mass-local",
        "0.02",
        "--lambda-spatial-deform",
        "0.01",
        "--spatial-deform-neighbors",
        "8",
        "--lambda-action",
        "0.001",
        "--lambda-expr",
        "0.0",
    )
    return [
        Variant(
            name="r10_lr5e4_seed19491003",
            hypothesis="Stability intervention: lower lr=5e-4 on a previously mediocre seed.",
            args=base + ("--lr", "0.0005", "--seed", "19491003"),
        ),
        Variant(
            name="r10_lr5e4_seed19491004",
            hypothesis="Stability intervention: lower lr=5e-4 on the previous best seed.",
            args=base + ("--lr", "0.0005", "--seed", "19491004"),
        ),
        Variant(
            name="r10_lr3e4_seed19491003",
            hypothesis="Stronger stability intervention: lower lr=3e-4 on a previously mediocre seed.",
            args=base + ("--lr", "0.0003", "--seed", "19491003"),
        ),
    ]


def r11_variants() -> list[Variant]:
    def base(spatial_ot: str, seed: str) -> tuple[str, ...]:
        return (
            "--use-cell-type-prior",
            "--cell-type-prior-min-count",
            "5",
            "--state-spatial-cost-weight",
            "0.60",
            "--state-gene-cost-weight",
            "0.40",
            "--lambda-state-ot",
            "1.0",
            "--lambda-spatial-ot",
            spatial_ot,
            "--lambda-rollout-spatial-ot",
            "0.0",
            "--lambda-rollout-mass-global",
            "0.0",
            "--lambda-mass-global",
            "0.02",
            "--lambda-mass-local",
            "0.02",
            "--lambda-spatial-deform",
            "0.01",
            "--spatial-deform-neighbors",
            "8",
            "--lambda-action",
            "0.001",
            "--lambda-expr",
            "0.0",
            "--lr",
            "0.0005",
            "--seed",
            seed,
        )

    return [
        Variant(
            name="r11_lr5e4_e220_spatial025_seed03",
            hypothesis="Low-lr longer training at the current spatial OT baseline; tests whether coverage recovers with more epochs.",
            args=base("0.25", "19491003"),
        ),
        Variant(
            name="r11_lr5e4_e220_spatial028_seed03",
            hypothesis="Low-lr longer training with slightly stronger spatial OT; tests whether silhouette coverage improves without rollout.",
            args=base("0.28", "19491003"),
        ),
        Variant(
            name="r11_lr5e4_e220_spatial028_seed04",
            hypothesis="Repeat the low-lr stronger-spatial setting on the previously best seed to assess robustness.",
            args=base("0.28", "19491004"),
        ),
    ]


def r12_variants() -> list[Variant]:
    def base(lambda_coverage: str, bandwidth: str, spatial_ot: str = "0.25") -> tuple[str, ...]:
        return (
            "--use-cell-type-prior",
            "--cell-type-prior-min-count",
            "5",
            "--state-spatial-cost-weight",
            "0.60",
            "--state-gene-cost-weight",
            "0.40",
            "--lambda-state-ot",
            "1.0",
            "--lambda-spatial-ot",
            spatial_ot,
            "--lambda-rollout-spatial-ot",
            "0.0",
            "--lambda-rollout-mass-global",
            "0.0",
            "--lambda-mass-global",
            "0.02",
            "--lambda-mass-local",
            "0.02",
            "--lambda-spatial-deform",
            "0.01",
            "--spatial-deform-neighbors",
            "8",
            "--lambda-spatial-coverage",
            lambda_coverage,
            "--spatial-coverage-bandwidth",
            bandwidth,
            "--spatial-coverage-anchor-count",
            "256",
            "--lambda-action",
            "0.001",
            "--lambda-expr",
            "0.0",
            "--lr",
            "0.0005",
            "--seed",
            "19491003",
        )

    return [
        Variant(
            name="r12_coverage005_bw035",
            hypothesis="Add very weak target-anchor undercoverage loss to the r11 visual winner; aim to fill terminal silhouette holes without moving the centroid.",
            args=base("0.005", "0.35"),
        ),
        Variant(
            name="r12_coverage010_bw035",
            hypothesis="Double the weak undercoverage pressure; tests whether coverage improves before labels/mass degrade.",
            args=base("0.010", "0.35"),
        ),
        Variant(
            name="r12_coverage005_bw050",
            hypothesis="Use a broader undercoverage kernel; tests whether boundary/body support improves more smoothly than stronger spatial OT.",
            args=base("0.005", "0.50"),
        ),
    ]


def r13_variants() -> list[Variant]:
    def base(spatial_ot: str, rollout_spatial: str = "0.0") -> tuple[str, ...]:
        return (
            "--use-cell-type-prior",
            "--cell-type-prior-min-count",
            "5",
            "--state-spatial-cost-weight",
            "0.60",
            "--state-gene-cost-weight",
            "0.40",
            "--lambda-state-ot",
            "1.0",
            "--lambda-spatial-ot",
            spatial_ot,
            "--lambda-rollout-spatial-ot",
            rollout_spatial,
            "--lambda-rollout-mass-global",
            "0.0",
            "--rollout-start-epoch",
            "50",
            "--rollout-ramp-epochs",
            "25",
            "--lambda-mass-global",
            "0.0",
            "--lambda-mass-local",
            "0.0",
            "--lambda-spatial-deform",
            "0.01",
            "--spatial-deform-neighbors",
            "8",
            "--lambda-action",
            "0.001",
            "--lambda-expr",
            "0.0",
            "--lr",
            "0.0005",
            "--seed",
            "19491003",
        )

    return [
        Variant(
            name="r13_nomass_spatial025",
            hypothesis=(
                "Remove external mass losses from the r11-style setting. Test whether mass terms were subtly "
                "hurting visible spatial silhouette while matching already carries distribution information."
            ),
            args=base("0.25"),
        ),
        Variant(
            name="r13_nomass_spatial030",
            hypothesis=(
                "No external mass loss, but slightly stronger spatial OT. Goal: recover final silhouette "
                "coverage without the centroid drift seen in coverage-loss variants."
            ),
            args=base("0.30"),
        ),
        Variant(
            name="r13_nomass_late_rollout015",
            hypothesis=(
                "No external mass loss and only a tiny late multi-step spatial rollout. Goal: improve temporal "
                "coherence while avoiding the large rollout drift from early rounds."
            ),
            args=base("0.25", "0.015"),
        ),
    ]


def r14_variants() -> list[Variant]:
    def base(
        *,
        spatial_weight: str,
        gene_weight: str,
        spatial_ot: str,
        use_type_prior: bool,
        potential_z_only: bool = False,
    ) -> tuple[str, ...]:
        args = [
            "--cell-type-prior-min-count",
            "5",
            "--state-spatial-cost-weight",
            spatial_weight,
            "--state-gene-cost-weight",
            gene_weight,
            "--lambda-state-ot",
            "1.0",
            "--lambda-spatial-ot",
            spatial_ot,
            "--lambda-rollout-spatial-ot",
            "0.0",
            "--lambda-rollout-mass-global",
            "0.0",
            "--lambda-mass-global",
            "0.0",
            "--lambda-mass-local",
            "0.0",
            "--lambda-spatial-deform",
            "0.005",
            "--spatial-deform-neighbors",
            "8",
            "--lambda-action",
            "0.001",
            "--lambda-expr",
            "0.0",
            "--lr",
            "0.0005",
            "--seed",
            "19491003",
        ]
        if use_type_prior:
            args.insert(0, "--use-cell-type-prior")
        if potential_z_only:
            args.append("--potential-z-only")
        return tuple(args)

    return [
        Variant(
            name="r14_notype_nomass_state6040_spatial020",
            hypothesis=(
                "Classifier-based labels remove the need for KNN-type matching during evaluation. Test whether "
                "dropping the cell-type prior gives a less scattered, more natural silhouette."
            ),
            args=base(
                spatial_weight="0.60",
                gene_weight="0.40",
                spatial_ot="0.20",
                use_type_prior=False,
            ),
        ),
        Variant(
            name="r14_typeprior_nomass_state7030_spatial020",
            hypothesis=(
                "Keep a weak type prior but move closer to the original stVCR-like 70/30 state matching; "
                "external mass losses remain off."
            ),
            args=base(
                spatial_weight="0.70",
                gene_weight="0.30",
                spatial_ot="0.20",
                use_type_prior=True,
            ),
        ),
        Variant(
            name="r14_zt_nomass_state6040_spatial020",
            hypothesis=(
                "Ablate U(s,z,t) versus U(z,t): keep the same spatial vector field and matching but make gene "
                "potential depend only on latent state and time."
            ),
            args=base(
                spatial_weight="0.60",
                gene_weight="0.40",
                spatial_ot="0.20",
                use_type_prior=False,
                potential_z_only=True,
            ),
        ),
    ]


def variants(preset: str) -> list[Variant]:
    if preset == "r14":
        return r14_variants()
    if preset == "r13":
        return r13_variants()
    if preset == "r12":
        return r12_variants()
    if preset == "r11":
        return r11_variants()
    if preset == "r10":
        return r10_variants()
    if preset == "r09":
        return r09_variants()
    if preset == "r08":
        return r08_variants()
    if preset == "r07":
        return r07_variants()
    if preset == "r06":
        return r06_variants()
    if preset == "r05":
        return r05_variants()
    if preset == "r04":
        return r04_variants()
    if preset == "r03":
        return r03_variants()
    if preset == "r02":
        return r02_variants()
    return r01_variants()


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit_code] {proc.returncode}\n[seconds] {time.time() - started:.3f}\n")
    return int(proc.returncode)


def _summarize(run_dir: Path, variant: Variant, exit_code: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant.name,
        "hypothesis": variant.hypothesis,
        "run_dir": str(run_dir),
        "exit_code": int(exit_code),
        "status": "DONE" if exit_code == 0 and (run_dir / "metrics.csv").exists() else "FAILED",
    }
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        scored = metrics.iloc[1:].copy() if len(metrics) > 1 else metrics.copy()
        final = metrics.iloc[-1]
        row.update(
            {
                "mean_spatial_grid_iou": float(scored["spatial_grid_iou"].mean()),
                "final_spatial_grid_iou": float(final["spatial_grid_iou"]),
                "mean_centroid": float(scored["centroid_mean"].mean()),
                "final_centroid": float(final["centroid_mean"]),
                "mean_chamfer": float(scored["spatial_chamfer_norm"].mean()),
                "final_chamfer": float(final["spatial_chamfer_norm"]),
                "final_label_prop_corr": float(final["label_prop_corr"]),
                "final_label_prop_l1": float(final["label_prop_l1"]),
                "final_compare_png": str(final["compare_png"]),
            }
        )
    mass_path = run_dir / "mass_diagnostics.csv"
    if mass_path.exists():
        mass = pd.read_csv(mass_path)
        scored_mass = mass.iloc[1:].copy() if len(mass) > 1 else mass.copy()
        err = (scored_mass["pred_mean_weight"] - scored_mass["expected_ratio_from_initial"]).abs()
        row["mean_mass_ratio_abs_error"] = float(err.mean())
        row["final_pred_mean_weight"] = float(mass.iloc[-1]["pred_mean_weight"])
        row["final_expected_ratio"] = float(mass.iloc[-1]["expected_ratio_from_initial"])
    return row


def _make_montage(rows: list[dict[str, Any]], out_path: Path) -> None:
    panels: list[Image.Image] = []
    labels: list[str] = []
    for row in rows:
        image_path = row.get("final_compare_png")
        if row.get("status") != "DONE" or not image_path or not Path(str(image_path)).exists():
            continue
        panels.append(Image.open(str(image_path)).convert("RGB").resize((760, 380)))
        labels.append(
            f"{row['variant']} | IoU {float(row.get('final_spatial_grid_iou', np.nan)):.3f} | "
            f"cent {float(row.get('final_centroid', np.nan)):.3f}"
        )
    if not panels:
        return
    width = 1520
    height = 440 * int(np.ceil(len(panels) / 2))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        x = (idx % 2) * 760
        y = (idx // 2) * 440 + 42
        draw.text((x + 8, y - 26), label, fill=(0, 0, 0))
        canvas.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_report(round_dir: Path, rows: list[dict[str, Any]], montage_path: Path) -> Path:
    done = [r for r in rows if r.get("status") == "DONE"]
    ranked_iou = sorted(done, key=lambda r: -float(r.get("final_spatial_grid_iou", -np.inf)))
    ranked_centroid = sorted(done, key=lambda r: float(r.get("final_centroid", np.inf)))
    ranked_mass = sorted(done, key=lambda r: float(r.get("mean_mass_ratio_abs_error", np.inf)))
    lines = [
        "# SpaPOT loss search round",
        "",
        "This round is visual-first: metrics are recorded, but final choice must inspect the montage and per-time comparison images.",
        "",
        "## Variants",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['variant']}`: {row['hypothesis']}")
    lines.extend(["", "## Metric summary", ""])
    cols = [
        "variant",
        "final_spatial_grid_iou",
        "final_centroid",
        "final_chamfer",
        "mean_mass_ratio_abs_error",
        "final_label_prop_l1",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Current ranking", ""])
    if ranked_iou:
        lines.append(f"- Best final IoU: `{ranked_iou[0]['variant']}`.")
    if ranked_centroid:
        lines.append(f"- Best final centroid: `{ranked_centroid[0]['variant']}`.")
    if ranked_mass:
        lines.append(f"- Best mass ratio: `{ranked_mass[0]['variant']}`.")
    if montage_path.exists():
        lines.extend(["", "## Montage", "", f"![final montage]({montage_path})"])
    lines.extend(
        [
            "",
            "## Visual inspection notes",
            "",
            "- Fill this after opening the montage: compare whether the predicted silhouette is too scattered, too collapsed, or shifted.",
            "- Prefer the method whose final and intermediate shapes look coherent, even if grid IoU is only slightly lower.",
        ]
    )
    report = round_dir / "ROUND_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    round_dir = (args.out_root / args.round_name).expanduser().resolve()
    round_dir.mkdir(parents=True, exist_ok=True)
    preset = args.preset
    if preset is None:
        if str(args.round_name).startswith("round14"):
            preset = "r14"
        elif str(args.round_name).startswith("round13"):
            preset = "r13"
        elif str(args.round_name).startswith("round12"):
            preset = "r12"
        elif str(args.round_name).startswith("round11"):
            preset = "r11"
        elif str(args.round_name).startswith("round10"):
            preset = "r10"
        elif str(args.round_name).startswith("round09"):
            preset = "r09"
        elif str(args.round_name).startswith("round08"):
            preset = "r08"
        elif str(args.round_name).startswith("round07"):
            preset = "r07"
        elif str(args.round_name).startswith("round06"):
            preset = "r06"
        elif str(args.round_name).startswith("round05"):
            preset = "r05"
        elif str(args.round_name).startswith("round04"):
            preset = "r04"
        elif str(args.round_name).startswith("round03"):
            preset = "r03"
        elif str(args.round_name).startswith("round02"):
            preset = "r02"
        else:
            preset = "r01"
    selected = variants(preset)
    if args.only:
        names = set(args.only)
        selected = [v for v in selected if v.name in names]
        missing = names - {v.name for v in selected}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    rows = []
    for variant in selected:
        run_dir = round_dir / variant.name
        if args.skip_existing and (run_dir / "metrics.csv").exists():
            exit_code = 0
        else:
            cmd = [
                sys.executable,
                str(RUNNER),
                "--input-h5ad",
                str(args.input_h5ad.expanduser().resolve()),
                "--network-tsv",
                str(DEFAULT_NETWORK),
                "--gat-out-dir",
                str(DEFAULT_GAT_DIR),
                "--run-dir",
                str(run_dir),
                "--use-precomputed-embedding",
                "--spatial-key",
                args.spatial_key,
                "--latent-key",
                args.latent_key,
                "--time-key",
                args.time_key,
                "--raw-time-key",
                args.raw_time_key,
                "--annotation-key",
                args.annotation_key,
                "--expression-layer-key",
                args.expression_layer_key,
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--sample-size",
                str(args.sample_size),
                "--steps-per-interval",
                str(args.steps_per_interval),
                *variant.args,
            ]
            exit_code = _run(cmd, round_dir / "logs" / f"{variant.name}.log")
        row = _summarize(run_dir, variant, exit_code)
        rows.append(row)
        pd.DataFrame(rows).to_csv(round_dir / "round_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(round_dir / "round_summary.csv", index=False)
    (round_dir / "round_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    montage_path = round_dir / "final_visual_montage.png"
    _make_montage(rows, montage_path)
    report = _write_report(round_dir, rows, montage_path)
    print(
        json.dumps(
            {
                "status": "DONE",
                "round_dir": str(round_dir),
                "summary_csv": str(round_dir / "round_summary.csv"),
                "montage": str(montage_path) if montage_path.exists() else None,
                "report": str(report),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
