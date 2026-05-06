#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render large-point real/pred comparison for every predicted frame in one run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--annotation-key", default="cell_type")
    parser.add_argument("--spatial-key", default="X_spatial_input")
    parser.add_argument("--pred-spatial-key", default="X_spatial_aligned")
    parser.add_argument("--raw-time-key", default="time")
    parser.add_argument("--point-size", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _time_sort_key(path: Path) -> float:
    match = re.search(r"predict_(.+)\.h5ad$", path.name)
    if not match:
        return float("inf")
    return float(match.group(1).replace("p", "."))


def _palette(categories: list[str]) -> dict[str, object]:
    fixed = {
        "type_1": "#0072B2",
        "type_2": "#D55E00",
        "type_3": "#E41A1C",
    }
    cmap = plt.get_cmap("tab20")
    return {cat: fixed.get(cat, cmap(i % 20)) for i, cat in enumerate(categories)}


def _scatter_panel(
    ax,
    xy: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    colors: dict[str, object],
    *,
    point_size: float,
) -> None:
    for category in categories:
        mask = labels == category
        if mask.any():
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=point_size,
                c=[colors[category]],
                linewidths=0,
                alpha=1.0,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    predictions = sorted((run_dir / "predictions").glob("predict_*.h5ad"), key=_time_sort_key)
    if not predictions:
        raise FileNotFoundError(f"No predict_*.h5ad under {run_dir / 'predictions'}")

    observed = ad.read_h5ad(args.input_h5ad.expanduser().resolve())
    observed_labels = observed.obs[args.annotation_key].astype(str).to_numpy()
    categories = sorted(set(observed_labels))
    colors = _palette(categories)

    n_frames = len(predictions)
    fig, axes = plt.subplots(n_frames, 2, figsize=(9.0, 3.9 * n_frames), squeeze=False)
    fig.suptitle(run_dir.name, fontsize=14)
    for row_idx, pred_path in enumerate(predictions):
        pred = ad.read_h5ad(pred_path)
        target_time = float(pred.obs[args.raw_time_key].iloc[0])
        real_mask = np.isclose(observed.obs[args.raw_time_key].astype(float).to_numpy(), target_time)
        real = observed[real_mask].copy()
        real_xy = np.asarray(real.obsm[args.spatial_key], dtype=np.float32)
        pred_xy = np.asarray(pred.obsm[args.pred_spatial_key], dtype=np.float32)
        real_labels = real.obs[args.annotation_key].astype(str).to_numpy()
        pred_labels = pred.obs[args.annotation_key].astype(str).to_numpy()
        categories_this = sorted(set(real_labels) | set(pred_labels))
        for category in categories_this:
            if category not in colors:
                colors[category] = plt.get_cmap("tab20")(len(colors) % 20)

        union = np.vstack([real_xy, pred_xy])
        xpad = 0.05 * max(float(np.ptp(union[:, 0])), 1e-6)
        ypad = 0.05 * max(float(np.ptp(union[:, 1])), 1e-6)
        for ax, xy, labels, title in [
            (axes[row_idx, 0], real_xy, real_labels, f"real t={target_time:g}"),
            (axes[row_idx, 1], pred_xy, pred_labels, f"pred t={target_time:g}"),
        ]:
            _scatter_panel(ax, xy, labels, categories_this, colors, point_size=args.point_size)
            ax.set_xlim(union[:, 0].min() - xpad, union[:, 0].max() + xpad)
            ax.set_ylim(union[:, 1].min() - ypad, union[:, 1].max() + ypad)
            ax.set_title(title)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = args.out
    if out is None:
        out = run_dir / "all_frames_real_vs_pred_clear.png"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
