# SpaPOT

**SpaPOT** stands for **Spatial Potential Optimal Transport**.

This repository contains the current working version of SpaPOT, a visual-first spatial transcriptomic trajectory model that centers gene potential dynamics while using spatial transport and OT matching as the training framework.

The selected baseline is:

```text
r14_typeprior_nomass_state7030_spatial020
```

It keeps adjacent-time OT matching as the stable backbone, uses a spatial vector field for spatial motion, uses a gene potential field for latent gene dynamics, and evaluates predicted cell types with a classifier trained on latent `z` rather than KNN labels. Graph AE / GAT can be used to construct the gene latent space, but it is an encoder module rather than the main identity of the method.

## Model

The state is:

```text
x = [s, z]
```

where `s` is the scaled 2D spatial coordinate and `z` is the gene latent state.

The neural dynamics are:

```text
ds/dt      = spatial_net(s, z, t)
dz/dt      = -grad_z U(s, z, t)
d log w/dt = growth_net(s, z, t)
```

The current preferred training setting is deliberately simple:

```text
state OT matching:        1.0
state spatial/gene cost:  0.70 / 0.30
extra spatial OT:         0.20
external global mass:     0.0
external local mass:      0.0
rollout spatial OT:       0.0
expression decoder loss:  0.0
spatial deformation:      0.005
action regularization:    0.001
cell-type prior:          on, weak
```

Mass/growth is not used as a strong external loss in this version. It remains available through learned weights in transport and is reported as a diagnostic.

## Repository Layout

```text
src/spapot/      main SpaPOT model implementation
src/stvcr_jointae/          minimal GAT-AE preprocessing/decoder helpers
scripts/                   training, loss-search, relabeling, visualization
examples/                  runnable command templates
results/r14_classifier_nomass_hybrid/
                            lightweight result summary and selected figures
```

Large inputs, checkpoints, intermediate `.h5ad` files, and full run directories are intentionally excluded from Git.

## Installation

```bash
conda create -n spapot python=3.10
conda activate spapot
pip install -r requirements.txt
pip install -e .
```

PyTorch installation depends on the machine. On Apple Silicon, install a PyTorch build with MPS support.

## Data Contract

The training script expects an AnnData file with:

```text
obsm["X_spatial_input"]   aligned 2D spatial coordinates
obsm["X_gene_input"]      gene latent embedding, e.g. GAT-AE latent
obs["time_input"]         normalized time
obs["time"]               raw time for labels/plots
obs["cell_type"]          cell-type annotation
X or layers["X"]          expression matrix
```

Place a prepared input here:

```text
data/prepared_input.h5ad
```

Optionally place a gene network here if rebuilding the GAT embedding:

```text
references/network_mouse.tsv
```

## Run the Current Best Variant

```bash
bash examples/run_r14_best.sh
```

Equivalent direct command:

```bash
python scripts/run_spapot_loss_search_round.py \
  --preset r14 \
  --round-name round14_classifier_nomass_hybrid \
  --only r14_typeprior_nomass_state7030_spatial020 \
  --device mps \
  --epochs 120 \
  --sample-size 384 \
  --steps-per-interval 8
```

The script writes outputs under:

```text
runs/spapot_loss_search/round14_classifier_nomass_hybrid/
```

## Current Result Snapshot

The selected r14 result is included as a lightweight snapshot:

```text
results/r14_classifier_nomass_hybrid/round_summary.csv
results/r14_classifier_nomass_hybrid/ROUND_REPORT.md
results/r14_classifier_nomass_hybrid/r14_typeprior_nomass_state7030_spatial020/all_frames_classifier_largepoints.png
```

In the r14 comparison, the selected model was the best visual tradeoff:

| variant | final IoU | final centroid | label L1 | visual note |
|---|---:|---:|---:|---|
| no type prior, 60/40 | 0.0295 | 0.1795 | 0.1554 | scattered |
| weak type prior, 70/30 | 0.0337 | 0.0863 | 0.1455 | selected |
| U(z,t), no mass | 0.0346 | 0.6177 | 0.3901 | metric trap |

The `U(z,t)` control has slightly higher final IoU, but the large-point visual check shows strong type mixing and global drift. The selected model is therefore based on visual coherence, not IoU alone.
