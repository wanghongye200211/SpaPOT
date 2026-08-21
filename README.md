# stCTD

**stCTD** is the **Spatiotemporal Critical Transition Decipherer**, a generative
deep-learning framework for reconstructing continuous single-cell or spot-level
spatiotemporal population dynamics from discrete time-series spatial
transcriptomic snapshots.

stCTD jointly models molecular-potential reorganization, spatial transport, and
source-sink population growth. Along the reconstructed trajectory, it can
retrospectively localize transition-sensitive intervals, resolve the populations
contributing to those signals, and support in silico perturbation of candidate
gene programs.

- Code repository: <https://github.com/wanghongye200211/stCTD>
- Interactive project website: <https://wanghongye200211.github.io/stCTDcover/>
- Website source: <https://github.com/wanghongye200211/stCTDcover>

## Model

Each cell or spot is represented by a joint state `x = [s, z]`, where `s` is a
two-dimensional tissue coordinate and `z` is a graph-informed molecular latent
state. stCTD evaluates three coupled, time-dependent neural fields:

```text
ds/dt      = v_theta(s, z, t)
dz/dt      = -grad_z Phi_theta(s, z, t)
d log w/dt = g_theta(s, z, t)
```

Here, `v_theta` controls spatial redistribution, `Phi_theta` defines the
molecular-potential landscape, `g_theta` controls source-sink population change,
and `w` is the particle mass. These dynamics induce the population equation:

```text
partial_t rho + div_s(rho v_theta)
              - div_z(rho grad_z Phi_theta) = g_theta rho
```

The implementation uses six hidden layers of width 128 for the spatial and
molecular-potential branches, a three-hidden-layer growth branch, ReLU
activations, bidirectional fitting across the observed time series, Adam
optimization, and fixed-step RK4 integration by default. Spatial and molecular
features are standardized separately, and the standardized spatial component is
weighted by `3.0` in the default data configuration. Analysis-specific loss
weights should be chosen for each dataset rather than treated as universal.

## Installation

```bash
git clone https://github.com/wanghongye200211/stCTD.git
cd stCTD
pip install -e .
```

Python 3.9 or later is required.

## Minimal API

```python
from stctd import ModelConfig, STCTDModel

model = STCTDModel(ModelConfig())
```

Training and evaluation helpers are available from submodules:

```python
from stctd.train import train_stctd_model
from stctd.evaluate import evaluate_stctd_model
```

## Input Data

The training entry point expects an AnnData file containing the following
fields unless alternative keys are supplied:

| AnnData field | Default key | Meaning |
| --- | --- | --- |
| `obsm` | `X_spatial_input` | Two-dimensional aligned tissue coordinates |
| `obsm` | `X_gene_input` | Graph-informed molecular latent state |
| `obs` | `time_input` | Model time used for integration |
| `obs` | `time` | Original experimental time |
| `obs` | `Annotation` | Cell or spot annotation |
| `layers` | `lognorm` | Normalized expression used for downstream readouts |

The model reconstructs population-level dynamics between independently sampled
specimens; it does not claim to track the same physical cells across time.

## Training

```bash
python examples/train_stctd.py \
  --input-h5ad path/to/prepared_spatial_latent.h5ad \
  --out-dir runs/stctd \
  --device auto
```

The default `stctd_fullgrid` objective propagates particles forward from the
earliest observed stage and backward from the latest observed stage, matching
both trajectories to all sampled stages. Optional endpoint, spatial-coherence,
cell-type-prior, and first-order Hamilton-Jacobi terms are exposed for controlled
ablation or analysis-specific configurations.

Useful options include:

```text
--spatial-weight 3.0
--scale-features / --no-scale-features
--lambda-hj FLOAT
--velocity-parameterization potential|vector
--require-obs-value KEY VALUE
```

`potential` is the manuscript model. `vector` is retained only as a direct
gene-velocity ablation and cannot use the Hamilton-Jacobi loss.

## Outputs

Each run writes:

```text
config.json
summary.json
stctd_vs_baseline_metrics.csv
checkpoints/model.pt
checkpoints/training_trace.jsonl
predicted/*.h5ad
comparisons/*.png
```

## Repository Layout

```text
src/stctd/                 Core model, training, integration, and evaluation
src/embedding/             Graph-informed molecular embedding helpers
examples/train_stctd.py    Dataset-neutral training entry point
tests/test_stctd_model.py  Architecture and behavior regression tests
docs/model_provenance.md   Checkpoint and figure provenance notes
tools/audit_stctd_equivalence.py
                           Optional checkpoint-level equivalence audit
```

The repository is source-only by design. Data matrices, trained checkpoints,
intermediate AnnData files, and generated manuscript figures are not bundled.

## Interpretation Boundary

The current transition score is evaluated along a trajectory reconstructed from
the full observed time series. It therefore supports retrospective localization
of transition-sensitive intervals. Strict prospective early warning requires a
separate past-only training and held-out future validation design.
