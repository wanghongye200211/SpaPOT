# SpaPOT

**SpaPOT** stands for **Spatial Potential Optimal Transport**.

This repository contains the source implementation of our spatial potential
model for spatial transcriptomic trajectory reconstruction.

The current public model is the SpaPOT Hybrid
spatial-potential-growth field selected from the figure provenance audit. Its
density-level form is:

```text
∂tρ + ∇s · (ρ vθ) - ∇z · (ρ ∇z Φθ) = gθ ρ
```

The particle dynamics used in training and reconstruction are:

```text
state x = [s, z]

ds/dt      = vθ(s, z, t)
dz/dt      = -∇z Φθ(s, z, t)
d log w/dt = gθ(s, z, t)
```

where `s` is the spatial coordinate, `z` is a gene-expression embedding, and
`w` is the learned mass/abundance weight.

Training defaults to the SpaPOT Hybrid full time-grid objective:
one forward rollout from the first slice and one backward rollout from the last
slice are matched to every observed slice with optimal transport. The matching
cost combines spatial and gene-latent distances, growth is regularized by a weak
cell-number ratio term, and the signed WFR/action integral is added in the same
style as the original Hybrid training loop. Feature scaling is disabled by
default so the model trains on the raw concatenated `[spatial, latent]` state.
The model architecture is fixed as the Hybrid branch used by the reference
figures: direct spatial velocity MLP, gene-latent potential-gradient MLP, and a
three-hidden-layer growth MLP. The spatial and gene-potential branches use
`ModelConfig.n_hidden`; the growth branch is intentionally fixed to the original
three hidden layers.
SpaPOT Hybrid sampling uses Python `random.sample` under the configured seed.
The default optimizer is Adam with weight decay
`1e-5`, and gradient clipping is disabled by default.
Evaluation in SpaPOT Hybrid mode uses the spatiotemporal classifier on
`[spatial, latent, time]` rather than the newer latent-only classifier. An
endpoint/action mode remains available for ablation, but it is no longer the
default. Optional cell-type grouped matching, spatial velocity smoothness, and
HJ/HJB regularization are available but disabled by default. Rollout is performed
with `torchdiffeq.odeint`; the default method is fixed-step `rk4` for
reproducibility, while adaptive methods such as `dopri5` can be used for
sensitivity checks.

Install:

```bash
pip install -e .
```

Minimal model use:

```python
from spapot import ModelConfig, SpaPOTPotentialModel

model = SpaPOTPotentialModel(ModelConfig())
```

Training and evaluation helpers are available from submodules:

```python
from spapot.train import train_spapot_model
from spapot.evaluate import evaluate_spapot_model
```

SpaPOT Hybrid training entry point:

```bash
python examples/train_spapot_hybrid.py \
  --input-h5ad path/to/prepared_spatial_latent.h5ad \
  --out-dir runs/spapot_hybrid \
  --device auto
```

This entry point is dataset-neutral. It exposes the model-level SpaPOT Hybrid
configuration: `spapot_fullgrid`, raw unscaled state, `sample_size=1024`,
sample-size growth every 100 epochs, `lambda_match=4e5`, no endpoint
`lambda_action` term, no default gradient clipping, and the original
`[spatial, latent, time]` classifier for
predicted labels. Dataset restrictions are explicit input choices, not hidden
model behavior. For example, to require an input AnnData to contain only injured
cells, add:

```bash
--require-obs-value inj_uninj inj
```

Main package:

```text
src/spapot/
```

Model provenance:

```text
docs/model_provenance.md
```

Optional checkpoint-level equivalence audit:

```bash
python tools/audit_spapot_hybrid_equivalence.py \
  --reference-src path/to/reference/src \
  --checkpoint path/to/reference_hybrid_model.pth
```

Embedding helpers:

```text
src/embedding/
```

The repository is source-only by design. Data matrices, trained checkpoints,
intermediate AnnData files, and generated figures are kept outside the public
package.
