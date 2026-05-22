# SpaPOT

**SpaPOT** stands for **Spatial Potential Optimal Transport**.

This repository contains the source implementation of our spatial potential
model for spatial transcriptomic trajectory reconstruction.

The current public model is a hybrid potential dynamics model. Its density-level
form is:

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

Training uses stVCR-like endpoint-to-all-time optimal-transport matching. The
matching cost combines spatial and gene-latent distances, while growth is
regularized by a weak cell-number ratio term. The gene-latent velocity is
constrained by the potential gradient, and spatial motion and growth are learned
as separate neural fields. Optional cell-type grouped matching, spatial velocity
smoothness, and HJ/HJB regularization are available but disabled by default.
Rollout is performed with `torchdiffeq.odeint`; the default method is fixed-step
`rk4` for reproducibility, while adaptive methods such as `dopri5` can be used
for sensitivity checks.

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

Main package:

```text
src/spapot/
```

Embedding helpers:

```text
src/embedding/
```

The repository is source-only by design. Data matrices, trained checkpoints,
intermediate AnnData files, and generated figures are kept outside the public
package.
