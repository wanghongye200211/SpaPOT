# SpaPOT

**SpaPOT** stands for **Spatial Potential Optimal Transport**.

This repository contains the source implementation of our spatial potential
model for spatial transcriptomic trajectory reconstruction.

The current public model is potential based:

```text
state x = [s, z]

ds/dt      = spatial_net(s, z, t)
dz/dt      = -grad_z U(s, z, t)
d log w/dt = growth_net(s, z, t)
```

where `s` is the spatial coordinate, `z` is a gene-expression embedding, and
`w` is the learned mass/abundance weight.

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
