# Model Provenance

This repository fixes SpaPOT to the model family that generated the strongest
current axolotl trajectory figures, rather than treating the figure renderer or a
dataset-specific example as the model.

## Reference Figure Result

The paper-style trajectory GIF was rendered from:

```text
result_figures/axolotl_artista_injured_adjacent_aligned_direct_ae_hybrid_hj_compare/nohj_lam0_e1000_s1024_samplegrowth
```

The render report identifies the same run directory, the
`trajectory_visual_concepts_step0p2_paper_model_allsource` trajectory cache, 91
frames from 2.0 to 20.0, and `inj_uninj_counts = {"inj": 22820}`. The renderer
did not define the model.

## Reference Model

The run summary identifies:

```text
model_family: SpaPOT Hybrid
model_path: save_model/hybrid_model.pth
velocity_parameterization: hybrid
lambda_hjb: 0.0
n_epochs: 1000
num_samples: [1204, 1204, 1204, 1204, 1204]
seed: 19491001
lambda_match: 400000.0
alpha_exp: 0.01
alpha_gro: 0.0002
kappa_exp: 0.02
kappa_gro: 0.1
ode_method: rk4
ode_step_size: 0.25
use_alignment: false
```

The reference checkpoint uses `velocity_parameterization = "hybrid"`,
`spatial_dim = 2`, `gene_dim = 10`, and `hidden_dim = 128`.

Observed checkpoint parameter shapes:

```text
spatial_velocity_net:
  hidden linears: (128, 13), then five (128, 128), output (2, 128)
gene_velocity_net:
  hidden linears: (128, 13), then five (128, 128), scalar output (1, 128)
growth_rate_net:
  hidden linears: (128, 13), (128, 128), (128, 128), output (1, 128)
```

## Fixed Model Family

SpaPOT therefore uses this fixed model family:

```text
state = [spatial, latent]
ds/dt = spatial MLP([t, state])
dz/dt = -grad_z U([t, state])
dlogw/dt = growth MLP([t, state])
```

Architecture details:

```text
spatial branch: n_hidden MLP layers, output spatial_dim
gene branch: n_hidden scalar-potential MLP layers, output -grad_z U
growth branch: fixed 3 hidden MLP layers, output 1
```

The fixed 3-layer growth branch is important: the reference SpaPOT Hybrid
checkpoint used three hidden layers for growth even when the spatial and gene
branches used `n_hiddens = 6`.

## Training Objective

The reference objective is not endpoint mean plus positive action regularization.
It is full forward/backward time-grid matching:

```text
forward rollout: first observed slice -> all observed time points
backward rollout: last observed slice -> all observed time points
loss = lambda_match * sum(OT + growth-ratio penalty)
       + signed forward/backward WFR action
       + optional HJ regularizer
```

Feature scaling is off by default, so training uses the raw concatenated
`[spatial, latent]` state. The label readout in SpaPOT Hybrid mode is the
spatiotemporal classifier on `[spatial, latent, time]`.

SpaPOT Hybrid training also preserves non-architectural details that affect the learned
trajectory: Python `random.sample` slice sampling under the configured seed,
Adam with weight decay `1e-5`, no gradient clipping by default, fixed-step RK4
with `ode_step_size = 0.25`, and sample-size growth by +20 every 100 epochs.

## Equivalence Audit

The optional audit script below loads a reference Hybrid checkpoint, maps its
weights into the SpaPOT Hybrid model, and compares `ds/dt`, `dz/dt`, and
`dlogw/dt` on random probe states:

```bash
python tools/audit_spapot_hybrid_equivalence.py \
  --reference-src path/to/reference/src \
  --checkpoint path/to/reference_hybrid_model.pth
```

For the reference checkpoint above, the observed max absolute velocity and
growth differences are both `0.0` after weight mapping.
