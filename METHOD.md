# SpaPOT Method Notes

## Current Position

SpaPOT should be understood as a potential-centered spatial OT trajectory model. The current implementation still uses adjacent-time matching as the stable training backbone, but the method name is intentionally separated from stVCR because the main new object is the learned potential field.

The most stable visual behavior came from:

- keeping adjacent-time matching as the main training signal;
- representing gene evolution with a potential field;
- using a separate spatial vector field for spatial motion;
- avoiding strong external mass/growth losses;
- using latent `z` classifier labels for evaluation instead of KNN label transfer.

## Selected Variant

```text
r14_typeprior_nomass_state7030_spatial020
```

Core settings:

```text
U(s,z,t) gene potential
ds/dt = spatial_net(s,z,t)
growth enabled, but no strong external mass loss
state matching spatial/gene cost = 0.70 / 0.30
spatial OT = 0.20
weak cell-type prior in matching
latent-z classifier for predicted labels
```

## Why Not Strong Mass Loss?

Earlier variants showed that strong global/local mass terms and late rollout matching can improve some numeric diagnostics while making the visible trajectory less coherent. In the current simulation, mass is treated mainly as a diagnostic and weak internal transport quantity.

## Why Keep `U(s,z,t)`?

The `U(z,t)` r14 control had the highest final IoU, but the all-frame plot showed large centroid drift and severe type mixing. This makes it a metric trap. The selected model keeps spatial conditioning in the potential and relies on the spatial vector field for geometry.

## Evaluation Rule

Do not rank variants by final IoU alone.

Use:

```text
large-point all-frame visualization
final centroid
label proportion L1/correlation from latent-z classifier
spatial IoU/chamfer
mass diagnostics as secondary evidence
```
