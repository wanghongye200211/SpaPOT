# SpaPOT

**SpaPOT** stands for **Spatial Potential Optimal Transport**.

This repository keeps only the model implementation.

Core dynamics:

```text
state x = [s, z]

ds/dt      = spatial_net(s, z, t)
dz/dt      = -grad_z U(s, z, t)
d log w/dt = growth_net(s, z, t)
```

Install:

```bash
pip install -e .
```

Package:

```text
src/spapot/
```

