# PyDoppler3D

PyDoppler3D is a new, separate prototype repository for three-dimensional
Doppler tomography of phase-resolved emission-line profiles. It is inspired by
Marsh (2022), *Three dimensional Doppler tomography*, MNRAS, 510, 1340,
https://doi.org/10.1093/mnras/stab3335, and by the existing two-dimensional
PyDoppler workflow.

This repository is intentionally honest about its status: it contains the core
geometry, a forward projector, an adjoint/backprojection operator, 3D default-map
helpers, tests, CI, and a toy nonnegative iterative reconstructor. It is not yet
a full replacement for the maximum-entropy 3D implementation described by Marsh
or for `trm-doppler`.

## Physical Conventions

Velocity coordinates are `(vx, vy, vz)` in km/s in the binary co-rotating frame:

- `x` points from star 1 toward star 2;
- `y` points in the direction of motion of star 2;
- `z` is parallel to the orbital angular-momentum axis.

For orbital phase `phi` in cycles and inclination `i`, the line-of-sight velocity
used by the projector is

```text
V = gamma + sin(i) * [-vx cos(2 pi phi) + vy sin(2 pi phi)] + vz cos(i)
```

At `i = 90 deg`, the expression reduces to the usual 2D Doppler-tomography
projection. The `vz` dimension therefore encodes out-of-plane motion, but the
paper shows why this is much less constrained than the 2D map: the profile data
only constrain a double-cone surface in Fourier space. Treat 3D features as
model-dependent unless simulations, defaults, and residuals support them.

## What Exists Now

- `VelocityGrid3D` for regular velocity cubes.
- Relativistic wavelength-to-velocity helpers.
- `project_cube`, a 3D image-to-trailed-spectrum forward operator.
- `back_project`, the matching adjoint-style operator for residuals.
- Isotropic Gaussian and Marsh-style squeezed defaults.
- `landweber_reconstruct`, a toy positive iterative reconstructor useful for
  smoke tests and method development, not publication-grade inference.
- Unit tests and GitHub Actions for Python 3.10 through 3.14.

## What Still Needs Real Scientific Work

The state-of-the-art path should include:

1. A true maximum-entropy objective and optimizer, ideally with a verified
   forward/transpose pair.
2. FITS input/output compatible with multiple data sets, flux/error arrays,
   wavelengths, phases, and exposure widths.
3. Instrumental and temporal smearing kernels.
4. Benchmarks against the simulations in Marsh (2022): spot, cube, disc, and
   polar-like tilted stream tests.
5. Strong visualization tools for projections, slices, and residual trails.
6. A careful uncertainty and artefact analysis, because 3D maps are intrinsically
   underconstrained relative to 2D maps.

## Quick Start

```python
import numpy as np

from pydoppler3d import VelocityGrid3D, project_cube

velocities = np.linspace(-2500.0, 2500.0, 401)
phases = np.linspace(0.0, 1.0, 80, endpoint=False)
grid = VelocityGrid3D.regular(vlim_xy=1500.0, nxy=61, vlim_z=1000.0, nz=41)

vx, vy, vz = grid.mesh()
cube = np.exp(-0.5 * (((vx + 500.0) / 120.0) ** 2
                      + ((vy - 700.0) / 120.0) ** 2
                      + ((vz - 300.0) / 180.0) ** 2))

profiles = project_cube(cube, grid, phases, velocities, inclination_deg=75.0)
```

Run the included smoke example:

```bash
python sample_script.py
```

Run tests:

```bash
python -m pytest -q
```

## Relationship To Existing Code

This is a separate repository from `/Users/francesco/pydoppler`. The older
project remains a lightweight wrapper around Spruit's classic 2D Fortran code.
This project starts from the 3D geometry and algorithmic ideas in Marsh (2022),
with Python-first kernels that can later be optimized or replaced by compiled
operators.
