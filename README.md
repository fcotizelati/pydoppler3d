# PyDoppler3D

[![CI](https://github.com/fcotizelati/pydoppler3d/actions/workflows/ci.yml/badge.svg)](https://github.com/fcotizelati/pydoppler3d/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.14-blue)

Experimental three-dimensional Doppler tomography of phase-resolved emission-line
spectra, implemented in Python with NumPy and SciPy. Reconstruct a nonnegative
velocity cube, inspect its predicted spectra, and test sensitivity to assumptions.

**A good spectral fit does not uniquely determine a 3D flow.** Orbital spectra
sample only part of the information needed to reconstruct a volume. Noise,
phase coverage, systemic velocity, inclination, emissivity variations, and the
default map can all affect apparent vertical structure. This is an experimental
MEM-style implementation, not a validated replacement for MEMSYS/trm-doppler.
See [scientific conventions and limitations](docs/scientific_conventions.md).

Version 0.2 changes the vertical sign convention and flux scaling, requires an
explicit inclination, propagates preprocessing covariance, and reports incomplete
fits. **Existing 0.1 maps need migration.** See [changes](CHANGELOG.md) and the
[migration guide](docs/scientific_conventions.md#migration-from-01).

## Installation

Python 3.10 or newer is required. From a cloned repository:

```bash
python -m pip install -e ".[plot,volume,fits]"
```

The core requires NumPy and SciPy; Matplotlib, Plotly, and Astropy are optional
plotting, interactive-volume, and FITS extras. No custom compiled extension or
Fortran compiler is required. Development setup is in [CONTRIBUTING.md](CONTRIBUTING.md).

## One complete example

This example uses the bundled V834 Cen He II spectra, retains negative noisy
samples, and fits in the original flux units. It is intentionally a small
exploratory run; inspect the returned status before interpreting its map.

```python
from pathlib import Path
from pydoppler3d import (
    DopplerMap,
    MemConfig,
    ProjectionOperator,
    VelocityGrid3D,
    copy_test_data,
    load_v834cen_dataset,
    mem_reconstruct,
    plot_map_slices,
    plot_reconstruction,
)

workdir = Path("pydoppler3d-workdir")
outdir = Path("output_images")
outdir.mkdir(parents=True, exist_ok=True)
copy_test_data(workdir)
spectra = load_v834cen_dataset(workdir / "v834cen").spectra
spectra.to_npz(outdir / "v834cen_heii4686_trails.npz")

grid = VelocityGrid3D.regular(vlim_xy=1800, nxy=21, vlim_z=1200, nz=11)
response = dict(inclination_deg=50.0, gamma=0.0, exposure=spectra.exposure)
result = mem_reconstruct(
    spectra.flux,
    grid,
    spectra.phases,
    spectra.velocity,
    error=spectra.error,
    covariance=spectra.covariance,
    mask=spectra.mask,
    config=MemConfig(iterations=100, default_updates=2),
    **response,
)
print(result.status, result.message)
print("Mean chi2:", result.chi2_history[-1])
# success=False means this is a partial result, not an accepted reconstruction.
doppler_map = DopplerMap(result.image, grid, meta=result.diagnostics())
doppler_map.to_npz(outdir / "Doppler_Map.npz")
model = ProjectionOperator(
    grid,
    spectra.phases,
    spectra.velocity,
    **response,
).forward(result.image)
plot_map_slices(doppler_map, outdir / "Doppler_Map.pdf")
plot_reconstruction(
    spectra,
    model,
    outdir / "Reconstruction.pdf",
    standardized=True,
)
```

`sample_script.py` writes a larger example with covariance, map/trail products,
JSON diagnostics, PDFs, and a standalone interactive HTML volume. See the
[regenerated example gallery](docs/example_gallery.md), including its fit status:

```bash
python sample_script.py --allow-incomplete
```

`--allow-incomplete` explicitly permits exploratory products from an unfinished
fit. Without it, an incomplete fit writes its diagnostics and exits with code 2.
Use `--no-plots` for numerical products only and `--output PATH` to choose a
product directory. The script does not delete previous figures or overwrite
copied native input spectra.

## Input and output contracts

| Quantity | Meaning |
|---|---|
| `phases` | Orbital cycles; zero is inferior conjunction of star 2/donor |
| `velocity` | Increasing spectral coordinates in km/s |
| `flux` | Signed spectral-bin values; observations are never clipped |
| `error` | Positive finite marginal standard deviation on included samples |
| `covariance` | Optional `(nphase,nvelocity,nvelocity)` covariance; profiles independent |
| `mask` | Boolean array matching flux; `True` means included |
| `exposure` | Full exposure width in orbital cycles, centered on each phase |
| cube | Flux per voxel in the discrete input spectral-bin convention |
| cube axes | Regularly spaced true velocity components in km/s |

The text loader accepts `wavelength flux [error]` spectra and a phase file with
`spectrum_name phase [delta_phase]`. It fits a polynomial continuum on each native
spectrum, resamples once, and propagates both operations into covariance. Its
ordinates retain the input numerical units; these sampled profiles are **not an
absolute integrated line-flux calibration**. Convert flux density and bin widths
consistently before interpreting absolute luminosities.

For custom data, call `pydoppler3d.pydoppler_compat.load_pydoppler_dataset` with
explicit `list_file`, `lam0`, `delw`, `gamma`, and `continuum_band` arguments. Windows
are in the observed wavelength frame. The loader removes gamma once; pass
`gamma=0` when projecting those corrected profiles. Missing native errors are
estimated from continuum scatter and labeled as estimates; an unmeasurable noise
level is an error. `propagate_covariance=False` requests a documented diagonal
approximation. NPZ and FITS preserve metadata, covariance, masks, and exposures.

## Mathematical conventions and fit acceptance

The right-handed x axis points from star 1 to star 2, y follows star 2's motion,
and +z follows orbital angular momentum. Inclination is measured from +z toward
the observer; radial velocity is positive for recession:

```text
Vrad = gamma + sin(i) [-Vx cos(2π phase) + Vy sin(2π phase)] - Vz cos(i)
```

The components are inertial barycentric velocities expressed in rotating axes,
not velocities relative to the rotating frame. Standard projected coordinates
are `vx = Vx sin(i)`, `vy = Vy sin(i)`, and `vz = Vz cos(i)`.
At 90 degrees **all Vz structure is unconstrained**; at face-on inclinations the
in-plane structure is unconstrained. The code warns for these exact limits.

The solver minimizes a Gaussian residual term plus generalized relative entropy
in internally normalized units. Returned images preserve input units. `alpha`
and `positivity_floor` refer to internal units. `chi2_history` is chi-square per
retained constraint, **not a fitted-degrees-of-freedom reduced chi-square**.
With singular covariance, only retained eigenmodes contribute; inspect the
[likelihood policy](docs/scientific_conventions.md#likelihood-and-scaling).

Check `status`, `success`, `converged`, `target_reached`, `default_converged`, and
residuals. `target_chi2` is an immediate discrepancy stopping rule; reaching it
does not imply stationarity or solve constrained maximum entropy. Objective
histories must be compared within the same `history_stage`, because default
updates change the objective. An accepted numerical run still needs scientific
validation against noise, default, geometry, and observation-model sensitivity.

Forward and adjoint operators include optional Gaussian `instrumental_fwhm` and
exposure quadrature. Pass the **same response** during fitting and prediction.
Increase `phase_samples` until long-exposure predictions stabilize. Instrumental
broadening requires a uniformly sampled spectral axis. Signed/modulated component
projection utilities are available, but simultaneous modulation inversion is not
implemented.

## Command line

```bash
pydoppler3d-info
pydoppler3d-reconstruct output_images/v834cen_heii4686_trails.npz output_images/cli_map.npz --inclination 50 --iterations 200
pydoppler3d-plot-trails output_images/v834cen_heii4686_trails.npz output_images/Trail.pdf
pydoppler3d-plot-map output_images/cli_map.npz output_images/Map.pdf --vz -600 0 600
pydoppler3d-plot-volume output_images/cli_map.npz output_images/Volume.html
```

The reconstruct command saves a map and JSON diagnostics, including input hash,
configuration, and status. Exit 2 means an incomplete/unaccepted numerical fit;
`--allow-incomplete` explicitly changes that exit policy. Supply `--fwhm` in km/s
when known, and `--phase-samples` for exposure quadrature. `--diagonal-errors`
explicitly ignores off-diagonal stored covariance. Use `--help` for all options.

## Validation, performance, and development

Run `python -m pytest` for regression tests. They cover response adjoints,
finite-difference gradients, realistic flux scales, covariance propagation,
convergence contracts, independent analytic injection, plotting, and IO.
The [validation workflow](docs/validation.md) includes a 2D baseline, prior
sensitivity, observation-model mismatch, and a bounded geometry-cache benchmark.
Passing tests do not calibrate feature significance or remove 3D nonuniqueness.

Useful references: [scientific guide](docs/scientific_conventions.md),
[validation](docs/validation.md), [contributing](CONTRIBUTING.md),
[release notes](CHANGELOG.md), and [software citation](CITATION.cff).

## Method and data attribution

Please cite the software and relevant method/data sources:

- [Marsh 2022, three-dimensional Doppler tomography](https://doi.org/10.1093/mnras/stab3335).
- [Spruit 1998, maximum-entropy Doppler mapping](https://arxiv.org/abs/astro-ph/9806141).
- Kotze, Potter & McBride 2016, A&A 595, A47: bundled CDS V834 Cen data.
- Echevarría, de la Fuente & Costero 2007, AJ 134, 262: bundled U Gem spectra.
- The original PyDoppler project for its workflow and compatibility data.

Source code is MIT licensed. Third-party research data retain their own provenance
and are not relicensed as package source; see [NOTICE](NOTICE).
