# Validation and benchmarks

Run the full suite with `python -m pytest --cov=pydoppler3d --cov-report=term-missing`.
It includes independent analytic and Monte Carlo checks of the mathematical
contracts, IO, CLI failures, documentation, and plotting coordinates. Coverage
includes visualization; it is not a proxy for scientific accuracy.

After installing the checkout, run:

```bash
python -m benchmarks.validate_pipeline --output docs/validation_results.json
```

The simulation uses independently integrated Gaussian profiles, a compact
component plus a ring, and dense exposure integration. It compares Gaussian
and squeezed defaults, a planar source, phase gaps, unmodeled modulation, wrong
instrumental resolution, and a 2D projected-velocity baseline. It records residual
statistics, comparison with noiseless truth, vertical moments, status, runtime,
versions, seed, and source hashes. The cache benchmark records setup cost,
repeated forward/adjoint cost, and memory separately. Times are machine-specific.
The narrow single-source analytic regression in the test suite starts from a
uniform map, not a blurred truth image.

Interpret structural changes together with fit quality; similar predicted spectra
can correspond to different volumes. These simulations do not calibrate feature
significance or prove unbiased recovery on real accretion flows. The V834 example
also depends on a continuum-derived noise estimate and incomplete physical
emissivity/response information. Its final statistic is diagnostic, not a pass/fail
criterion supplied by the dataset authors.

For an external comparison, run identical synthetic and real spectra through
[trm-doppler](https://github.com/trmrsh/trm-doppler) after converting true axes to
its projected conventions. Match exposure integration, spectral sampling, LSF,
noise model, masks, and stopping criterion. Compare predicted spectra and
recoverable feature position/flux/width, prior sensitivity, runtime, and memory.
That separate native/MEMSYS toolchain is not a dependency of this package; the
checked-in results explicitly state that a live external solver comparison has
not been performed. A direct alpha-to-alpha comparison would be inappropriate
because entropy normalization and stopping formulations differ.

CI tests Python 3.10–3.14, a macOS job, declared minimum core dependencies, optional
plot/FITS extras, and a wheel installed outside the checkout. The local validation
record is not evidence that remote GitHub jobs have already run.

## Recorded local run

Python 3.14.8, NumPy 2.5.4, SciPy 1.17.1; seed 901.
The following are diagnostics from the recorded simulation, including unfinished
fits. Numerical convergence alone does not establish an acceptable noise fit.

| Case | Mean chi-square | Termination | Vertical RMS (km/s) |
|---|---:|---|---:|
| off_plane | 1.099 | iteration_limit | 84.7 |
| gaussian_default | 1.087 | iteration_limit | 91.5 |
| planar_source | 1.070 | iteration_limit | 67.9 |
| phase_gaps | 1.105 | iteration_limit | 84.4 |
| modulated_misspecification | 2.008 | converged | 111.6 |
| wrong_resolution | 0.864 | iteration_limit | 81.0 |
| 2d_projected_baseline | 11.934 | converged | undefined |

Caching reduced the repeated forward/adjoint benchmark from 0.0685 s to 0.0137 s
(4.99 times faster), using 1,776,864 bytes.
This is an operator benchmark, not a claim of the same speedup for every full fit.

The planar simulation acquires nonzero vertical width, and the wrong-resolution
model still fits the spectra well. These are reasons to test assumptions, not
claims of successful 3D recovery. [Machine-readable results and source hashes](validation_results.json).

The bundled V834 example reports `default_not_converged` and mean chi-square
approximately 2.648 after the configured two default stages. The gallery and
bundled provenance record that status explicitly. The full local suite passed
79 tests with 82% branch-aware coverage, including plotting; Ruff lint/format
and distribution builds passed. Remote CI and the external reference-code
comparison remain unverified.
