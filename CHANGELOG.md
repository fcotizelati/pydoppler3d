# Changelog

## 0.2.0 — unreleased

Breaking scientific changes:

- Adopt right-handed velocity axes and the `-Vz cos(i)` radial-velocity term.
  Old 0.1 maps have the opposite vertical orientation; see the migration guide.
- Reconstruction requires an explicit inclination. Spectra retain signed flux;
  internal nondimensional scaling makes the floor and entropy weight independent
  of input flux units. Previous alpha values are not directly comparable.
- The loader propagates continuum/interpolation covariance by default and rejects
  invalid errors or incomplete exposure widths instead of inventing values.
- CLI fits write diagnostics and exit with code 2 for incomplete results unless
  the user explicitly supplies `--allow-incomplete`.

Fixes and additions:

- Matched forward/adjoint endpoints, Gaussian LSF with padded support, and exposure
  quadrature; scaled/modulated components integrate consistently over exposure.
- Bounded geometry caching and reuse of objective evaluations in callbacks.
- Strict iteration budgets, immediate discrepancy stopping, stage-aware histories,
  optimizer/default convergence status, and full-covariance likelihood support.
- Stable narrow squeezed defaults, regular-grid validation, correct plot extents,
  common observation/model color scales, and centered standardized residuals.
- Versioned pickle-free NPZ/FITS metadata, masks, covariance, and map provenance.
- Executable examples, independent simulations, scientific regression tests,
  optional-dependency/minimum-dependency/macOS/installed-wheel CI.

The software remains experimental. No unique 3D recovery or publication-level
uncertainty calibration is implied by these corrections.
