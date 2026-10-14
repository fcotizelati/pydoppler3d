# Scientific conventions and numerical contracts

This document describes version 0.2. It specifies the implemented finite-dimensional
problem and its limits; it is not a proof of unique recovery of an astrophysical flow.

## Coordinates

Use inertial barycentric velocity components in a rotating, right-handed basis.
+x points from star 1 to star 2; +y follows star 2's motion; +z is angular momentum.
Phase zero is inferior conjunction of star 2. Inclination is the angle between
+z and the direction from binary to observer. Positive radial velocity is recession.

`vr = gamma + sin(i) * (-Vx*cos(2*pi*phi) + Vy*sin(2*pi*phi)) - Vz*cos(i)`.

Velocities and instrumental FWHM are km/s. Standard projected tomogram axes are
`vx=Vx sin(i)`, `vy=Vy sin(i)`, `vz=Vz cos(i)`. Do not compare true-velocity maps
with projected maps without conversion. The wavelength helper uses the longitudinal
relativistic formula `v/c = ((lambda/lambda0)^2-1)/((lambda/lambda0)^2+1)`; the
orbital projector uses the usual nonrelativistic vector projection. This is not a
fully relativistic radiative-transfer model.

The text loader divides wavelengths by `sqrt((1+gamma/c)/(1-gamma/c))`; gamma is
therefore already removed in its output. Supplying the original gamma again to
the projector double-corrects the data. A fitted residual gamma is a separate
model choice. Rest wavelengths, air/vacuum conventions, barycentric corrections,
and ephemerides are user inputs; the loader does not infer or perform them.

At i=90 degrees, any rearrangement along z with the same collapsed map produces
the same spectra. At i=0 or 180 the in-plane map is invisible. More generally,
`Vz -> Vz+delta, gamma -> gamma+delta*cos(i)` leaves the model unchanged.
A known systemic velocity is essential for absolute vertical-velocity claims.

## Discrete forward model

Each voxel stores flux, not a velocity-space density. Linear deposition distributes
it between adjacent spectral coordinates, including both endpoints. Outside-support
emission contributes zero unless it scatters into the window through the LSF.
The discrete sum of a sufficiently wide profile equals the cube sum. Converting
a cube to a velocity-space density requires division by voxel volume; changing
spectral sampling likewise requires a deliberate flux/bin-width convention.

The text loader retains sampled input ordinates. It does not apply a density
Jacobian, perform flux-conserving pixel-edge integration, or calibrate absolute
luminosity. For absolute spectrophotometry, transform `F_lambda d_lambda` to
spectral-bin flux before fitting and keep the same bin convention in prediction.
An arbitrary normalization is acceptable for morphology but must be recorded.

A Gaussian LSF uses `sigma=FWHM/(2*sqrt(2*ln(2)))`, a unit-sum discrete kernel,
and six-sigma support. Projection is padded before convolution and cropped to
observed support; the adjoint reverses these operations. This handles emission
scattered from outside the measured window. The LSF requires uniform spectral
sampling. Cube axes are always regular; unbroadened spectral axes may be irregular.

Exposure widths are full widths in phase cycles, centered on supplied phases.
Gauss-Legendre quadrature averages the instantaneous operator. Forward and adjoint
use identical nodes and weights; sinusoidal component modulation is evaluated at
each quadrature node. Zero/unknown exposure means an instantaneous model. Convert
seconds to cycles using the orbital period before constructing the container.
The default seven nodes are not an accuracy guarantee: increase phase_samples
until results stabilize at the actual velocity/exposure resolution.

## Preprocessing and covariance

Spectra are sorted by wavelength; duplicate wavelength samples are averaged with
independent-error propagation. Invalid measurements/errors are rejected by the
text loader. Native spectra are continuum-fit before one resampling step.
Continuum sidebands are defined in observed wavelengths. A scaled polynomial
basis avoids conditioning problems at large wavelength values. With provided
native errors the continuum fit uses inverse-variance weights.

For a fixed polynomial model and fixed native variances, the final line vector is
`y = L f`, where L combines interpolation and continuum subtraction. The loader
propagates `C = L diag(sigma_native^2) L.T`. This captures interpolation correlation,
continuum-fit uncertainty, and cross-terms if windows overlap. It does not include
wavelength calibration uncertainty, an incorrect continuum model, or native
correlation not supplied by the text format. Errors estimated from continuum
scatter use the residual degrees of freedom and remain conditional estimates.
No unit errors or tiny replacements are fabricated if estimation fails.

`propagate_covariance=False` returns the correct diagonal while discarding
correlations, and labels that approximation in metadata. Partial exposure rows
are rejected; all missing exposure widths remain explicitly unknown/instantaneous.

## Likelihood and scaling

An explicit boolean mask defines retained samples (`True` means included).
Unmasked data must be finite; unmasked errors must be positive finite. Invalid
masked data never receive weight. Missing errors select an explicitly recorded
unweighted least-squares objective, not a calibrated noise model.

The internal flux scale is the larger of the 99th percentile absolute retained
flux and median supplied/derived marginal error. For all-zero unweighted input,
scale is one. Data, errors, initial cube, and covariance are transformed
consistently; returned images are multiplied back into the original units.
The positivity floor and alpha are in internal units. This makes a uniform change
of flux units leave the fitted morphology and reported statistic unchanged.

For diagonal errors the data term is `sum(((A x-y)/sigma)^2)/(2*N)`.
For covariance, symmetric eigendecomposition constructs a whitening operator on
eigenvalues above `covariance_rtol * largest_eigenvalue` in each profile.
Significant negative eigenvalues are rejected. Redundant/small eigenmodes are
excluded; they are **not imposed as exact zero-noise constraints**. This is a
projected generalized-least-squares policy, recorded as
`covariance_gaussian_retained_modes`. Inspect residuals in original coordinates,
retained rank, and sensitivity to the threshold if near-singularity matters.
Profiles are assumed statistically independent.

The reported statistic is chi-square per retained sample/eigenmode. No effective
parameter-count correction is attempted. It is not an automatic goodness-of-fit
p-value, especially with estimated errors, nonlinear priors, or covariance truncation.

Relative entropy is `S=sum(x-d-x*log(x/d))` in internal units. For fixed positive
default d, the minimized objective is `0.5*mean_chi2-alpha*S` and its entropy
gradient is `alpha*log(x/d)`. This is generalized unnormalized relative entropy,
not the normalized p/q entropy used in every MEM implementation. Changing defaults
changes the objective; use history_stage when inspecting monotonicity.

## Defaults and stopping

Gaussian defaults smooth in physical km/s and preserve total flux. Squeezed
defaults compute each column's vertical centroid, redistribute its flux with a
Gaussian about that centroid, smooth in x/y, and blend with the Gaussian default.
They do not force emission toward Vz=0. Stable log weights prevent narrow kernels
from disappearing between grid points. These are priors, not measured resolution.

The total iteration budget is divided exactly across the allowed default stages.
Each SciPy status is retained. `converged` requires optimizer success and, when
multiple default stages are requested, the relative L2 default-change criterion.
One stage solves a fixed-default objective and does not claim outer fixed-point
convergence. The gradient diagnostic is the maximum absolute projected gradient.

`target_chi2` stops at the first accepted iterate reaching the requested statistic.
This is a discrepancy rule, not an automatic alpha search or constrained-MEM
solution. `success=True, status=target_reached` means deliberate stopping;
`converged` remains false because stationarity was not established. Numerical
success does not imply physical uniqueness or scientific acceptance.

Iteration exhaustion, line-search failure, a missed target, and nonconverged
outer defaults are returned explicitly. The CLI always saves the result and
JSON diagnostics, then exits with code 2 unless `--allow-incomplete` is explicit.
Stored image/configuration, input hash, normalization, coordinate convention,
response, likelihood, and histories make an incomplete result reviewable.

## Structural limitations and validation

The Fourier-cone null space prevents unique general 3D recovery, even without
noise. Compare priors at comparable fit quality, noise realizations, phase
coverage, system parameters, and observing response. Independent simulations
must use a different/finer discretization or analytic spectra. Test false
vertical features from planar or modulated sources. Statistical resampling alone
cannot measure structural nonuniqueness.

The package has no simultaneous modulation inversion, joint multi-line inversion,
full radiative transfer, Bayesian posterior sampler, or validated feature-detection
significance. Do not turn map peak/background ratios into significance. A 2D
baseline remains essential. See the validation workflow and these primary sources:
[Marsh 2022](https://doi.org/10.1093/mnras/stab3335),
[Steeghs 2003](https://arxiv.org/abs/astro-ph/0305365), and
[the reference trm-doppler implementation](https://github.com/trmrsh/trm-doppler).

## Migration from 0.1

1. Regenerate input products to retain covariance and preprocessing provenance.
2. Supply inclination explicitly and use gamma=0 on loader-corrected spectra.
3. Remove manual clipping and scaling unless deliberately required and recorded.
   The solver now normalizes internally; retune old alpha values.
4. Old map z labels use the opposite sign. For a symmetric z grid, reversing the
   image's last axis expresses the same velocity distribution in the new convention.
   More generally use new_vz=-old_vz[::-1], new_image=old_image[:,:,::-1]. Old files
   have no convention metadata, so this conversion must be an explicit user choice.
5. Use the same exposure/LSF response in fitting and forward diagnostics.
6. Inspect status and residuals. Increased iterations do not guarantee fit acceptance.
7. Old NPZ/FITS products still load, but absent metadata stays absent; no missing
   provenance or noise model is inferred automatically.
