"""Compatibility helpers for the classic PyDoppler text-data layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._version import __version__
from .data import TrailedSpectra
from .geometry import CLIGHT_KMS, line_velocity_axis


@dataclass(frozen=True)
class PyDopplerPreparedData:
    """Continuum-subtracted spectra prepared from a PyDoppler-style dataset."""

    spectra: TrailedSpectra
    wavelength: np.ndarray
    flux: np.ndarray
    filenames: np.ndarray
    average_flux: np.ndarray
    continuum: np.ndarray
    average_line_flux: np.ndarray
    continuum_band: tuple[float, float, float, float]
    lam0: float
    delw: float
    gamma: float


def _read_phase_file(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"invalid phase row: {line!r}")
            exposure = float(parts[2]) if len(parts) >= 3 else np.nan
            rows.append((parts[0], float(parts[1]), exposure))
    if not rows:
        raise ValueError(f"{path} contains no spectra.")
    return rows


def _read_spectrum(path: Path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"{path} requires at least two wavelength/flux samples.")
    wave, flux = data[:, 0], data[:, 1]
    error = data[:, 2] if data.shape[1] >= 3 else None
    if not np.all(np.isfinite(data[:, :3])) or np.any(wave <= 0):
        raise ValueError(f"{path} contains nonfinite data or invalid wavelengths.")
    if error is not None and np.any(error <= 0):
        raise ValueError(f"{path} errors must be positive.")
    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]
    error = None if error is None else error[order]
    if np.any(np.diff(wave) == 0):
        if error is None:
            raise ValueError(
                "duplicate wavelengths require supplied native errors for propagation."
            )
        unique, inverse = np.unique(wave, return_inverse=True)
        counts = np.bincount(inverse)
        flux = np.bincount(inverse, weights=flux) / counts
        if error is not None:
            error = np.sqrt(np.bincount(inverse, weights=error**2)) / counts
        wave = unique
    if wave.size < 2:
        raise ValueError("at least two unique wavelengths are required.")
    return wave, flux, error


def _common_grid(loaded):
    lower = max(row[1][0] for row in loaded)
    upper = min(row[1][-1] for row in loaded)
    if lower >= upper:
        raise ValueError("input spectra do not share a wavelength overlap.")
    candidates = [row[1][(row[1] >= lower) & (row[1] <= upper)] for row in loaded]
    grid = max(candidates, key=len)
    if grid.size < 2:
        raise ValueError("common overlap has fewer than two samples.")
    return grid


def _interpolation_matrix(source, target):
    """Linear interpolation transform, including both endpoints."""
    if np.any(target < source[0]) or np.any(target > source[-1]):
        raise ValueError(
            "interpolation targets must be within native wavelength coverage."
        )
    lower = np.clip(
        np.searchsorted(source, target, side="right") - 1, 0, len(source) - 2
    )
    frac = (target - source[lower]) / (source[lower + 1] - source[lower])
    matrix = np.zeros((len(target), len(source)))
    matrix[np.arange(len(target)), lower] = 1 - frac
    matrix[np.arange(len(target)), lower + 1] = frac
    return matrix


def load_pydoppler_dataset(
    base_dir: str | Path,
    *,
    list_file="ugem0all.fas",
    spectra_dir=None,
    lam0=6562.8,
    delw=35.0,
    gamma=36.0,
    continuum_band: Sequence[float] | None = (6500.0, 6537.0, 6591.0, 6620.0),
    poly_degree=2,
    propagate_covariance=True,
) -> PyDopplerPreparedData:
    """Fit a native-grid continuum and propagate its linear transform exactly.

    Input wavelengths and continuum windows share the observed frame. gamma
    is removed once using the longitudinal relativistic Doppler factor.
    Outputs retain signed flux. The velocity grid is uniformly sampled; flux
    is a discrete spectral-bin value in the input ordinate units (not a
    calibrated integrated line luminosity). Missing native errors are estimated
    from continuum residuals; this conditional Gaussian estimate is recorded.
    Full covariance is conditional on independent native errors and the chosen
    polynomial continuum; it excludes wavelength/calibration/model uncertainty.
    """
    if not np.isfinite(lam0) or lam0 <= 0 or not np.isfinite(delw) or delw <= 0:
        raise ValueError("lam0 and delw must be positive and finite.")
    if not np.isfinite(gamma) or abs(gamma) >= CLIGHT_KMS:
        raise ValueError("gamma must be finite and satisfy |gamma| < c.")
    if int(poly_degree) != poly_degree or poly_degree < 0:
        raise ValueError("poly_degree must be a nonnegative integer.")
    poly_degree = int(poly_degree)
    base = Path(base_dir)
    list_path = Path(list_file)
    rows = _read_phase_file(list_path if list_path.is_absolute() else base / list_path)
    spectrum_base = base if spectra_dir is None else base / spectra_dir
    loaded = [(name, *_read_spectrum(spectrum_base / name)) for name, _, _ in rows]
    wavelength = _common_grid(loaded)
    flux = np.vstack(
        [np.interp(wavelength, wave, native) for _, wave, native, _ in loaded]
    )
    phases = np.asarray([r[1] for r in rows])
    exposure = np.asarray([r[2] for r in rows])
    if np.all(np.isnan(exposure)):
        exposure = None
    elif np.any(~np.isfinite(exposure)):
        raise ValueError("provide exposure widths for every spectrum or omit them all.")
    if continuum_band is None:
        count = min(max(3, int(round(0.05 * len(wavelength)))), len(wavelength) // 2)
        continuum_band = (
            wavelength[0],
            wavelength[count - 1],
            wavelength[-count],
            wavelength[-1],
        )
    band = tuple(float(v) for v in continuum_band)
    if len(band) != 4 or not np.all(np.isfinite(band)) or not np.all(np.diff(band) > 0):
        raise ValueError("continuum_band must contain four increasing finite limits.")
    beta = gamma / CLIGHT_KMS
    factor = np.sqrt((1 + beta) / (1 - beta))
    corrected = wavelength / factor
    line = (corrected > lam0 - delw) & (corrected < lam0 + delw)
    if line.sum() < 2:
        raise ValueError("line window contains fewer than two samples.")
    native_velocity = line_velocity_axis(corrected[line], lam0)
    velocity = np.linspace(
        native_velocity[0], native_velocity[-1], len(native_velocity)
    )
    target = lam0 * np.sqrt((1 + velocity / CLIGHT_KMS) / (1 - velocity / CLIGHT_KMS))
    # Roundoff at endpoints must not imply extrapolation.
    target[0], target[-1] = corrected[line][0], corrected[line][-1]
    line_flux, line_error, covariances, continua, estimated = [], [], [], [], []
    center, width = (band[0] + band[3]) / (2 * factor), (band[3] - band[0]) / factor
    for name, wave, native, error in loaded:
        side = ((wave > band[0]) & (wave < band[1])) | (
            (wave > band[2]) & (wave < band[3])
        )
        if side.sum() <= poly_degree + 1:
            raise ValueError(
                f"{name}: continuum requires more samples than fitted coefficients."
            )
        corrected_native = wave / factor
        design = np.polynomial.polynomial.polyvander(
            (corrected_native - center) / width, poly_degree
        )
        xside = design[side]
        if np.linalg.matrix_rank(xside) != poly_degree + 1:
            raise ValueError("continuum design is rank deficient.")
        is_estimated = error is None
        if is_estimated:
            residual = (
                native[side]
                - xside @ np.linalg.lstsq(xside, native[side], rcond=None)[0]
            )
            sigma = float(np.sqrt(np.sum(residual**2) / (side.sum() - poly_degree - 1)))
            if not np.isfinite(sigma) or sigma <= np.max(np.abs(native[side])) * 1e-14:
                raise ValueError(
                    f"{name}: cannot estimate noise from a noise-free continuum; provide errors."
                )
            error = np.full_like(native, sigma)
        # Rescale errors before weighted least squares to avoid tiny-flux conditioning.
        relative = error[side] / np.median(error[side])
        fit = np.linalg.pinv(xside / relative[:, None]) / relative[None, :]
        coeff = fit @ native[side]
        target_design = np.polynomial.polynomial.polyvander(
            (target - center) / width, poly_degree
        )
        transform = _interpolation_matrix(corrected_native, target)
        transform[:, side] -= target_design @ fit
        values = transform @ native
        weighted = transform * error[None, :]
        variances = np.sum(weighted**2, axis=1)
        line_flux.append(values)
        line_error.append(np.sqrt(variances))
        if propagate_covariance:
            covariances.append(weighted @ weighted.T)
        display_design = np.polynomial.polynomial.polyvander(
            (corrected - center) / width, poly_degree
        )
        continua.append(display_design @ coeff)
        estimated.append(is_estimated)
    meta = {
        "format_version": 2,
        "package_version": __version__,
        "source_directory": str(base.resolve()),
        "filenames": [r[0] for r in rows],
        "source_sha256": {
            name: hashlib.sha256((spectrum_base / name).read_bytes()).hexdigest()
            for name, *_ in rows
        },
        "lam0": float(lam0),
        "delw": float(delw),
        "gamma_removed_kms": float(gamma),
        "velocity_frame": "systemic_corrected",
        "velocity_units": "km/s",
        "phase_units": "cycle",
        "exposure_units": "cycle",
        "flux_units": "input_ordinate_per_discrete_spectral_bin",
        "continuum_band_observed": list(band),
        "poly_degree": poly_degree,
        "estimated_native_errors": estimated,
        "noise_assumption": "independent_native_gaussian_conditional_on_continuum",
        "covariance_policy": "full_linear_propagation"
        if propagate_covariance
        else "diagonal_approximation",
        "exposure_policy": "instantaneous_unknown"
        if exposure is None
        else "provided_cycle_widths",
    }
    spectra = TrailedSpectra(
        phases,
        velocity,
        np.asarray(line_flux),
        error=np.asarray(line_error),
        exposure=exposure,
        meta=meta,
        covariance=np.asarray(covariances) if propagate_covariance else None,
    )
    return PyDopplerPreparedData(
        spectra,
        wavelength,
        flux,
        np.asarray([r[0] for r in rows]),
        np.mean(flux, axis=0),
        np.mean(continua, axis=0),
        np.mean(line_flux, axis=0),
        band,
        float(lam0),
        float(delw),
        float(gamma),
    )


V834_CEN_CONFIG = {
    "list_file": "spectra/mcv",
    "spectra_dir": "spectra",
    "lam0": 4686.0,
    "delw": 18.0,
    "gamma": 0.0,
    "continuum_band": (4632.0, 4668.0, 4704.0, 4740.0),
}


def load_v834cen_dataset(
    base_dir: str | Path,
    **overrides,
) -> PyDopplerPreparedData:
    """Load the bundled V834 Cen He II 4686 spectra from the CDS doptomog set."""

    config = dict(V834_CEN_CONFIG)
    config.update(overrides)
    return load_pydoppler_dataset(base_dir, **config)
