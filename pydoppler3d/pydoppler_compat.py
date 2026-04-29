"""Compatibility helpers for the classic PyDoppler text-data layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

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
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain wavelength and flux columns.")
    wave = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    error = np.asarray(data[:, 2], dtype=float) if data.shape[1] >= 3 else None

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if error is not None:
        error = error[order]

    good = np.isfinite(wave) & np.isfinite(flux)
    if error is not None:
        good &= np.isfinite(error)
    wave = wave[good]
    flux = flux[good]
    if error is not None:
        error = error[good]
    if wave.size < 2:
        raise ValueError(f"{path} contains fewer than two finite samples.")

    if np.any(np.diff(wave) <= 0):
        unique, inverse = np.unique(wave, return_inverse=True)
        counts = np.bincount(inverse).astype(float)
        flux = np.bincount(inverse, weights=flux) / counts
        if error is not None:
            error = np.sqrt(np.bincount(inverse, weights=error**2)) / counts
        wave = unique
    return wave, flux, error


def _common_grid(loaded):
    overlap_min = float(max(wave[0] for _name, wave, _flux, _error in loaded))
    overlap_max = float(min(wave[-1] for _name, wave, _flux, _error in loaded))
    if overlap_min >= overlap_max:
        raise ValueError("input spectra do not share a common wavelength overlap.")

    counts = [
        int(np.count_nonzero((wave >= overlap_min) & (wave <= overlap_max)))
        for _name, wave, _flux, _error in loaded
    ]
    ref = int(np.argmax(counts))
    wave = loaded[ref][1]
    grid = np.asarray(wave[(wave >= overlap_min) & (wave <= overlap_max)], dtype=float)
    if grid.size < 2:
        raise ValueError("common wavelength overlap contains fewer than two samples.")
    return grid


def _auto_continuum_band(wavelength: np.ndarray):
    window = max(3, int(round(0.05 * wavelength.size)))
    left = wavelength[:window]
    right = wavelength[-window:]
    return (float(left.min()), float(left.max()), float(right.min()), float(right.max()))


def load_pydoppler_dataset(
    base_dir: str | Path,
    *,
    list_file: str = "ugem0all.fas",
    spectra_dir: str | Path | None = None,
    lam0: float = 6562.8,
    delw: float = 35.0,
    gamma: float = 36.0,
    continuum_band: Sequence[float] | None = (6500.0, 6537.0, 6591.0, 6620.0),
    poly_degree: int = 2,
) -> PyDopplerPreparedData:
    """Load and continuum-subtract a PyDoppler-style text dataset.

    The input layout is the same one used by classic :mod:`pydoppler`: a phase
    file with ``spectrum_name phase [delta_phase]`` rows and individual
    whitespace-separated spectra with ``wavelength flux [error]`` columns.
    """

    base = Path(base_dir)
    list_path = Path(list_file)
    if not list_path.is_absolute():
        list_path = base / list_path
    rows = _read_phase_file(list_path)
    spectrum_base = base if spectra_dir is None else base / spectra_dir
    loaded = []
    for filename, _phase, _exposure in rows:
        spectrum_path = Path(filename)
        if not spectrum_path.is_absolute():
            spectrum_path = spectrum_base / spectrum_path
        wave, flux, error = _read_spectrum(spectrum_path)
        loaded.append((filename, wave, flux, error))

    wavelength = _common_grid(loaded)
    fluxes = []
    errors = []
    for _filename, wave, flux, error in loaded:
        fluxes.append(np.interp(wavelength, wave, flux))
        if error is None:
            errors.append(None)
        else:
            errors.append(np.sqrt(np.interp(wavelength, wave, error**2)))
    flux = np.vstack(fluxes)

    phases = np.asarray([row[1] for row in rows], dtype=float)
    exposure = np.asarray([row[2] for row in rows], dtype=float)
    if not np.any(np.isfinite(exposure)):
        exposure_out = None
    else:
        fill = float(np.nanmedian(exposure[np.isfinite(exposure)]))
        exposure_out = np.where(np.isfinite(exposure), exposure, fill)

    if continuum_band is None:
        band = _auto_continuum_band(wavelength)
    else:
        if len(continuum_band) != 4:
            raise ValueError("continuum_band must contain four wavelength limits.")
        band = tuple(float(value) for value in continuum_band)
    mask = ((wavelength > band[0]) & (wavelength < band[1])) | (
        (wavelength > band[2]) & (wavelength < band[3])
    )
    if np.count_nonzero(mask) < max(poly_degree + 1, 3):
        raise ValueError("continuum windows contain too few wavelength samples.")

    beta = gamma / CLIGHT_KMS
    if abs(beta) >= 1:
        raise ValueError("gamma must satisfy |gamma| < c.")
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    corrected_wave = wavelength / doppler_factor
    line_mask = (corrected_wave > lam0 - delw) & (corrected_wave < lam0 + delw)
    if np.count_nonzero(line_mask) < 2:
        raise ValueError("no spectral samples fall inside lam0 +/- delw.")

    line_wave = corrected_wave[line_mask]
    velocity_native = line_velocity_axis(line_wave, lam0)
    velocity = np.linspace(velocity_native[0], velocity_native[-1], velocity_native.size)

    line_flux = np.zeros((flux.shape[0], velocity.size), dtype=float)
    line_error = np.zeros_like(line_flux)
    for index, row in enumerate(flux):
        coeff = np.polyfit(corrected_wave[mask], row[mask], poly_degree)
        continuum = np.poly1d(coeff)(line_wave)
        residual = row[line_mask] - continuum
        line_flux[index] = np.interp(velocity, velocity_native, residual)

        if errors[index] is None:
            cont_resid = row[mask] - np.poly1d(coeff)(corrected_wave[mask])
            sigma = float(np.nanstd(cont_resid))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            line_error[index] = sigma
        else:
            native_error = np.sqrt(np.interp(line_wave, corrected_wave, errors[index] ** 2))
            line_error[index] = np.interp(velocity, velocity_native, native_error)

    finite_positive = np.isfinite(line_error) & (line_error > 0)
    floor = float(np.nanmedian(line_error[finite_positive]) * 1e-6)
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-12
    line_error = np.clip(line_error, floor, None)

    average_flux = np.mean(flux, axis=0)
    average_coeff = np.polyfit(corrected_wave[mask], average_flux[mask], poly_degree)
    average_continuum = np.poly1d(average_coeff)(wavelength)
    average_line_flux = np.mean(line_flux, axis=0)

    spectra = TrailedSpectra(
        phases=phases,
        velocity=velocity,
        flux=line_flux,
        error=line_error,
        exposure=exposure_out,
    )
    return PyDopplerPreparedData(
        spectra=spectra,
        wavelength=wavelength,
        flux=flux,
        filenames=np.asarray([row[0] for row in rows], dtype=str),
        average_flux=average_flux,
        continuum=average_continuum,
        average_line_flux=average_line_flux,
        continuum_band=band,
        lam0=float(lam0),
        delw=float(delw),
        gamma=float(gamma),
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
