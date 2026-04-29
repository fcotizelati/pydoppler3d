"""Forward and adjoint projection operators for 3D Doppler maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import VelocityGrid3D, radial_velocity


def _validate_axis(axis: np.ndarray) -> np.ndarray:
    out = np.asarray(axis, dtype=float)
    if out.ndim != 1 or out.size < 2:
        raise ValueError("velocity_axis must be a 1D array with at least two samples.")
    if not np.all(np.diff(out) > 0):
        raise ValueError("velocity_axis must be strictly increasing.")
    return out


def _deposit_linear(profile: np.ndarray, velocity_axis: np.ndarray, velocity, flux) -> None:
    idx = np.searchsorted(velocity_axis, velocity, side="right") - 1
    good = (idx >= 0) & (idx < velocity_axis.size - 1) & np.isfinite(velocity)
    if not np.any(good):
        return

    idx = idx[good]
    vel = velocity[good]
    values = flux[good]
    width = velocity_axis[idx + 1] - velocity_axis[idx]
    frac = (vel - velocity_axis[idx]) / width
    np.add.at(profile, idx, values * (1.0 - frac))
    np.add.at(profile, idx + 1, values * frac)


def project_cube(
    cube: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
    instrumental_fwhm: float | None = None,
) -> np.ndarray:
    """Project a 3D velocity cube into phase-resolved line profiles.

    `cube` is treated as flux per voxel. The returned array has shape
    `(nphase, nvelocity)`.
    """

    image = np.asarray(cube, dtype=float)
    if image.shape != grid.shape:
        raise ValueError(f"cube shape {image.shape} does not match grid {grid.shape}.")
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("phases must be a non-empty 1D array.")
    axis = _validate_axis(velocity_axis)

    vx, vy, vz = grid.mesh()
    profiles = np.zeros((phases.size, axis.size), dtype=float)
    flux = image.ravel()

    for iph, phase in enumerate(phases):
        vr = radial_velocity(
            vx,
            vy,
            vz,
            phase,
            inclination_deg=inclination_deg,
            gamma=gamma,
        ).ravel()
        _deposit_linear(profiles[iph], axis, vr, flux)

    if instrumental_fwhm is not None and instrumental_fwhm > 0:
        dv = float(np.median(np.diff(axis)))
        sigma_pix = float(instrumental_fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)) * dv)
        profiles = gaussian_filter1d(profiles, sigma=sigma_pix, axis=1, mode="nearest")

    return profiles


def back_project(
    residuals: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
) -> np.ndarray:
    """Back-project profile residuals onto a 3D velocity grid."""

    resid = np.asarray(residuals, dtype=float)
    phases = np.asarray(phases, dtype=float)
    axis = _validate_axis(velocity_axis)
    if resid.shape != (phases.size, axis.size):
        raise ValueError("residuals must have shape (nphase, nvelocity).")

    vx, vy, vz = grid.mesh()
    cube = np.zeros(grid.shape, dtype=float)
    for iph, phase in enumerate(phases):
        vr = radial_velocity(
            vx,
            vy,
            vz,
            phase,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
        cube += np.interp(vr.ravel(), axis, resid[iph], left=0.0, right=0.0).reshape(
            grid.shape
        )
    return cube
