"""Default-map helpers for regularized 3D Doppler tomography."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import VelocityGrid3D


def _fwhm_to_sigma_pix(fwhm_kms: float | tuple[float, float, float], grid: VelocityGrid3D):
    if np.isscalar(fwhm_kms):
        fwhm = (float(fwhm_kms),) * 3
    else:
        fwhm = tuple(float(value) for value in fwhm_kms)
        if len(fwhm) != 3:
            raise ValueError("fwhm_kms must be scalar or length 3.")
    factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return tuple(max(0.0, value / (factor * step)) for value, step in zip(fwhm, grid.spacing))


def _preserve_total(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_total = float(np.nansum(reference))
    cand_total = float(np.nansum(candidate))
    if cand_total > 0 and np.isfinite(cand_total):
        candidate = candidate * (ref_total / cand_total)
    return candidate


def gaussian_default(
    cube: np.ndarray,
    grid: VelocityGrid3D,
    *,
    fwhm_kms: float | tuple[float, float, float] = 200.0,
) -> np.ndarray:
    """Return an isotropically or anisotropically blurred default map."""

    image = np.asarray(cube, dtype=float)
    if image.shape != grid.shape:
        raise ValueError(f"cube shape {image.shape} does not match grid {grid.shape}.")
    sigma = _fwhm_to_sigma_pix(fwhm_kms, grid)
    blurred = gaussian_filter(image, sigma=sigma, mode="nearest")
    return _preserve_total(image, blurred)


def squeezed_default(
    cube: np.ndarray,
    grid: VelocityGrid3D,
    *,
    sigma_vz_kms: float = 100.0,
    pull: float = 0.5,
    fwhm_xy_kms: float = 200.0,
    fwhm_iso_kms: float = 200.0,
) -> np.ndarray:
    """Blend a Gaussian default with a vz-squeezed default.

    The squeezed component collapses each `(vx, vy)` column, computes its
    flux-weighted `vz` centroid, and re-expands it as a Gaussian along `vz`.
    `pull=0` gives the isotropic Gaussian default; `pull=1` gives the pure
    squeezed default.
    """

    image = np.asarray(cube, dtype=float)
    if image.shape != grid.shape:
        raise ValueError(f"cube shape {image.shape} does not match grid {grid.shape}.")
    if not 0.0 <= pull <= 1.0:
        raise ValueError("pull must be between 0 and 1.")
    if sigma_vz_kms <= 0:
        raise ValueError("sigma_vz_kms must be positive.")

    total_xy = np.sum(image, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.sum(image * grid.vz.reshape(1, 1, -1), axis=2) / total_xy
    centroid = np.where(np.isfinite(centroid), centroid, 0.0)

    dz = grid.vz.reshape(1, 1, -1) - centroid[:, :, None]
    weights = np.exp(-0.5 * (dz / float(sigma_vz_kms)) ** 2)
    weights_sum = np.sum(weights, axis=2, keepdims=True)
    weights = np.divide(weights, weights_sum, out=np.zeros_like(weights), where=weights_sum > 0)
    squeezed = total_xy[:, :, None] * weights

    if fwhm_xy_kms > 0:
        factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sig_x = float(fwhm_xy_kms) / (factor * grid.spacing[0])
        sig_y = float(fwhm_xy_kms) / (factor * grid.spacing[1])
        squeezed = gaussian_filter(squeezed, sigma=(sig_x, sig_y, 0.0), mode="nearest")
        squeezed = _preserve_total(image, squeezed)

    isotropic = gaussian_default(image, grid, fwhm_kms=fwhm_iso_kms)
    blended = (1.0 - pull) * isotropic + pull * squeezed
    return _preserve_total(image, blended)
