"""Profile-space convolution helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def gaussian_kernel(axis: np.ndarray, fwhm: float, *, nsigma: float = 6.0) -> np.ndarray:
    """Return a normalized Gaussian kernel sampled on the spacing of ``axis``."""

    velocity_axis = np.asarray(axis, dtype=float)
    if velocity_axis.ndim != 1 or velocity_axis.size < 2:
        raise ValueError("axis must be a 1D array with at least two samples.")
    if fwhm <= 0:
        raise ValueError("fwhm must be positive.")
    dv = float(np.median(np.diff(velocity_axis)))
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half_width = max(1, int(np.ceil(nsigma * sigma / abs(dv))))
    samples = np.arange(-half_width, half_width + 1, dtype=float) * dv
    kernel = np.exp(-0.5 * (samples / sigma) ** 2)
    total = float(kernel.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("failed to normalize Gaussian kernel.")
    return kernel / total


def convolve_profiles_fft(profiles: np.ndarray, axis: np.ndarray, fwhm: float) -> np.ndarray:
    """Convolve phase-resolved profiles with a Gaussian using FFT convolution."""

    data = np.asarray(profiles, dtype=float)
    if data.ndim == 1:
        data_2d = data[None, :]
        squeeze = True
    elif data.ndim == 2:
        data_2d = data
        squeeze = False
    else:
        raise ValueError("profiles must be a 1D or 2D array.")
    kernel = gaussian_kernel(axis, fwhm)
    out = np.vstack(
        [fftconvolve(row, kernel, mode="same") for row in data_2d]
    )
    return out[0] if squeeze else out
