"""Matched discrete forward/adjoint operators, including the observing response."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .convolution import gaussian_kernel
from .geometry import VelocityGrid3D


def _validate_axis(axis: np.ndarray) -> np.ndarray:
    out = np.array(axis, dtype=float, copy=True)
    if out.ndim != 1 or out.size < 2 or not np.all(np.isfinite(out)):
        raise ValueError("velocity_axis must contain at least two finite samples.")
    if not np.all(np.diff(out) > 0):
        raise ValueError("velocity_axis must be strictly increasing.")
    return out


class ProjectionOperator:
    """Flux-per-voxel to flux-per-spectral-bin linear operator.

    Exposure is a full phase width in cycles, centered on each supplied phase.
    Gauss-Legendre quadrature averages over the exposure. Increase phase_samples
    until the model is stable for long exposures or narrow spectral bins.
    Gaussian broadening is zero-padded outside the measured spectral window;
    emission outside the window can scatter into it. Forward and adjoint share
    every interpolation weight and response operation. Geometry caching has a
    strict byte budget; uncached entries are evaluated as needed.
    """

    def __init__(
        self,
        grid,
        phases,
        velocity_axis,
        *,
        inclination_deg=90.0,
        gamma=0.0,
        instrumental_fwhm=None,
        exposure=None,
        phase_samples=7,
        cache_max_bytes=64 * 1024**2,
    ):
        self.grid = grid
        self.phases = np.array(phases, dtype=float, copy=True)
        if (
            self.phases.ndim != 1
            or not self.phases.size
            or not np.all(np.isfinite(self.phases))
        ):
            raise ValueError("phases must be a non-empty finite 1D array.")
        self.velocity_axis = _validate_axis(velocity_axis)
        if not np.isfinite(inclination_deg) or not 0 <= inclination_deg <= 180:
            raise ValueError("inclination_deg must be finite and between 0 and 180.")
        if not np.isfinite(gamma):
            raise ValueError("gamma must be finite.")
        if (
            isinstance(phase_samples, bool)
            or int(phase_samples) != phase_samples
            or phase_samples < 1
        ):
            raise ValueError("phase_samples must be a positive integer.")
        if int(cache_max_bytes) != cache_max_bytes or cache_max_bytes < 0:
            raise ValueError("cache_max_bytes must be a nonnegative integer.")
        self.cube_shape = grid.shape
        self.profile_shape = (self.phases.size, self.velocity_axis.size)
        self.shape = (int(np.prod(self.profile_shape)), int(np.prod(grid.shape)))
        widths = (
            np.zeros_like(self.phases)
            if exposure is None
            else np.broadcast_to(exposure, self.phases.shape).astype(float)
        )
        if not np.all(np.isfinite(widths)) or np.any(widths < 0):
            raise ValueError("exposure must contain finite nonnegative phase widths.")
        nodes, weights = np.polynomial.legendre.leggauss(int(phase_samples))
        self.sample_phases = self.phases[:, None] + widths[:, None] * nodes[None, :] / 2
        self.sample_weights = weights / 2
        self._samples = [
            (j, k)
            for j in range(self.phases.size)
            for k in (range(int(phase_samples)) if widths[j] > 0 else [0])
        ]
        self._widths = widths
        self._sin = np.sin(2 * np.pi * self.sample_phases)
        self._cos = np.cos(2 * np.pi * self.sample_phases)
        inc = np.deg2rad(inclination_deg)
        # Snap the mathematically degenerate directions to exact zeros.
        si, ci = np.sin(inc), np.cos(inc)
        si = 0.0 if abs(si) < 1e-15 else si
        ci = 0.0 if abs(ci) < 1e-15 else ci
        vx, vy, vz = grid.mesh()
        self._x = -si * vx.ravel()
        self._y = si * vy.ravel()
        self._z = float(gamma) - ci * vz.ravel()
        self._kernel = None
        self._pad = 0
        self._axis = self.velocity_axis
        if instrumental_fwhm is not None:
            if not np.isfinite(instrumental_fwhm) or instrumental_fwhm < 0:
                raise ValueError("instrumental_fwhm must be finite and nonnegative.")
            if instrumental_fwhm > 0:
                self._kernel = gaussian_kernel(self.velocity_axis, instrumental_fwhm)
                self._pad = self._kernel.size // 2
                dv = self.velocity_axis[1] - self.velocity_axis[0]
                self._axis = (
                    self.velocity_axis[0]
                    + np.arange(-self._pad, self.velocity_axis.size + self._pad) * dv
                )
        self._cache = {}
        self.cache_bytes = 0
        for key in self._samples:
            # Upper bound avoids allocating an entry when no room remains.
            if self.cache_bytes + self.shape[1] * 24 > cache_max_bytes:
                break
            entry = self._entry(*key)
            self._cache[key] = entry
            self.cache_bytes += sum(a.nbytes for a in entry)

    def _entry(self, j, k):
        velocity = self._x * self._cos[j, k] + self._y * self._sin[j, k] + self._z
        good = np.flatnonzero(
            (velocity >= self._axis[0]) & (velocity <= self._axis[-1])
        )
        vel = velocity[good]
        lower = np.clip(
            np.searchsorted(self._axis, vel, side="right") - 1, 0, self._axis.size - 2
        )
        frac = (vel - self._axis[lower]) / (self._axis[lower + 1] - self._axis[lower])
        return good, lower, frac

    def _entries(self, phase_weights):
        if phase_weights is not None:
            phase_weights = np.asarray(phase_weights, dtype=float)
            if phase_weights.shape != self.sample_phases.shape or not np.all(
                np.isfinite(phase_weights)
            ):
                raise ValueError(
                    "phase_weights must match sample_phases and be finite."
                )
        for j, k in self._samples:
            weight = self.sample_weights[k] if self._widths[j] > 0 else 1.0
            if phase_weights is not None:
                weight *= phase_weights[j, k]
            entry = self._cache.get((j, k))
            yield j, weight, self._entry(j, k) if entry is None else entry

    def forward(self, cube, *, phase_weights=None):
        image = np.asarray(cube, dtype=float)
        if image.shape != self.cube_shape or not np.all(np.isfinite(image)):
            raise ValueError(f"cube must be finite with shape {self.cube_shape}.")
        out = np.zeros((self.phases.size, self._axis.size))
        flat = image.ravel()
        for j, weight, (good, lower, frac) in self._entries(phase_weights):
            flux = weight * flat[good]
            np.add.at(out[j], lower, flux * (1 - frac))
            np.add.at(out[j], lower + 1, flux * frac)
        if self._kernel is not None:
            out = fftconvolve(out, self._kernel[None, :], mode="same", axes=1)
            out = out[:, self._pad : -self._pad]
        return out

    def adjoint(self, profiles, *, phase_weights=None):
        residual = np.asarray(profiles, dtype=float)
        if residual.shape != self.profile_shape or not np.all(np.isfinite(residual)):
            raise ValueError(
                f"profiles must be finite with shape {self.profile_shape}."
            )
        if self._kernel is not None:
            residual = np.pad(residual, ((0, 0), (self._pad, self._pad)))
            residual = fftconvolve(residual, self._kernel[None, :], mode="same", axes=1)
        out = np.zeros(self.shape[1])
        for j, weight, (good, lower, frac) in self._entries(phase_weights):
            out[good] += weight * (
                (1 - frac) * residual[j, lower] + frac * residual[j, lower + 1]
            )
        return out.reshape(self.cube_shape)


def project_cube(cube, grid: VelocityGrid3D, phases, velocity_axis, **response):
    """Project flux per voxel to spectral-bin flux; see ProjectionOperator."""
    return ProjectionOperator(grid, phases, velocity_axis, **response).forward(cube)


def back_project(residuals, grid: VelocityGrid3D, phases, velocity_axis, **response):
    """Apply the exact transpose of project_cube with the same response."""
    return ProjectionOperator(grid, phases, velocity_axis, **response).adjoint(
        residuals
    )
