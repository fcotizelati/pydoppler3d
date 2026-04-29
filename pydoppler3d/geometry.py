"""Velocity-space geometry for 3D Doppler tomography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

CLIGHT_KMS = 299_792.458


@dataclass(frozen=True)
class VelocityGrid3D:
    """Regular 3D velocity grid in km/s."""

    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    def __post_init__(self) -> None:
        for name, axis in (("vx", self.vx), ("vy", self.vy), ("vz", self.vz)):
            arr = np.asarray(axis, dtype=float)
            if arr.ndim != 1 or arr.size < 2:
                raise ValueError(f"{name} must be a 1D array with at least two samples.")
            if not np.all(np.diff(arr) > 0):
                raise ValueError(f"{name} must be strictly increasing.")
            object.__setattr__(self, name, arr)

    @classmethod
    def regular(
        cls,
        *,
        vlim_xy: float = 2_000.0,
        nxy: int = 101,
        vlim_z: float | None = None,
        nz: int | None = None,
    ) -> "VelocityGrid3D":
        """Create a symmetric regular grid."""

        if vlim_z is None:
            vlim_z = float(vlim_xy)
        if nz is None:
            nz = int(nxy)
        if nxy < 2 or nz < 2:
            raise ValueError("nxy and nz must be at least 2.")
        vx = np.linspace(-float(vlim_xy), float(vlim_xy), int(nxy))
        vy = np.linspace(-float(vlim_xy), float(vlim_xy), int(nxy))
        vz = np.linspace(-float(vlim_z), float(vlim_z), int(nz))
        return cls(vx=vx, vy=vy, vz=vz)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.vx.size, self.vy.size, self.vz.size)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        return tuple(float(axis[1] - axis[0]) for axis in (self.vx, self.vy, self.vz))

    def mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(vx, vy, vz)` mesh arrays with `ij` indexing."""

        return np.meshgrid(self.vx, self.vy, self.vz, indexing="ij")


def radial_velocity(
    vx: np.ndarray | float,
    vy: np.ndarray | float,
    vz: np.ndarray | float,
    phase: np.ndarray | float,
    *,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
) -> np.ndarray:
    """Project 3D velocity coordinates into observed radial velocity.

    The returned velocity is in km/s. Phase is measured in orbital cycles.
    For array-valued phases, the phase dimensions are appended to the velocity
    coordinate dimensions.
    """

    vx_arr, vy_arr, vz_arr = np.broadcast_arrays(vx, vy, vz)
    phase_arr = np.asarray(phase, dtype=float)
    angle = 2.0 * np.pi * phase_arr
    inc = np.deg2rad(float(inclination_deg))
    sin_i = np.sin(inc)
    cos_i = np.cos(inc)

    if angle.ndim == 0:
        return (
            float(gamma)
            + sin_i * (-vx_arr * np.cos(angle) + vy_arr * np.sin(angle))
            + cos_i * vz_arr
        )

    expand = (Ellipsis,) + (None,) * angle.ndim
    phase_shape = (1,) * vx_arr.ndim + angle.shape
    cos_phi = np.cos(angle).reshape(phase_shape)
    sin_phi = np.sin(angle).reshape(phase_shape)

    return (
        float(gamma)
        + sin_i * (-vx_arr[expand] * cos_phi + vy_arr[expand] * sin_phi)
        + cos_i * vz_arr[expand]
    )


def line_velocity_axis(
    wavelength: np.ndarray | float,
    rest_wavelength: float,
    *,
    relativistic: bool = True,
) -> np.ndarray:
    """Convert wavelength to velocity relative to a line rest wavelength."""

    wave = np.asarray(wavelength, dtype=float)
    ratio = wave / float(rest_wavelength)
    if relativistic:
        return CLIGHT_KMS * (ratio**2 - 1.0) / (ratio**2 + 1.0)
    return CLIGHT_KMS * (ratio - 1.0)
