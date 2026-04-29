"""Data containers and lightweight FITS/NPZ IO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import VelocityGrid3D


@dataclass(frozen=True)
class TrailedSpectra:
    """Phase-resolved line profiles in velocity space."""

    phases: np.ndarray
    velocity: np.ndarray
    flux: np.ndarray
    error: np.ndarray | None = None
    exposure: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phases = np.asarray(self.phases, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        if phases.ndim != 1 or phases.size == 0:
            raise ValueError("phases must be a non-empty 1D array.")
        if velocity.ndim != 1 or velocity.size < 2:
            raise ValueError("velocity must be a 1D array with at least two samples.")
        if flux.shape != (phases.size, velocity.size):
            raise ValueError("flux must have shape (nphase, nvelocity).")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "flux", flux)
        if self.error is not None:
            error = np.asarray(self.error, dtype=float)
            if error.shape != flux.shape:
                raise ValueError("error must have the same shape as flux.")
            object.__setattr__(self, "error", error)
        if self.exposure is not None:
            exposure = np.asarray(self.exposure, dtype=float)
            if exposure.shape != phases.shape:
                raise ValueError("exposure must have shape (nphase,).")
            object.__setattr__(self, "exposure", exposure)

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            phases=self.phases,
            velocity=self.velocity,
            flux=self.flux,
            error=np.array([]) if self.error is None else self.error,
            exposure=np.array([]) if self.exposure is None else self.exposure,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrailedSpectra":
        with np.load(path) as data:
            files = set(data.files)
            velocity_name = "velocity" if "velocity" in files else "velocities"
            flux_name = "flux" if "flux" in files else "profiles"
            error = data["error"] if "error" in files and data["error"].size else None
            exposure = (
                data["exposure"] if "exposure" in files and data["exposure"].size else None
            )
            return cls(
                phases=data["phases"],
                velocity=data[velocity_name],
                flux=data[flux_name],
                error=error,
                exposure=exposure,
            )

    def to_fits(self, path: str | Path, *, overwrite: bool = False) -> None:
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("FITS IO requires astropy: pip install astropy") from exc

        primary = fits.PrimaryHDU()
        primary.header["CONTENT"] = "PYDOP3D"
        hdus = [
            primary,
            fits.ImageHDU(self.flux, name="FLUX"),
            fits.ImageHDU(self.phases, name="PHASE"),
            fits.ImageHDU(self.velocity, name="VELOCITY"),
        ]
        if self.error is not None:
            hdus.append(fits.ImageHDU(self.error, name="ERROR"))
        if self.exposure is not None:
            hdus.append(fits.ImageHDU(self.exposure, name="EXPOSURE"))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)

    @classmethod
    def from_fits(cls, path: str | Path) -> "TrailedSpectra":
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("FITS IO requires astropy: pip install astropy") from exc

        with fits.open(path) as hdul:
            error = hdul["ERROR"].data if "ERROR" in hdul else None
            exposure = hdul["EXPOSURE"].data if "EXPOSURE" in hdul else None
            return cls(
                phases=hdul["PHASE"].data,
                velocity=hdul["VELOCITY"].data,
                flux=hdul["FLUX"].data,
                error=error,
                exposure=exposure,
            )


@dataclass(frozen=True)
class DopplerMap:
    """A 3D Doppler image and its velocity grid."""

    image: np.ndarray
    grid: VelocityGrid3D

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=float)
        if image.shape != self.grid.shape:
            raise ValueError(f"image shape {image.shape} does not match grid {self.grid.shape}.")
        object.__setattr__(self, "image", image)

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            image=self.image,
            vx=self.grid.vx,
            vy=self.grid.vy,
            vz=self.grid.vz,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "DopplerMap":
        with np.load(path) as data:
            return cls(
                image=data["image"],
                grid=VelocityGrid3D(vx=data["vx"], vy=data["vy"], vz=data["vz"]),
            )
