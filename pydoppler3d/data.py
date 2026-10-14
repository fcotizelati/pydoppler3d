"""Validated scientific containers and versioned, pickle-free NPZ/FITS IO."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import VelocityGrid3D


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value is not JSON serializable: {type(value).__name__}")


def metadata_json(meta):
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary.")
    return json.dumps(meta, default=_json_default, allow_nan=False, sort_keys=True)


def _path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class TrailedSpectra:
    """Signed spectral-bin flux, errors/covariance, and full exposure widths.

    Velocity is km/s, phase and exposure are cycles. mask=True means included.
    covariance has shape (nphase,nvelocity,nvelocity); profiles are independent.
    Nonfinite measurements are allowed only where mask=False.
    """

    phases: np.ndarray
    velocity: np.ndarray
    flux: np.ndarray
    error: np.ndarray | None = None
    exposure: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    covariance: np.ndarray | None = None
    mask: np.ndarray | None = None

    def __post_init__(self):
        phases, velocity, flux = (
            np.array(a, dtype=float, copy=True)
            for a in (self.phases, self.velocity, self.flux)
        )
        if phases.ndim != 1 or not phases.size or not np.all(np.isfinite(phases)):
            raise ValueError("phases must be a non-empty finite 1D array.")
        if (
            velocity.ndim != 1
            or velocity.size < 2
            or not np.all(np.isfinite(velocity))
            or not np.all(np.diff(velocity) > 0)
        ):
            raise ValueError("velocity must be finite and strictly increasing.")
        if flux.shape != (phases.size, velocity.size):
            raise ValueError("flux must have shape (nphase,nvelocity).")
        mask = (
            np.ones_like(flux, bool)
            if self.mask is None
            else np.array(self.mask, copy=True)
        )
        if mask.dtype != bool or mask.shape != flux.shape or not np.any(mask):
            raise ValueError("mask must be boolean, match flux, and include samples.")
        if not np.all(np.isfinite(flux[mask])):
            raise ValueError("included flux samples must be finite.")
        for name, arr in (("phases", phases), ("velocity", velocity), ("flux", flux)):
            object.__setattr__(self, name, arr)
        if self.mask is not None:
            object.__setattr__(self, "mask", mask)
        if self.error is not None:
            error = np.array(self.error, dtype=float, copy=True)
            if (
                error.shape != flux.shape
                or not np.all(np.isfinite(error[mask]))
                or np.any(error[mask] <= 0)
            ):
                raise ValueError(
                    "included errors must be positive finite and match flux."
                )
            object.__setattr__(self, "error", error)
        if self.exposure is not None:
            exposure = np.array(self.exposure, dtype=float, copy=True)
            if (
                exposure.shape != phases.shape
                or not np.all(np.isfinite(exposure))
                or np.any(exposure < 0)
            ):
                raise ValueError(
                    "exposure must match phases and be finite nonnegative cycles."
                )
            object.__setattr__(self, "exposure", exposure)
        if self.covariance is not None:
            cov = np.array(self.covariance, dtype=float, copy=True)
            if cov.shape != (phases.size, velocity.size, velocity.size):
                raise ValueError(
                    "covariance must have shape (nphase,nvelocity,nvelocity)."
                )
            for j, include in enumerate(mask):
                block = cov[j][np.ix_(include, include)]
                if not block.size:
                    continue
                if (
                    not np.all(np.isfinite(block))
                    or not np.allclose(
                        block, block.T, rtol=1e-10, atol=np.max(np.abs(block)) * 1e-12
                    )
                    or np.any(np.diag(block) <= 0)
                ):
                    raise ValueError(
                        "included covariance must be finite symmetric with positive diagonal."
                    )
            if self.error is not None and not np.allclose(
                np.diagonal(cov, axis1=1, axis2=2)[mask],
                self.error[mask] ** 2,
                rtol=1e-6,
                atol=0,
            ):
                raise ValueError("error squared must equal the covariance diagonal.")
            object.__setattr__(self, "covariance", cov)
        metadata_json(self.meta)

    def to_npz(self, path):
        arrays = {
            "phases": self.phases,
            "velocity": self.velocity,
            "flux": self.flux,
            "format_version": np.array(2),
            "meta_json": np.array(metadata_json(self.meta)),
        }
        for name in ("error", "exposure", "covariance", "mask"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(_path(path), **arrays)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            optional = {
                name: data[name]
                for name in ("error", "exposure", "covariance", "mask")
                if name in data and data[name].size
            }
            return cls(
                phases=data["phases"],
                velocity=data["velocity" if "velocity" in data else "velocities"],
                flux=data["flux" if "flux" in data else "profiles"],
                meta=json.loads(str(data["meta_json"])) if "meta_json" in data else {},
                **optional,
            )

    def to_fits(self, path, *, overwrite=False):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError("FITS IO requires pydoppler3d[fits].") from exc
        primary = fits.PrimaryHDU()
        primary.header["CONTENT"] = "PYDOP3D"
        primary.header["FMTVER"] = 2
        hdus = [primary]
        for name in (
            "flux",
            "phases",
            "velocity",
            "error",
            "exposure",
            "covariance",
            "mask",
        ):
            value = getattr(self, name)
            if value is not None:
                hdu = fits.ImageHDU(
                    value.astype(np.uint8) if name == "mask" else value,
                    name="PHASE" if name == "phases" else name.upper(),
                )
                if name == "velocity":
                    hdu.header["BUNIT"] = "km/s"
                elif name in ("phases", "exposure"):
                    hdu.header["BUNIT"] = "cycle"
                hdus.append(hdu)
        encoded = metadata_json(self.meta)
        hdus.append(
            fits.BinTableHDU.from_columns(
                [fits.Column(name="JSON", format=f"{len(encoded)}A", array=[encoded])],
                name="METADATA",
            )
        )
        fits.HDUList(hdus).writeto(_path(path), overwrite=overwrite)

    @classmethod
    def from_fits(cls, path):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError("FITS IO requires pydoppler3d[fits].") from exc
        with fits.open(path) as hdul:
            optional = {
                name: hdul[name.upper()].data
                for name in ("error", "exposure", "covariance", "mask")
                if name.upper() in hdul
            }
            if "mask" in optional:
                optional["mask"] = optional["mask"].astype(bool)
            meta = (
                json.loads(hdul["METADATA"].data["JSON"][0])
                if "METADATA" in hdul
                else {}
            )
            return cls(
                hdul["PHASE"].data,
                hdul["VELOCITY"].data,
                hdul["FLUX"].data,
                meta=meta,
                **optional,
            )


@dataclass(frozen=True)
class DopplerMap:
    """Flux-per-voxel cube, true velocity coordinates, and run provenance."""

    image: np.ndarray
    grid: VelocityGrid3D
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        image = np.array(self.image, dtype=float, copy=True)
        if image.shape != self.grid.shape or not np.all(np.isfinite(image)):
            raise ValueError(f"image must be finite with shape {self.grid.shape}.")
        object.__setattr__(self, "image", image)
        metadata_json(self.meta)

    def to_npz(self, path):
        np.savez_compressed(
            _path(path),
            image=self.image,
            vx=self.grid.vx,
            vy=self.grid.vy,
            vz=self.grid.vz,
            format_version=np.array(2),
            meta_json=np.array(metadata_json(self.meta)),
        )

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["image"],
                VelocityGrid3D(data["vx"], data["vy"], data["vz"]),
                json.loads(str(data["meta_json"])) if "meta_json" in data else {},
            )
