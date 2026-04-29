"""Prototype reconstruction loops.

This module deliberately does not claim to implement the Marsh (2022) maximum
entropy solver yet. It provides a small positive Landweber iteration so the
forward and adjoint operators can be exercised in a realistic workflow while the
publication-grade MEM objective is developed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .defaults import gaussian_default
from .geometry import VelocityGrid3D
from .project import back_project, project_cube


@dataclass(frozen=True)
class LandweberConfig:
    iterations: int = 25
    step: float = 1e-4
    default_weight: float = 0.0
    default_fwhm_kms: float = 200.0
    nonnegative: bool = True


@dataclass(frozen=True)
class ReconstructionResult:
    image: np.ndarray
    chi2_history: np.ndarray


def landweber_reconstruct(
    profiles: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    config: LandweberConfig | None = None,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
) -> ReconstructionResult:
    """Run a simple positive gradient iteration for development tests."""

    if config is None:
        config = LandweberConfig()
    data = np.asarray(profiles, dtype=float)
    if initial is None:
        total = max(float(np.sum(data)), 1.0)
        image = np.full(grid.shape, total / np.prod(grid.shape), dtype=float)
    else:
        image = np.asarray(initial, dtype=float).copy()
        if image.shape != grid.shape:
            raise ValueError(f"initial shape {image.shape} does not match {grid.shape}.")

    history = []
    for _ in range(int(config.iterations)):
        model = project_cube(
            image,
            grid,
            phases,
            velocity_axis,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
        residual = data - model
        history.append(float(np.mean(residual**2)))
        grad = back_project(
            residual,
            grid,
            phases,
            velocity_axis,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
        image = image + float(config.step) * grad
        if config.default_weight > 0:
            default = gaussian_default(image, grid, fwhm_kms=config.default_fwhm_kms)
            image = (1.0 - config.default_weight) * image + config.default_weight * default
        if config.nonnegative:
            image = np.maximum(image, 0.0)

    return ReconstructionResult(image=image, chi2_history=np.asarray(history))
