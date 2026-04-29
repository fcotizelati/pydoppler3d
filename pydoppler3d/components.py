"""Linear image components for modulation and signed maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .geometry import VelocityGrid3D
from .project import back_project, project_cube

ComponentKind = Literal[
    "constant",
    "negative",
    "sin",
    "negative_sin",
    "cos",
    "negative_cos",
    "sin2",
    "negative_sin2",
    "cos2",
    "negative_cos2",
]


@dataclass(frozen=True)
class MapComponent:
    """One linear image component in a modulated Doppler model."""

    image: np.ndarray
    kind: ComponentKind = "constant"
    scale: float = 1.0


def phase_weight(kind: ComponentKind, phases: np.ndarray) -> np.ndarray:
    """Return the phase-dependent multiplier for a component kind."""

    phase = np.asarray(phases, dtype=float)
    angle = 2.0 * np.pi * phase
    sign = -1.0 if kind.startswith("negative") else 1.0
    base = kind.removeprefix("negative_")
    if base == "negative":
        base = "constant"
    if base == "constant":
        return np.full_like(phase, sign, dtype=float)
    if base == "sin":
        return sign * np.sin(angle)
    if base == "cos":
        return sign * np.cos(angle)
    if base == "sin2":
        return sign * np.sin(2.0 * angle)
    if base == "cos2":
        return sign * np.cos(2.0 * angle)
    raise ValueError(f"unsupported component kind: {kind!r}")


def project_components(
    components: Iterable[MapComponent],
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
    instrumental_fwhm: float | None = None,
) -> np.ndarray:
    """Project signed and modulated components into profiles."""

    phases = np.asarray(phases, dtype=float)
    profiles = np.zeros((phases.size, np.asarray(velocity_axis).size), dtype=float)
    for component in components:
        model = project_cube(
            component.image,
            grid,
            phases,
            velocity_axis,
            inclination_deg=inclination_deg,
            gamma=gamma,
            instrumental_fwhm=instrumental_fwhm,
        )
        profiles += float(component.scale) * phase_weight(component.kind, phases)[:, None] * model
    return profiles


def back_project_components(
    residuals: np.ndarray,
    kinds: Iterable[ComponentKind],
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
) -> dict[ComponentKind, np.ndarray]:
    """Back-project residuals for each requested component kind."""

    phases = np.asarray(phases, dtype=float)
    result = {}
    for kind in kinds:
        weighted = np.asarray(residuals, dtype=float) * phase_weight(kind, phases)[:, None]
        result[kind] = back_project(
            weighted,
            grid,
            phases,
            velocity_axis,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
    return result
