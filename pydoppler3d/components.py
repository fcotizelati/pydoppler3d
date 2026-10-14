"""Linear image components for modulation and signed maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .geometry import VelocityGrid3D
from .project import ProjectionOperator

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
    phases,
    velocity_axis,
    **response,
) -> np.ndarray:
    """Project components; modulation is averaged within each exposure."""
    operator = ProjectionOperator(grid, phases, velocity_axis, **response)
    profiles = np.zeros(operator.profile_shape)
    for component in components:
        if not np.isfinite(component.scale):
            raise ValueError("component scale must be finite.")
        weights = float(component.scale) * phase_weight(
            component.kind, operator.sample_phases
        )
        profiles += operator.forward(component.image, phase_weights=weights)
    return profiles


def back_project_components(
    residuals,
    kinds: Iterable[ComponentKind],
    grid: VelocityGrid3D,
    phases,
    velocity_axis,
    *,
    scales=None,
    **response,
) -> dict[ComponentKind, np.ndarray]:
    """Transpose of project_components for distinct kinds and matching scales.

    scales maps component kinds to the same scalar multipliers used forward.
    All observing-response keywords must also match the forward call.
    """
    operator = ProjectionOperator(grid, phases, velocity_axis, **response)
    kinds = list(kinds)
    if len(set(kinds)) != len(kinds):
        raise ValueError("component kinds must be distinct for dictionary output.")
    scales = {} if scales is None else scales
    result = {}
    for kind in kinds:
        scale = float(scales.get(kind, 1.0))
        if not np.isfinite(scale):
            raise ValueError("component scale must be finite.")
        weights = scale * phase_weight(kind, operator.sample_phases)
        result[kind] = operator.adjoint(residuals, phase_weights=weights)
    return result
