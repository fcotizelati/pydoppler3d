"""Pure-Python entropy-regularized Doppler-map reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, minimize

from .defaults import gaussian_default, squeezed_default
from .geometry import VelocityGrid3D
from .project import back_project, project_cube

DefaultKind = Literal["gaussian", "squeezed"]


@dataclass(frozen=True)
class MemConfig:
    """Configuration for the maximum-entropy Doppler-map optimizer.

    The solver uses SciPy's bounded L-BFGS-B optimizer with analytic gradients
    from the forward projector and its transpose. This is still not a drop-in
    clone of MEMSYS, but it follows the same numerical structure needed for
    serious maximum-entropy Doppler tomography:

    ``0.5 * chi2 - alpha * entropy(image, default)``

    with non-negativity bounds.
    """

    iterations: int = 100
    alpha: float = 1e-3
    target_chi2: float | None = None
    default: DefaultKind = "squeezed"
    default_fwhm_kms: float = 200.0
    squeeze_pull: float = 0.5
    squeeze_sigma_vz_kms: float = 100.0
    positivity_floor: float = 1e-12
    tolerance: float = 1e-7
    default_updates: int = 2
    lbfgsb_history: int = 10
    max_line_search: int = 20


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
    objective_history: np.ndarray | None = None
    entropy_history: np.ndarray | None = None


def _validated_error(error: np.ndarray | None, data: np.ndarray) -> np.ndarray:
    if error is None:
        return np.ones_like(data, dtype=float)
    sigma = np.asarray(error, dtype=float)
    if sigma.shape != data.shape:
        raise ValueError("error must have the same shape as profiles.")
    good = np.isfinite(sigma) & (sigma > 0)
    if not np.any(good):
        raise ValueError("error must contain at least one positive finite value.")
    floor = float(np.nanmedian(sigma[good]) * 1e-6)
    out = np.where(good, sigma, floor)
    return np.clip(out, floor, None)


def _default_map(image: np.ndarray, grid: VelocityGrid3D, config: MemConfig) -> np.ndarray:
    if config.default == "gaussian":
        default = gaussian_default(image, grid, fwhm_kms=config.default_fwhm_kms)
    elif config.default == "squeezed":
        default = squeezed_default(
            image,
            grid,
            sigma_vz_kms=config.squeeze_sigma_vz_kms,
            pull=config.squeeze_pull,
            fwhm_xy_kms=config.default_fwhm_kms,
            fwhm_iso_kms=config.default_fwhm_kms,
        )
    else:  # pragma: no cover - protected by Literal typing and normal API
        raise ValueError(f"unsupported default type: {config.default!r}")
    return np.clip(default, config.positivity_floor, None)


def entropy(image: np.ndarray, default: np.ndarray, *, floor: float = 1e-12) -> float:
    """Return relative entropy used by the MEM objective."""

    img = np.clip(np.asarray(image, dtype=float), floor, None)
    default = np.clip(np.asarray(default, dtype=float), floor, None)
    return float(np.sum(img - default - img * np.log(img / default)))


def _initial_image(
    data: np.ndarray,
    grid: VelocityGrid3D,
    initial: np.ndarray | None,
    config: MemConfig,
) -> np.ndarray:
    if initial is None:
        positive = np.clip(data, 0.0, None)
        total = max(
            float(np.mean(np.sum(positive, axis=1))),
            config.positivity_floor,
        )
        return np.full(grid.shape, total / np.prod(grid.shape), dtype=float)

    image = np.asarray(initial, dtype=float).copy()
    if image.shape != grid.shape:
        raise ValueError(f"initial shape {image.shape} does not match {grid.shape}.")
    return np.clip(image, config.positivity_floor, None)


def _objective_and_gradient(
    image: np.ndarray,
    default: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    config: MemConfig,
    *,
    inclination_deg: float,
    gamma: float,
) -> tuple[float, np.ndarray, float, float]:
    model = project_cube(
        image,
        grid,
        phases,
        velocity_axis,
        inclination_deg=inclination_deg,
        gamma=gamma,
    )
    residual = model - data
    chi2 = float(np.mean(np.square(residual / sigma)))
    ent = entropy(image, default, floor=config.positivity_floor)
    objective = 0.5 * float(np.mean(np.square(residual) * weights)) - config.alpha * ent

    weighted_residual = residual * weights / residual.size
    likelihood_grad = back_project(
        weighted_residual,
        grid,
        phases,
        velocity_axis,
        inclination_deg=inclination_deg,
        gamma=gamma,
    )
    entropy_grad = np.log(np.clip(image, config.positivity_floor, None) / default)
    grad = likelihood_grad + config.alpha * entropy_grad
    return objective, grad, chi2, ent


def _mem_reconstruct_lbfgsb(
    image: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    config: MemConfig,
    *,
    inclination_deg: float,
    gamma: float,
) -> ReconstructionResult:
    chi2_history: list[float] = []
    objective_history: list[float] = []
    entropy_history: list[float] = []

    default_updates = max(1, int(config.default_updates))
    maxiter = max(1, int(np.ceil(max(1, int(config.iterations)) / default_updates)))

    for _update in range(default_updates):
        fixed_default = _default_map(image, grid, config)
        objective, grad, chi2, ent = _objective_and_gradient(
            image,
            fixed_default,
            data,
            sigma,
            weights,
            grid,
            phases,
            velocity_axis,
            config,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
        if not objective_history:
            objective_history.append(objective)
            chi2_history.append(chi2)
            entropy_history.append(ent)

        if config.target_chi2 is not None and chi2 <= config.target_chi2:
            break

        def objective_with_jac(flat_image: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = flat_image.reshape(grid.shape)
            value, candidate_grad, _chi2, _ent = _objective_and_gradient(
                candidate,
                fixed_default,
                data,
                sigma,
                weights,
                grid,
                phases,
                velocity_axis,
                config,
                inclination_deg=inclination_deg,
                gamma=gamma,
            )
            return value, candidate_grad.ravel()

        def callback(flat_image: np.ndarray) -> None:
            candidate = flat_image.reshape(grid.shape)
            value, _grad, candidate_chi2, candidate_ent = _objective_and_gradient(
                candidate,
                fixed_default,
                data,
                sigma,
                weights,
                grid,
                phases,
                velocity_axis,
                config,
                inclination_deg=inclination_deg,
                gamma=gamma,
            )
            objective_history.append(value)
            chi2_history.append(candidate_chi2)
            entropy_history.append(candidate_ent)

        result = minimize(
            objective_with_jac,
            image.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(config.positivity_floor, np.inf),
            callback=callback,
            options={
                "maxiter": maxiter,
                "ftol": max(config.tolerance, 0.0),
                "maxcor": max(1, int(config.lbfgsb_history)),
                "maxls": max(1, int(config.max_line_search)),
            },
        )
        image = np.asarray(result.x, dtype=float).reshape(grid.shape)
        value, _grad, chi2, ent = _objective_and_gradient(
            image,
            fixed_default,
            data,
            sigma,
            weights,
            grid,
            phases,
            velocity_axis,
            config,
            inclination_deg=inclination_deg,
            gamma=gamma,
        )
        if not objective_history or value != objective_history[-1]:
            objective_history.append(value)
            chi2_history.append(chi2)
            entropy_history.append(ent)
        if config.target_chi2 is not None and chi2 <= config.target_chi2:
            break

    return ReconstructionResult(
        image=np.clip(image, config.positivity_floor, None),
        chi2_history=np.asarray(chi2_history, dtype=float),
        objective_history=np.asarray(objective_history, dtype=float),
        entropy_history=np.asarray(entropy_history, dtype=float),
    )


def mem_reconstruct(
    profiles: np.ndarray,
    grid: VelocityGrid3D,
    phases: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    error: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    config: MemConfig | None = None,
    inclination_deg: float = 90.0,
    gamma: float = 0.0,
) -> ReconstructionResult:
    """Run a pure-Python maximum-entropy-style reconstruction.

    The routine is designed to be transparent and testable. It should be used
    with simulations and residual checks before scientific interpretation,
    especially because 3D Doppler tomography is intrinsically underconstrained.
    """

    if config is None:
        config = MemConfig()
    data = np.asarray(profiles, dtype=float)
    if data.ndim != 2:
        raise ValueError("profiles must have shape (nphase, nvelocity).")
    sigma = _validated_error(error, data)
    weights = 1.0 / np.square(sigma)
    image = _initial_image(data, grid, initial, config)

    return _mem_reconstruct_lbfgsb(
        image,
        data,
        sigma,
        weights,
        grid,
        phases,
        velocity_axis,
        config,
        inclination_deg=inclination_deg,
        gamma=gamma,
    )


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
        total = max(float(np.mean(np.sum(np.clip(data, 0.0, None), axis=1))), 1.0)
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
