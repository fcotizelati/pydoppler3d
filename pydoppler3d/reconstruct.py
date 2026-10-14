"""Scale-invariant entropy-regularized inversion with explicit diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, minimize

from .defaults import gaussian_default, squeezed_default
from .geometry import VelocityGrid3D
from .project import ProjectionOperator
from .provenance import runtime_provenance


@dataclass(frozen=True)
class MemConfig:
    """Settings for 0.5 * mean_chi2 - alpha * normalized-unit entropy.

    Image and data are internally divided by a common flux scale. The floor
    and alpha apply in these internal units; outputs retain input flux units.
    target_chi2 is a discrepancy stopping rule, not a constrained MEM solver.
    iterations is a strict total budget across all default updates.
    """

    iterations: int = 100
    alpha: float = 1e-3
    target_chi2: float | None = None
    default: Literal["gaussian", "squeezed"] = "squeezed"
    default_fwhm_kms: float = 200.0
    squeeze_pull: float = 0.5
    squeeze_sigma_vz_kms: float = 100.0
    positivity_floor: float = 1e-12
    tolerance: float = 1e-7
    gradient_tolerance: float = 1e-5
    default_tolerance: float = 1e-3
    default_updates: int = 2
    lbfgsb_history: int = 10
    max_line_search: int = 20
    cache_max_bytes: int = 64 * 1024**2
    covariance_rtol: float = 1e-10

    def __post_init__(self):
        for name in (
            "iterations",
            "default_updates",
            "lbfgsb_history",
            "max_line_search",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isfinite(value)
                or int(value) != value
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "alpha",
            "default_fwhm_kms",
            "tolerance",
            "gradient_tolerance",
            "default_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("positivity_floor", "squeeze_sigma_vz_kms"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not 0 <= self.squeeze_pull <= 1 or not 0 < self.covariance_rtol < 1:
            raise ValueError("invalid squeeze_pull or covariance_rtol.")
        if self.target_chi2 is not None and (
            not np.isfinite(self.target_chi2) or self.target_chi2 <= 0
        ):
            raise ValueError("target_chi2 must be positive and finite.")
        if self.default not in ("gaussian", "squeezed"):
            raise ValueError("default must be gaussian or squeezed.")
        if (
            not np.isfinite(self.cache_max_bytes)
            or int(self.cache_max_bytes) != self.cache_max_bytes
            or self.cache_max_bytes < 0
        ):
            raise ValueError("cache_max_bytes must be a nonnegative integer.")


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
    success: bool = False
    converged: bool = False
    status: str = "not_assessed"
    message: str = ""
    target_reached: bool | None = None
    default_converged: bool | None = None
    n_iterations: int = 0
    n_evaluations: int = 0
    projected_gradient_norm: float | None = None
    normalization: float = 1.0
    n_data: int = 0
    n_constraints: int = 0
    history_stage: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def diagnostics(self) -> dict:
        """JSON-compatible run provenance and diagnostics, excluding the cube."""
        return {
            **self.meta,
            "success": self.success,
            "converged": self.converged,
            "status": self.status,
            "message": self.message,
            "target_reached": self.target_reached,
            "default_converged": self.default_converged,
            "n_iterations": self.n_iterations,
            "n_evaluations": self.n_evaluations,
            "projected_gradient_norm": self.projected_gradient_norm,
            "normalization": self.normalization,
            "n_data": self.n_data,
            "n_constraints": self.n_constraints,
            "mean_chi2": float(self.chi2_history[-1]),
            "chi2_history": self.chi2_history.tolist(),
            "objective_history": None
            if self.objective_history is None
            else self.objective_history.tolist(),
            "history_stage": None
            if self.history_stage is None
            else self.history_stage.tolist(),
        }


def entropy(image, default, *, floor=1e-12) -> float:
    """Generalized relative entropy; zero at image=default (fixed default)."""
    image, default = np.asarray(image, float), np.asarray(default, float)
    if image.shape != default.shape or not np.isfinite(floor) or floor <= 0:
        raise ValueError(
            "entropy requires matching shapes and a positive finite floor."
        )
    if (
        not np.all(np.isfinite(image))
        or not np.all(np.isfinite(default))
        or np.any(image < 0)
        or np.any(default < 0)
    ):
        raise ValueError("entropy inputs must be finite and nonnegative.")
    image, default = np.maximum(image, floor), np.maximum(default, floor)
    return float(np.sum(image - default - image * (np.log(image) - np.log(default))))


def _default_map(image, grid, config):
    if config.default == "gaussian":
        default = gaussian_default(image, grid, fwhm_kms=config.default_fwhm_kms)
    else:
        default = squeezed_default(
            image,
            grid,
            sigma_vz_kms=config.squeeze_sigma_vz_kms,
            pull=config.squeeze_pull,
            fwhm_xy_kms=config.default_fwhm_kms,
            fwhm_iso_kms=config.default_fwhm_kms,
        )
    return np.maximum(default, config.positivity_floor)


class _Likelihood:
    """Whiten retained observations; covariance may have redundant modes.

    Singular covariance is projected onto eigenmodes above covariance_rtol.
    The null space is excluded, not treated as zero-noise constraints. This
    choice is recorded; inspect model residuals outside the retained modes.
    """

    def __init__(self, data, error, covariance, mask, rtol):
        self.mask = np.ones(data.shape, bool) if mask is None else np.asarray(mask)
        if (
            self.mask.dtype != bool
            or self.mask.shape != data.shape
            or not np.any(self.mask)
        ):
            raise ValueError(
                "mask must be boolean, match data, and include at least one sample."
            )
        if not np.all(np.isfinite(data[self.mask])):
            raise ValueError("unmasked profiles must be finite.")
        sigma = None if error is None else np.asarray(error, float)
        if sigma is not None and (
            sigma.shape != data.shape
            or not np.all(np.isfinite(sigma[self.mask]))
            or np.any(sigma[self.mask] <= 0)
        ):
            raise ValueError(
                "unmasked errors must be positive, finite, and match profiles."
            )
        cov = None if covariance is None else np.asarray(covariance, float)
        if cov is not None and cov.shape != (
            data.shape[0],
            data.shape[1],
            data.shape[1],
        ):
            raise ValueError("covariance must have shape (nphase,nvelocity,nvelocity).")
        amplitude = float(np.percentile(np.abs(data[self.mask]), 99))
        noise = float(np.median(sigma[self.mask])) if sigma is not None else 0.0
        if cov is not None:
            diag = np.diagonal(cov, axis1=1, axis2=2)[self.mask]
            if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
                raise ValueError(
                    "included covariance diagonal must be finite and positive."
                )
            noise = float(np.median(np.sqrt(diag)))
            if sigma is not None and not np.allclose(
                sigma[self.mask] ** 2, diag, rtol=1e-6, atol=0
            ):
                raise ValueError("error squared must agree with covariance diagonal.")
        self.scale = max(amplitude, noise) or 1.0
        self.data = np.where(self.mask, data, 0) / self.scale
        self.n_data = int(self.mask.sum())
        self.n_constraints = 0
        self.whiteners = []
        self.weights = None
        if cov is None:
            normalized_sigma = (
                np.ones_like(data)
                if sigma is None
                else np.where(self.mask, sigma, self.scale) / self.scale
            )
            self.weights = np.where(self.mask, 1 / normalized_sigma**2, 0)
            self.n_constraints = self.n_data
            self.kind = (
                "unweighted_normalized" if sigma is None else "diagonal_gaussian"
            )
        else:
            self.kind = "covariance_gaussian_retained_modes"
            for j, include in enumerate(self.mask):
                indices = np.flatnonzero(include)
                block = cov[j][np.ix_(indices, indices)] / self.scale**2
                if not indices.size:
                    self.whiteners.append((indices, np.empty((0, 0))))
                    continue
                if not np.all(np.isfinite(block)) or not np.allclose(
                    block, block.T, rtol=1e-10, atol=np.max(np.abs(block)) * 1e-12
                ):
                    raise ValueError(
                        "covariance must be finite and symmetric on included samples."
                    )
                values, vectors = np.linalg.eigh((block + block.T) / 2)
                threshold = rtol * values[-1]
                if values[-1] <= 0 or values[0] < -threshold:
                    raise ValueError("covariance must be positive semidefinite.")
                keep = values > threshold
                whitener = (vectors[:, keep] / np.sqrt(values[keep])).T
                self.whiteners.append((indices, whitener))
                self.n_constraints += int(keep.sum())
        if not self.n_constraints:
            raise ValueError("no informative covariance modes remain.")

    def value_gradient(self, model):
        residual = model - self.data
        if self.weights is not None:
            precision_residual = residual * self.weights
            chi2 = float(np.sum(residual * precision_residual))
        else:
            precision_residual = np.zeros_like(residual)
            chi2 = 0.0
            for j, (indices, whitener) in enumerate(self.whiteners):
                white = whitener @ residual[j, indices]
                chi2 += float(white @ white)
                precision_residual[j, indices] = whitener.T @ white
        return chi2 / self.n_constraints, precision_residual / self.n_constraints


class _TargetReached(Exception):
    pass


def mem_reconstruct(
    profiles,
    grid: VelocityGrid3D,
    phases,
    velocity_axis,
    *,
    error=None,
    covariance=None,
    mask=None,
    initial=None,
    config=None,
    inclination_deg=None,
    gamma=0.0,
    exposure=None,
    instrumental_fwhm=None,
    phase_samples=7,
) -> ReconstructionResult:
    """Fit signed observations with a nonnegative emissivity cube.

    Inclination must be explicit. Inspect success, converged, target_reached,
    and residuals separately. success includes deliberate discrepancy stopping;
    it is not evidence that a unique or correct 3D flow has been recovered.
    """
    config = MemConfig() if config is None else config
    if inclination_deg is None:
        raise ValueError("inclination_deg is required for a 3D reconstruction.")
    if np.isclose(inclination_deg, 90.0, atol=1e-12, rtol=0):
        warnings.warn(
            "At inclination 90 degrees Vz is completely unconstrained.",
            UserWarning,
            stacklevel=2,
        )
    elif inclination_deg in (0.0, 180.0):
        warnings.warn(
            "At face-on inclination Vx and Vy are completely unconstrained.",
            UserWarning,
            stacklevel=2,
        )
    operator = ProjectionOperator(
        grid,
        phases,
        velocity_axis,
        inclination_deg=inclination_deg,
        gamma=gamma,
        exposure=exposure,
        instrumental_fwhm=instrumental_fwhm,
        phase_samples=phase_samples,
        cache_max_bytes=config.cache_max_bytes,
    )
    data = np.asarray(profiles, float)
    if data.shape != operator.profile_shape:
        raise ValueError(f"profiles must have shape {operator.profile_shape}.")
    likelihood = _Likelihood(data, error, covariance, mask, config.covariance_rtol)
    if initial is None:
        # Positivity here initializes the model; observations remain signed.
        total = float(np.mean(np.maximum(likelihood.data, 0).sum(axis=1)))
        image = np.full(
            grid.shape, max(total / np.prod(grid.shape), config.positivity_floor)
        )
    else:
        image = np.asarray(initial, float) / likelihood.scale
        if (
            image.shape != grid.shape
            or not np.all(np.isfinite(image))
            or np.any(image < 0)
        ):
            raise ValueError(
                "initial must be a finite nonnegative cube matching the grid."
            )
        image = np.maximum(image, config.positivity_floor)
    chi_history, objective_history, entropy_history, stage_history = [], [], [], []
    records = []
    nit, nfev = 0, 0
    status, message = "iteration_limit", "Total iteration budget exhausted."
    success = converged = False
    default_converged = None if config.default_updates == 1 else False
    gradient_norm = None
    stage_count = min(config.iterations, config.default_updates)
    budgets = [
        config.iterations // stage_count + (j < config.iterations % stage_count)
        for j in range(stage_count)
    ]
    target_reached = None if config.target_chi2 is None else False
    for stage, budget in enumerate(budgets):
        fixed_default = _default_map(image, grid, config)
        cache_x, cache_value = None, None

        def evaluate(flat):
            nonlocal cache_x, cache_value, nfev
            if cache_x is not None and np.array_equal(flat, cache_x):
                return cache_value
            candidate = flat.reshape(grid.shape)
            chi2, profile_grad = likelihood.value_gradient(operator.forward(candidate))
            ent = entropy(candidate, fixed_default, floor=config.positivity_floor)
            gradient = operator.adjoint(profile_grad) + config.alpha * (
                np.log(candidate) - np.log(fixed_default)
            )
            value = 0.5 * chi2 - config.alpha * ent
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError(
                    "nonfinite objective/gradient; check data scales and errors."
                )
            cache_x = flat.copy()
            cache_value = (value, gradient.ravel(), chi2, ent)
            nfev += 1
            return cache_value

        def record(flat):
            value, gradient, chi2, ent = evaluate(flat)
            chi_history.append(chi2)
            objective_history.append(value)
            entropy_history.append(ent)
            stage_history.append(stage)
            return value, gradient, chi2, ent

        current = image.ravel().copy()
        _, gradient, chi2, _ = record(current)
        if config.target_chi2 is not None and chi2 <= config.target_chi2:
            status, message, target_reached, success = (
                "target_reached",
                "Discrepancy target reached; stationarity is not implied.",
                True,
                True,
            )
            break
        stage_nit = 0

        def callback(flat):
            nonlocal current, nit, stage_nit
            current = flat.copy()
            nit += 1
            stage_nit += 1
            _, _, chi2, _ = record(current)
            if config.target_chi2 is not None and chi2 <= config.target_chi2:
                raise _TargetReached

        try:
            opt = minimize(
                lambda x: evaluate(x)[:2],
                current,
                jac=True,
                method="L-BFGS-B",
                bounds=Bounds(config.positivity_floor, np.inf),
                callback=callback,
                options={
                    "maxiter": int(budget),
                    "ftol": config.tolerance,
                    "gtol": config.gradient_tolerance,
                    "maxcor": config.lbfgsb_history,
                    "maxls": config.max_line_search,
                },
            )
            current = np.asarray(opt.x)
            optimizer_success = bool(opt.success)
            records.append(
                {
                    "stage": stage,
                    "success": optimizer_success,
                    "status": int(opt.status),
                    "message": str(opt.message),
                    "iterations": int(opt.nit),
                }
            )
        except _TargetReached:
            optimizer_success = False
            status, message, target_reached, success = (
                "target_reached",
                "Discrepancy target reached; stationarity is not implied.",
                True,
                True,
            )
            records.append(
                {
                    "stage": stage,
                    "success": False,
                    "status": "target_reached",
                    "iterations": stage_nit,
                }
            )
        image = current.reshape(grid.shape)
        value, gradient, chi2, ent = evaluate(current)
        if stage_nit == 0 or value != objective_history[-1]:
            record(current)
        projected = gradient.copy()
        projected[
            (current <= config.positivity_floor * (1 + 1e-8)) & (projected > 0)
        ] = 0
        gradient_norm = float(np.max(np.abs(projected)))
        if target_reached:
            break
        next_default = _default_map(image, grid, config)
        relative_change = float(
            np.linalg.norm(next_default - fixed_default)
            / max(np.linalg.norm(fixed_default), np.finfo(float).tiny)
        )
        if config.default_updates > 1:
            default_converged = relative_change <= config.default_tolerance
        converged = optimizer_success and (default_converged is not False)
        if converged:
            success = config.target_chi2 is None or chi2 <= config.target_chi2
            status = "converged" if success else "target_not_reached"
            message = (
                "Optimizer and configured default criterion converged."
                if success
                else "Optimizer converged but requested fit target was not reached."
            )
            break
        if not optimizer_success and int(opt.status) != 1:
            status, message = "optimizer_failed", str(opt.message)
            break
        if optimizer_success:
            status, message = (
                "default_not_converged",
                "Fixed-default fit converged; outer default did not converge within the configured stages.",
            )
        else:
            status, message = (
                "iteration_limit",
                "An optimizer stage exhausted its iteration budget; convergence is not established.",
            )
    # Record the final gradient even if the starting image met the target.
    if gradient_norm is None:
        projected = gradient.copy()
        projected[
            (image.ravel() <= config.positivity_floor * (1 + 1e-8)) & (projected > 0)
        ] = 0
        gradient_norm = float(np.max(np.abs(projected)))
    return ReconstructionResult(
        image=image * likelihood.scale,
        chi2_history=np.asarray(chi_history),
        objective_history=np.asarray(objective_history),
        entropy_history=np.asarray(entropy_history),
        success=success,
        converged=converged,
        status=status,
        message=message,
        target_reached=target_reached,
        default_converged=default_converged,
        n_iterations=nit,
        n_evaluations=nfev,
        projected_gradient_norm=gradient_norm,
        normalization=likelihood.scale,
        n_data=likelihood.n_data,
        n_constraints=likelihood.n_constraints,
        history_stage=np.asarray(stage_history, int),
        meta={
            **runtime_provenance(),
            "config": asdict(config),
            "inclination_deg": float(inclination_deg),
            "gamma": float(gamma),
            "instrumental_fwhm": instrumental_fwhm,
            "phase_samples": int(phase_samples),
            "exposure": None if exposure is None else np.asarray(exposure).tolist(),
            "velocity_convention": "right_handed_minus_vz",
            "image_units": "flux_per_voxel_in_input_bin_flux_units",
            "likelihood": likelihood.kind,
            "statistic": "chi2_per_retained_constraint",
            "optimizer_stages": records,
            "geometry_cache_bytes": operator.cache_bytes,
        },
    )


def landweber_reconstruct(
    profiles,
    grid,
    phases,
    velocity_axis,
    *,
    initial=None,
    config=None,
    inclination_deg=None,
    **response,
):
    """Development-only projected least-squares iteration, with explicit status."""
    config = LandweberConfig() if config is None else config
    if inclination_deg is None:
        raise ValueError("inclination_deg is required.")
    if (
        int(config.iterations) != config.iterations
        or config.iterations < 1
        or not np.isfinite(config.step)
        or config.step <= 0
        or not 0 <= config.default_weight <= 1
    ):
        raise ValueError("invalid Landweber configuration.")
    operator = ProjectionOperator(
        grid, phases, velocity_axis, inclination_deg=inclination_deg, **response
    )
    data = np.asarray(profiles, float)
    if data.shape != operator.profile_shape or not np.all(np.isfinite(data)):
        raise ValueError("profiles must be finite and match phase/velocity shape.")
    image = (
        np.zeros(grid.shape) if initial is None else np.array(initial, float, copy=True)
    )
    if image.shape != grid.shape or not np.all(np.isfinite(image)):
        raise ValueError("initial must be finite and match the grid.")
    history = []
    for _ in range(config.iterations):
        residual = data - operator.forward(image)
        history.append(float(np.mean(residual**2)))
        image += config.step * operator.adjoint(residual)
        if config.nonnegative:
            image = np.maximum(image, 0)
        if config.default_weight:
            default = gaussian_default(image, grid, fwhm_kms=config.default_fwhm_kms)
            image = (
                1 - config.default_weight
            ) * image + config.default_weight * default
    history.append(float(np.mean((data - operator.forward(image)) ** 2)))
    return ReconstructionResult(
        image,
        np.asarray(history),
        status="fixed_iterations",
        message="Development iteration; no convergence claim.",
        n_iterations=config.iterations,
    )
