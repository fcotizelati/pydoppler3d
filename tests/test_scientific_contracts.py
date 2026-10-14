"""Regression tests of scientific contracts, not just optimizer execution."""

import numpy as np
import pytest
from scipy.special import ndtr

from pydoppler3d import (
    MapComponent,
    MemConfig,
    ProjectionOperator,
    VelocityGrid3D,
    back_project_components,
    mem_reconstruct,
    project_components,
    radial_velocity,
)
from pydoppler3d.convolution import gaussian_kernel
from pydoppler3d.defaults import squeezed_default
from pydoppler3d.reconstruct import _Likelihood, entropy


def problem():
    grid = VelocityGrid3D.regular(vlim_xy=100, nxy=5, vlim_z=60, nz=5)
    phases = np.linspace(0, 1, 8, endpoint=False)
    velocity = np.linspace(-180, 180, 61)
    operator = ProjectionOperator(grid, phases, velocity, inclination_deg=60)
    truth = np.zeros(grid.shape)
    truth[1, 3, 3] = 2
    return grid, phases, velocity, operator.forward(truth)


def test_endpoints_conserve_flux_and_transpose():
    grid = VelocityGrid3D.regular(vlim_xy=1, nxy=3, vlim_z=1, nz=3)
    operator = ProjectionOperator(grid, [0], [-1, 0, 1], inclination_deg=0)
    cube = np.ones(grid.shape)
    assert operator.forward(cube).sum() == 27
    assert np.vdot(operator.forward(cube), [[1, 2, 3]]) == np.vdot(
        cube, operator.adjoint([[1, 2, 3]])
    )


@pytest.mark.parametrize("fwhm", [None, 35.0])
@pytest.mark.parametrize("width", [None, [0.08, 0.0, 0.16]])
@pytest.mark.parametrize("cache", [0, 10**7])
def test_response_adjoint_with_truncation(fwhm, width, cache):
    rng = np.random.default_rng(4)
    grid = VelocityGrid3D.regular(vlim_xy=100, nxy=5, vlim_z=60, nz=3)
    operator = ProjectionOperator(
        grid,
        [0.98, 0.2, 0.5],
        np.linspace(-70, 70, 21),
        inclination_deg=51,
        instrumental_fwhm=fwhm,
        exposure=width,
        cache_max_bytes=cache,
    )
    cube = rng.normal(size=grid.shape)
    residual = rng.normal(size=operator.profile_shape)
    assert np.allclose(
        np.vdot(operator.forward(cube), residual),
        np.vdot(cube, operator.adjoint(residual)),
        rtol=1e-12,
        atol=1e-12,
    )
    assert operator.cache_bytes <= cache


def test_finite_exposure_centroid_matches_analytic_sinusoid():
    grid = VelocityGrid3D.regular(vlim_xy=100, nxy=3, vlim_z=50, nz=3)
    cube = np.zeros(grid.shape)
    cube[2, 1, 1] = 1
    phases = np.array([0.0, 0.2, 0.98])
    width = 0.3
    velocity = np.linspace(-200, 200, 401)
    operator = ProjectionOperator(
        grid, phases, velocity, inclination_deg=60, exposure=width, phase_samples=15
    )
    centroid = operator.forward(cube) @ velocity
    expected = -100 * np.sin(np.pi / 3) * np.cos(2 * np.pi * phases) * np.sinc(width)
    assert np.allclose(centroid, expected, atol=1e-10)


def test_modulated_components_transpose_includes_scales_and_exposure():
    rng = np.random.default_rng(5)
    grid, phases, velocity, _ = problem()
    cube = rng.random(grid.shape)
    residual = rng.normal(size=(len(phases), len(velocity)))
    response = dict(inclination_deg=50, exposure=0.1, instrumental_fwhm=20)
    model = project_components(
        [MapComponent(cube, "sin", 2.0)], grid, phases, velocity, **response
    )
    adjoint = back_project_components(
        residual, ["sin"], grid, phases, velocity, scales={"sin": 2.0}, **response
    )["sin"]
    assert np.isclose(
        np.vdot(model, residual), np.vdot(cube, adjoint), rtol=1e-12, atol=1e-12
    )


def test_objective_gradient_with_covariance_and_response():
    rng = np.random.default_rng(19)
    grid = VelocityGrid3D.regular(vlim_xy=50, nxy=3, vlim_z=20, nz=3)
    op = ProjectionOperator(
        grid,
        [0, 0.3],
        np.linspace(-60, 60, 21),
        inclination_deg=50,
        instrumental_fwhm=20,
        exposure=0.1,
    )
    data = rng.normal(size=op.profile_shape)
    covariance = np.tile(np.eye(21) + 0.2 * np.ones((21, 21)), (2, 1, 1))
    likelihood = _Likelihood(data, None, covariance, None, 1e-10)
    image, default = rng.uniform(0.1, 1, grid.shape), rng.uniform(0.1, 1, grid.shape)
    direction = rng.normal(size=grid.shape)

    def objective(x):
        return 0.5 * likelihood.value_gradient(op.forward(x))[0] - 0.03 * entropy(
            x, default
        )

    _, gradient = likelihood.value_gradient(op.forward(image))
    analytic = np.vdot(op.adjoint(gradient) + 0.03 * np.log(image / default), direction)
    h = 1e-6
    numerical = (
        objective(image + h * direction) - objective(image - h * direction)
    ) / (2 * h)
    assert np.isclose(analytic, numerical, rtol=1e-7, atol=1e-8)


def test_covariance_likelihood_matches_direct_inverse():
    data = np.array([[1.0, 2.0, 3.0]])
    cov = np.array([[[2.0, 0.4, 0], [0.4, 1.0, 0.2], [0, 0.2, 3.0]]])
    likelihood = _Likelihood(data, None, cov, None, 1e-10)
    model = np.array([[2.0, 1.0, 4.0]])
    chi2, _ = likelihood.value_gradient(model / likelihood.scale)
    residual = (model - data).ravel()
    assert np.isclose(chi2, residual @ np.linalg.solve(cov[0], residual) / 3)


def test_singular_covariance_records_retained_rank():
    likelihood = _Likelihood(np.ones((1, 3)), None, np.ones((1, 3, 3)), None, 1e-10)
    assert likelihood.n_constraints == 1
    assert np.isfinite(likelihood.value_gradient(np.zeros((1, 3)))[0])


@pytest.mark.parametrize("factor", [1e-14, 1e6])
def test_reconstruction_is_invariant_to_flux_units(factor):
    grid, phases, velocity, data = problem()
    config = MemConfig(iterations=25, default_updates=1, alpha=0.01)
    kwargs = dict(config=config, inclination_deg=60)
    baseline = mem_reconstruct(
        data, grid, phases, velocity, error=np.full_like(data, 0.1), **kwargs
    )
    scaled = mem_reconstruct(
        data * factor,
        grid,
        phases,
        velocity,
        error=np.full_like(data, 0.1 * factor),
        **kwargs,
    )
    assert np.allclose(scaled.image / factor, baseline.image, rtol=1e-6, atol=1e-9)
    assert np.allclose(scaled.chi2_history, baseline.chi2_history, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_errors_rejected_or_explicitly_masked(bad):
    grid, phases, velocity, data = problem()
    error = np.ones_like(data)
    error[0, 0] = bad
    with pytest.raises(ValueError, match="errors"):
        mem_reconstruct(data, grid, phases, velocity, error=error, inclination_deg=60)
    mask = np.ones_like(data, bool)
    mask[0, 0] = False
    data[0, 0] = np.nan
    result = mem_reconstruct(
        data,
        grid,
        phases,
        velocity,
        error=error,
        mask=mask,
        inclination_deg=60,
        config=MemConfig(iterations=1),
    )
    assert np.all(np.isfinite(result.image))
    assert result.n_data == data.size - 1


def test_shape_mismatch_does_not_broadcast():
    grid, phases, velocity, data = problem()
    with pytest.raises(ValueError, match="shape"):
        mem_reconstruct(data[:1], grid, phases, velocity, inclination_deg=60)
    with pytest.raises(ValueError, match="inclination"):
        mem_reconstruct(data, grid, phases, velocity)


@pytest.mark.parametrize("budget,updates", [(1, 2), (5, 2), (9, 4)])
def test_strict_iteration_budget_and_failure_reporting(budget, updates):
    grid, phases, velocity, data = problem()
    result = mem_reconstruct(
        data,
        grid,
        phases,
        velocity,
        inclination_deg=60,
        error=np.full_like(data, 0.1),
        config=MemConfig(
            iterations=budget,
            default_updates=updates,
            tolerance=0,
            gradient_tolerance=0,
        ),
    )
    assert result.n_iterations <= budget
    assert not result.converged
    assert result.status in ("iteration_limit", "optimizer_failed")
    assert not result.success


def test_target_stops_at_first_recorded_crossing():
    grid, phases, velocity, data = problem()
    target = 0.1
    result = mem_reconstruct(
        data,
        grid,
        phases,
        velocity,
        error=np.full_like(data, 0.1),
        inclination_deg=60,
        config=MemConfig(
            iterations=200, default_updates=1, alpha=1e-6, target_chi2=target
        ),
    )
    assert result.target_reached and result.success
    assert not result.converged
    assert result.chi2_history[-1] <= target
    assert np.all(result.chi2_history[:-1] > target)
    assert result.n_iterations < 200


@pytest.mark.parametrize("sigma", [1.0, 1e-200])
def test_narrow_squeezed_default_does_not_underflow(sigma):
    grid = VelocityGrid3D.regular(vlim_xy=1, nxy=3, vlim_z=100, nz=3)
    cube = np.zeros(grid.shape)
    cube[1, 1, :2] = 1
    default = squeezed_default(cube, grid, sigma_vz_kms=sigma, pull=1, fwhm_xy_kms=0)
    assert np.isclose(default.sum(), 2)
    assert np.all(np.isfinite(default))


def test_regular_grid_and_fft_sampling_contract():
    with pytest.raises(ValueError, match="regular"):
        VelocityGrid3D(np.array([0, 1, 3]), np.arange(3), np.arange(3))
    with pytest.raises(ValueError, match="uniform"):
        gaussian_kernel(np.array([0, 1, 3]), 10)


def test_gamma_vertical_translation_degeneracy_and_edge_on_nullspace():
    phases = np.linspace(0, 1, 20)
    original = radial_velocity(100, 200, 300, phases, inclination_deg=50, gamma=12)
    translated = radial_velocity(
        100,
        200,
        350,
        phases,
        inclination_deg=50,
        gamma=12 + 50 * np.cos(np.deg2rad(50)),
    )
    assert np.allclose(original, translated)
    assert np.array_equal(
        radial_velocity(100, 200, 0, phases), radial_velocity(100, 200, 1e10, phases)
    )


def test_independent_gaussian_spot_from_uniform_initial_map():
    grid = VelocityGrid3D.regular(vlim_xy=120, nxy=9, vlim_z=60, nz=5)
    phases = np.linspace(0, 1, 12, endpoint=False)
    velocity = np.linspace(-220, 220, 111)
    inc = np.deg2rad(60)
    # Analytic integrated Gaussian, independent of the discrete projector.
    centroid = np.sin(inc) * (
        -(-60) * np.cos(2 * np.pi * phases) + 30 * np.sin(2 * np.pi * phases)
    ) - 30 * np.cos(inc)
    edges = np.r_[velocity - 2, velocity[-1] + 2]
    data = 3 * np.diff(ndtr((edges[None, :] - centroid[:, None]) / 15), axis=1)
    result = mem_reconstruct(
        data,
        grid,
        phases,
        velocity,
        error=np.full_like(data, 0.01),
        inclination_deg=60,
        instrumental_fwhm=15 * 2 * np.sqrt(2 * np.log(2)),
        config=MemConfig(iterations=180, default_updates=1, alpha=1e-5),
    )
    model = ProjectionOperator(
        grid,
        phases,
        velocity,
        inclination_deg=60,
        instrumental_fwhm=15 * 2 * np.sqrt(2 * np.log(2)),
    ).forward(result.image)
    assert result.chi2_history[-1] < 0.1
    assert np.max(np.abs(model @ velocity / model.sum(axis=1) - centroid)) < 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"alpha": -1},
        {"target_chi2": -1},
        {"positivity_floor": 0},
        {"squeeze_pull": 2},
        {"default": "other"},
        {"cache_max_bytes": -1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        MemConfig(**kwargs)


def test_optimizer_failure_is_preserved(monkeypatch):
    from types import SimpleNamespace

    import pydoppler3d.reconstruct as module

    def fail(fun, x0, **kwargs):
        return SimpleNamespace(
            x=x0, success=False, status=2, message="ABNORMAL_LINE_SEARCH", nit=0
        )

    monkeypatch.setattr(module, "minimize", fail)
    grid, phases, velocity, data = problem()
    result = mem_reconstruct(data, grid, phases, velocity, inclination_deg=60)
    assert not result.success
    assert result.status == "optimizer_failed"
    assert result.message == "ABNORMAL_LINE_SEARCH"
    assert result.meta["optimizer_stages"][0]["status"] == 2
    assert "reconstruct.py" in result.meta["source_sha256"]


def test_initial_target_acceptance_is_not_stationarity():
    grid, phases, velocity, data = problem()
    result = mem_reconstruct(
        data,
        grid,
        phases,
        velocity,
        inclination_deg=60,
        config=MemConfig(target_chi2=1e6),
    )
    assert result.success and result.target_reached
    assert not result.converged and result.n_iterations == 0


def test_landweber_development_status_and_final_residual():
    from pydoppler3d import LandweberConfig, landweber_reconstruct

    grid, phases, velocity, data = problem()
    result = landweber_reconstruct(
        data,
        grid,
        phases,
        velocity,
        inclination_deg=60,
        config=LandweberConfig(iterations=5, step=0.001),
    )
    model = ProjectionOperator(grid, phases, velocity, inclination_deg=60).forward(
        result.image
    )
    assert result.chi2_history[-1] == pytest.approx(np.mean((model - data) ** 2))
    assert result.chi2_history[-1] < result.chi2_history[0]
    assert result.status == "fixed_iterations" and not result.converged
