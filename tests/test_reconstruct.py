import numpy as np

from pydoppler3d import (
    MemConfig,
    VelocityGrid3D,
    entropy,
    mem_reconstruct,
    project_cube,
)
from pydoppler3d.defaults import gaussian_default


def test_entropy_zero_when_image_matches_default():
    image = np.ones((3, 3, 3))

    assert entropy(image, image) == 0.0


def _small_synthetic_problem():
    grid = VelocityGrid3D.regular(vlim_xy=120.0, nxy=7, vlim_z=60.0, nz=5)
    phases = np.linspace(0.0, 1.0, 6, endpoint=False)
    velocity_axis = np.linspace(-220.0, 220.0, 61)
    image = np.zeros(grid.shape)
    image[2, 4, 3] = 3.0
    profiles = project_cube(image, grid, phases, velocity_axis, inclination_deg=70.0)
    initial = gaussian_default(np.maximum(image, 1e-6), grid, fwhm_kms=120.0)
    return grid, phases, velocity_axis, profiles, initial


def test_mem_reconstruct_lbfgsb_improves_objective_on_small_problem():
    grid, phases, velocity_axis, profiles, initial = _small_synthetic_problem()
    result = mem_reconstruct(
        profiles,
        grid,
        phases,
        velocity_axis,
        initial=initial,
        config=MemConfig(
            iterations=10,
            alpha=1e-5,
            default="gaussian",
            optimizer="lbfgsb",
            default_updates=1,
        ),
        inclination_deg=70.0,
    )

    assert result.image.shape == grid.shape
    assert np.all(result.image >= 0.0)
    assert result.objective_history is not None
    assert result.objective_history[-1] <= result.objective_history[0]
    assert result.chi2_history[-1] <= result.chi2_history[0]


def test_mem_reconstruct_projected_gradient_fallback_improves_objective():
    grid, phases, velocity_axis, profiles, initial = _small_synthetic_problem()
    result = mem_reconstruct(
        profiles,
        grid,
        phases,
        velocity_axis,
        initial=initial,
        config=MemConfig(
            iterations=8,
            step=1e-2,
            alpha=1e-5,
            default="gaussian",
            optimizer="projected_gradient",
            adaptive_step=True,
        ),
        inclination_deg=70.0,
    )

    assert result.objective_history is not None
    assert result.objective_history[-1] <= result.objective_history[0]
