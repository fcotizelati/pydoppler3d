import numpy as np

from pydoppler3d import VelocityGrid3D, back_project, project_cube
from pydoppler3d.convolution import convolve_profiles_fft


def test_project_cube_conserves_flux_when_velocity_axis_covers_all_samples():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=50.0, nz=3)
    cube = np.ones(grid.shape)
    phases = np.linspace(0.0, 1.0, 8, endpoint=False)
    velocity_axis = np.linspace(-250.0, 250.0, 101)

    profiles = project_cube(cube, grid, phases, velocity_axis, inclination_deg=90.0)

    assert profiles.shape == (phases.size, velocity_axis.size)
    assert np.allclose(profiles.sum(axis=1), cube.sum())


def test_back_project_returns_grid_shape():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=50.0, nz=3)
    phases = np.linspace(0.0, 1.0, 4, endpoint=False)
    velocity_axis = np.linspace(-250.0, 250.0, 101)
    residuals = np.ones((phases.size, velocity_axis.size))

    cube = back_project(residuals, grid, phases, velocity_axis)

    assert cube.shape == grid.shape
    assert np.all(cube >= 0.0)


def test_project_and_back_project_are_adjoint():
    rng = np.random.default_rng(123)
    grid = VelocityGrid3D.regular(vlim_xy=120.0, nxy=5, vlim_z=80.0, nz=3)
    phases = np.linspace(0.0, 1.0, 6, endpoint=False)
    velocity_axis = np.linspace(-250.0, 250.0, 121)
    cube = rng.random(grid.shape)
    residuals = rng.normal(size=(phases.size, velocity_axis.size))

    projected = project_cube(
        cube,
        grid,
        phases,
        velocity_axis,
        inclination_deg=72.0,
        gamma=17.0,
    )
    back_projected = back_project(
        residuals,
        grid,
        phases,
        velocity_axis,
        inclination_deg=72.0,
        gamma=17.0,
    )

    assert np.allclose(
        np.vdot(projected, residuals),
        np.vdot(cube, back_projected),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fft_profile_convolution_preserves_flux_approximately():
    axis = np.linspace(-100.0, 100.0, 101)
    profiles = np.zeros((2, axis.size))
    profiles[:, axis.size // 2] = 1.0

    blurred = convolve_profiles_fft(profiles, axis, fwhm=10.0)

    assert blurred.shape == profiles.shape
    assert np.allclose(blurred.sum(axis=1), profiles.sum(axis=1), rtol=1e-4)
