import numpy as np

from pydoppler3d import VelocityGrid3D, back_project, project_cube


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
