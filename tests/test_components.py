import numpy as np

from pydoppler3d import (
    MapComponent,
    VelocityGrid3D,
    phase_weight,
    project_components,
    project_cube,
)


def test_phase_weight_modulation_terms():
    phases = np.array([0.0, 0.25, 0.5])

    assert np.allclose(phase_weight("constant", phases), 1.0)
    assert np.allclose(phase_weight("negative", phases), -1.0)
    assert np.allclose(phase_weight("sin", phases), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(phase_weight("cos", phases), [1.0, 0.0, -1.0], atol=1e-12)


def test_project_components_matches_single_constant_component():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=50.0, nz=3)
    cube = np.ones(grid.shape)
    phases = np.linspace(0.0, 1.0, 4, endpoint=False)
    velocity_axis = np.linspace(-250.0, 250.0, 101)

    direct = project_cube(cube, grid, phases, velocity_axis)
    combined = project_components(
        [MapComponent(cube, kind="constant")],
        grid,
        phases,
        velocity_axis,
    )

    assert np.allclose(combined, direct)
