import numpy as np

from pydoppler3d import VelocityGrid3D, line_velocity_axis, radial_velocity


def test_radial_velocity_reduces_to_2d_edge_on_convention():
    assert np.isclose(
        radial_velocity(100.0, 0.0, 999.0, 0.0, inclination_deg=90.0),
        -100.0,
    )
    assert np.isclose(
        radial_velocity(0.0, 200.0, 999.0, 0.25, inclination_deg=90.0),
        200.0,
    )


def test_radial_velocity_includes_vz_face_on():
    assert np.isclose(radial_velocity(100.0, 200.0, 300.0, 0.3, inclination_deg=0.0), 300.0)


def test_regular_grid_shape_and_spacing():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=50.0, nz=3)

    assert grid.shape == (5, 5, 3)
    assert grid.spacing == (50.0, 50.0, 50.0)


def test_line_velocity_axis_relativistic_center():
    velocities = line_velocity_axis(np.array([6562.8]), 6562.8)

    assert np.allclose(velocities, [0.0])
