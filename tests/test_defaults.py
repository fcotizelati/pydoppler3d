import numpy as np

from pydoppler3d import VelocityGrid3D, gaussian_default, squeezed_default


def test_gaussian_default_preserves_total_flux():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=9, vlim_z=100.0, nz=7)
    cube = np.zeros(grid.shape)
    cube[4, 4, 3] = 10.0

    default = gaussian_default(cube, grid, fwhm_kms=50.0)

    assert default.shape == grid.shape
    assert np.isclose(default.sum(), cube.sum())


def test_squeezed_default_preserves_xy_projection_when_unblurred():
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=100.0, nz=9)
    cube = np.zeros(grid.shape)
    cube[2, 3, 6] = 4.0

    default = squeezed_default(
        cube,
        grid,
        sigma_vz_kms=25.0,
        pull=1.0,
        fwhm_xy_kms=0.0,
        fwhm_iso_kms=50.0,
    )

    assert default.shape == grid.shape
    assert np.isclose(default.sum(), cube.sum())
    assert np.allclose(default.sum(axis=2), cube.sum(axis=2))
