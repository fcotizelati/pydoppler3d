import numpy as np

from pydoppler3d import DopplerMap, TrailedSpectra, VelocityGrid3D


def test_trailed_spectra_npz_roundtrip(tmp_path):
    phases = np.array([0.0, 0.5])
    velocity = np.linspace(-100.0, 100.0, 5)
    flux = np.arange(10, dtype=float).reshape(2, 5)
    error = np.ones_like(flux) * 0.2
    path = tmp_path / "trail.npz"

    TrailedSpectra(phases, velocity, flux, error=error).to_npz(path)
    loaded = TrailedSpectra.from_npz(path)

    assert np.allclose(loaded.phases, phases)
    assert np.allclose(loaded.velocity, velocity)
    assert np.allclose(loaded.flux, flux)
    assert np.allclose(loaded.error, error)


def test_doppler_map_npz_roundtrip(tmp_path):
    grid = VelocityGrid3D.regular(vlim_xy=100.0, nxy=5, vlim_z=50.0, nz=3)
    image = np.ones(grid.shape)
    path = tmp_path / "map.npz"

    DopplerMap(image, grid).to_npz(path)
    loaded = DopplerMap.from_npz(path)

    assert np.allclose(loaded.image, image)
    assert np.allclose(loaded.grid.vx, grid.vx)
    assert np.allclose(loaded.grid.vz, grid.vz)
