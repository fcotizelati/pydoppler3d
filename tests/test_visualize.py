import numpy as np
import pytest

from pydoppler3d import DopplerMap, TrailedSpectra, VelocityGrid3D
from pydoppler3d.visualize import (
    plot_average_spectrum,
    plot_map_projection,
    plot_map_slices,
    plot_map_volume_html,
    plot_reconstruction,
    plot_residuals,
    plot_trails,
    save_volume_scatter_preview,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


def test_map_pixel_centers_are_physical_velocities(tmp_path, monkeypatch):
    grid = VelocityGrid3D.regular(vlim_xy=100, nxy=5, vlim_z=50, nz=3)
    cube = np.arange(np.prod(grid.shape), dtype=float).reshape(grid.shape)
    captured = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    plot_map_slices(
        DopplerMap(cube, grid), tmp_path / "new" / "slices.png", vz_values=[0]
    )
    image = captured[-1].axes[0].images[0]
    left, right, bottom, top = image.get_extent()
    centers = left + (np.arange(5) + 0.5) * (right - left) / 5
    assert np.allclose(centers, grid.vx)
    assert (bottom, top) == (-125, 125)
    close(captured[-1])


def test_diagnostics_share_scale_and_preserve_negative_data(tmp_path, monkeypatch):
    data = TrailedSpectra(
        [0, 0.4, 0.8],
        [-100, 0, 100],
        np.array([[-2, 1, 2], [-1, 2, 3], [0, 3, 4]]),
        error=np.ones((3, 3)),
    )
    captured = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    plot_reconstruction(data, np.ones((3, 3)), tmp_path / "diag.png", standardized=True)
    fig = captured[-1]
    panels = [
        ax
        for ax in fig.axes
        if ax.get_title() in ("Observed trail", "Reconstructed trail", "Residuals")
    ]
    assert panels[0].collections[0].get_clim() == panels[1].collections[0].get_clim()
    assert np.min(panels[0].collections[0].get_array()) == -2
    lo, hi = panels[2].collections[0].get_clim()
    assert lo == -hi
    close(fig)


def test_plotting_products_and_optional_html(tmp_path):
    grid = VelocityGrid3D.regular(vlim_xy=100, nxy=5, vlim_z=50, nz=3)
    cube = np.arange(np.prod(grid.shape), dtype=float).reshape(grid.shape)
    m = DopplerMap(cube, grid)
    data = TrailedSpectra(
        [0, 0.4, 0.8], [-100, 0, 100], np.arange(9).reshape(3, 3), error=np.ones((3, 3))
    )
    plot_trails(data, tmp_path / "trail.png", cycles=2)
    plot_map_projection(m, tmp_path / "sum.png")
    plot_map_projection(m, tmp_path / "max.png", method="max")
    plot_residuals(
        data, np.ones((3, 3)), tmp_path / "resid.png", standardized=True, cycles=2
    )
    plot_average_spectrum(
        np.arange(10),
        np.arange(10),
        np.arange(10),
        [-1, 0, 1],
        [0, 1, 0],
        tmp_path / "avg.png",
    )
    save_volume_scatter_preview(m, tmp_path / "preview.png")
    assert all(p.stat().st_size > 100 for p in tmp_path.glob("*.png"))
    pytest.importorskip("plotly")
    plot_map_volume_html(m, tmp_path / "new" / "volume.html")
    assert (tmp_path / "new" / "volume.html").stat().st_size > 1000


def test_standardization_requires_errors(tmp_path):
    data = TrailedSpectra([0, 0.5], [-1, 0, 1], np.ones((2, 3)))
    with pytest.raises(ValueError, match="errors"):
        plot_residuals(data, data.flux, tmp_path / "r.png", standardized=True)
