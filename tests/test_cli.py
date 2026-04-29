import numpy as np

from pydoppler3d import DopplerMap, TrailedSpectra, VelocityGrid3D, project_cube
from pydoppler3d.cli import info_main, reconstruct_main


def test_info_main_reports_backend(capsys):
    info_main([])

    captured = capsys.readouterr()
    assert "pydoppler3d" in captured.out
    assert "pure Python" in captured.out


def test_reconstruct_main_writes_doppler_map(tmp_path):
    input_path = tmp_path / "trail.npz"
    output_path = tmp_path / "map.npz"
    grid = VelocityGrid3D.regular(vlim_xy=80.0, nxy=3, vlim_z=40.0, nz=3)
    phases = np.linspace(0.0, 1.0, 4, endpoint=False)
    velocity = np.linspace(-160.0, 160.0, 49)
    image = np.zeros(grid.shape)
    image[1, 1, 1] = 1.0
    profiles = project_cube(image, grid, phases, velocity, inclination_deg=70.0)
    error = np.full_like(profiles, 0.1)
    TrailedSpectra(phases, velocity, profiles, error=error).to_npz(input_path)

    reconstruct_main(
        [
            str(input_path),
            str(output_path),
            "--nxy",
            "3",
            "--nz",
            "3",
            "--vlim-xy",
            "80",
            "--vlim-z",
            "40",
            "--iterations",
            "2",
            "--alpha",
            "1e-5",
            "--default-updates",
            "1",
            "--inclination",
            "70",
        ]
    )

    doppler_map = DopplerMap.from_npz(output_path)
    assert doppler_map.image.shape == grid.shape
    assert np.all(doppler_map.image >= 0.0)
