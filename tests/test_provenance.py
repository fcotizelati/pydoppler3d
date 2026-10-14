import json

import numpy as np
import pytest

from pydoppler3d import DopplerMap, TrailedSpectra, VelocityGrid3D
from pydoppler3d.cli import reconstruct_main


def spectra():
    covariance = np.tile(np.eye(5) + 0.1 * np.ones((5, 5)), (2, 1, 1)) * 1e-28
    return TrailedSpectra(
        [0, 0.5],
        np.linspace(-100, 100, 5),
        np.ones((2, 5)) * 1e-14,
        error=np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)),
        exposure=np.array([0.02, 0.03]),
        covariance=covariance,
        mask=np.ones((2, 5), bool),
        meta={
            "gamma_removed_kms": 12,
            "inclination_deg": 50,
            "units": "erg / s",
            "nested": {"version": 2},
        },
    )


@pytest.mark.parametrize("kind", ["npz", "fits"])
def test_scientific_metadata_and_covariance_roundtrip(tmp_path, kind):
    if kind == "fits":
        pytest.importorskip("astropy")
    original = spectra()
    path = tmp_path / "new" / f"data.{kind}"
    getattr(original, f"to_{kind}")(path)
    loaded = getattr(TrailedSpectra, f"from_{kind}")(path)
    assert loaded.meta == original.meta
    for field in ("flux", "error", "exposure", "covariance", "mask"):
        assert np.array_equal(getattr(loaded, field), getattr(original, field))


def test_map_metadata_roundtrip(tmp_path):
    grid = VelocityGrid3D.regular(nxy=3, nz=3)
    m = DopplerMap(
        np.ones(grid.shape),
        grid,
        {"velocity_convention": "right_handed_minus_vz", "config": {"alpha": 0.1}},
    )
    m.to_npz(tmp_path / "map.npz")
    assert DopplerMap.from_npz(tmp_path / "map.npz").meta == m.meta


def test_legacy_data_loads_without_pickle(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(
        path, phases=[0, 0.5], velocities=[-1, 0, 1], profiles=np.ones((2, 3)), error=[]
    )
    loaded = TrailedSpectra.from_npz(path)
    assert loaded.meta == {}
    assert loaded.error is None


def test_cli_raw_tiny_flux_and_failure_exit(tmp_path, capsys):
    data = spectra()
    source = tmp_path / "data.npz"
    data.to_npz(source)
    output = tmp_path / "out" / "map.npz"
    args = [
        str(source),
        str(output),
        "--inclination",
        "50",
        "--nxy",
        "3",
        "--nz",
        "3",
        "--vlim-xy",
        "100",
        "--vlim-z",
        "50",
        "--iterations",
        "1",
        "--default-updates",
        "1",
    ]
    with pytest.raises(SystemExit) as exc:
        reconstruct_main(args)
    assert exc.value.code == 2
    saved = DopplerMap.from_npz(output)
    diagnostics = json.loads(output.with_suffix(".json").read_text())
    assert saved.image.max() < 1e-12
    assert not diagnostics["success"]
    assert diagnostics["input_metadata"] == data.meta
    assert len(diagnostics["input_sha256"]) == 64
    assert diagnostics["mean_chi2"] < 100
    reconstruct_main(args + ["--allow-incomplete"])
    assert "iteration_limit" in capsys.readouterr().out


def test_masked_nan_roundtrip(tmp_path):
    data = spectra()
    mask = data.mask.copy()
    mask[0, 0] = False
    flux = data.flux.copy()
    flux[0, 0] = np.nan
    obj = TrailedSpectra(data.phases, data.velocity, flux, error=data.error, mask=mask)
    obj.to_npz(tmp_path / "masked.npz")
    assert not TrailedSpectra.from_npz(tmp_path / "masked.npz").mask[0, 0]


def test_cli_will_not_overwrite_input(tmp_path):
    source = tmp_path / "data.npz"
    spectra().to_npz(source)
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        reconstruct_main([str(source), str(source), "--inclination", "50"])
    assert source.read_bytes() == original
