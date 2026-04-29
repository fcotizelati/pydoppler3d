from pydoppler3d import copy_test_data
from pydoppler3d.pydoppler_compat import load_pydoppler_dataset, load_v834cen_dataset


def test_load_bundled_pydoppler_style_dataset(tmp_path):
    copy_test_data(tmp_path)

    prepared = load_pydoppler_dataset(tmp_path / "ugem99")

    assert prepared.spectra.phases.shape == (28,)
    assert prepared.spectra.flux.shape[0] == 28
    assert prepared.spectra.velocity.ndim == 1
    assert prepared.spectra.error is not None
    assert prepared.filenames[0] == "txhugem4004"


def test_load_bundled_v834cen_dataset(tmp_path):
    copy_test_data(tmp_path)

    prepared = load_v834cen_dataset(tmp_path / "v834cen")

    assert prepared.spectra.phases.shape == (59,)
    assert prepared.spectra.flux.shape[0] == 59
    assert prepared.spectra.velocity.ndim == 1
    assert prepared.spectra.error is not None
    assert prepared.filenames[0] == "mcv001.txt"
    assert prepared.lam0 == 4686.0
