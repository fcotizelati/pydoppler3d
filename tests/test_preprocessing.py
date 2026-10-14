import numpy as np
import pytest

from pydoppler3d.pydoppler_compat import _interpolation_matrix, load_pydoppler_dataset


def test_interpolation_variance_and_covariance():
    transform = _interpolation_matrix(
        np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.0, 1.5])
    )
    covariance = transform @ transform.T
    assert np.allclose(np.diag(covariance), [0.5, 1.0, 0.5])
    assert covariance[0, 2] == 0.25


def write_spectra(tmp_path, flux, error=True, gamma=0):
    wave = np.linspace(4900, 5100, 41)
    names = []
    for j, row in enumerate(flux):
        name = f"spec{j}.txt"
        names.append(name)
        columns = [wave, row]
        if error:
            columns.append(np.full_like(wave, 0.3))
        np.savetxt(tmp_path / name, np.column_stack(columns))
    (tmp_path / "phases").write_text(
        "".join(f"{name} {j / len(names)}\n" for j, name in enumerate(names))
    )
    return dict(
        base_dir=tmp_path,
        list_file="phases",
        lam0=5000,
        delw=20,
        gamma=gamma,
        continuum_band=[4900, 4965, 5035, 5100],
        poly_degree=1,
    )


def test_monte_carlo_continuum_and_interpolation_covariance(tmp_path):
    rng = np.random.default_rng(654)
    wave = np.linspace(4900, 5100, 41)
    flux = 2 + 0.003 * (wave - 5000) + rng.normal(0, 0.3, (450, 41))
    prepared = load_pydoppler_dataset(**write_spectra(tmp_path, flux))
    empirical = np.cov(prepared.spectra.flux, rowvar=False)
    predicted = prepared.spectra.covariance.mean(axis=0)
    assert np.allclose(np.diag(empirical), np.diag(predicted), rtol=0.18)
    # Shared fitted continuum induces correlation between well-separated bins.
    assert predicted[0, -1] > 0
    assert np.isclose(empirical[0, -1], predicted[0, -1], atol=0.008)
    assert np.allclose(
        prepared.spectra.error**2,
        np.diagonal(prepared.spectra.covariance, axis1=1, axis2=2),
    )


def test_gamma_continuum_display_uses_same_frame(tmp_path):
    wave = np.linspace(4900, 5100, 41)
    flux = np.tile(2 + 0.003 * (wave - 5000), (2, 1))
    prepared = load_pydoppler_dataset(**write_spectra(tmp_path, flux, gamma=300))
    assert np.allclose(prepared.continuum, prepared.average_flux, atol=1e-12)
    assert np.max(np.abs(prepared.spectra.flux)) < 1e-12
    assert prepared.spectra.meta["gamma_removed_kms"] == 300


def test_missing_noise_is_not_fabricated(tmp_path):
    wave = np.linspace(4900, 5100, 41)
    with pytest.raises(ValueError, match="cannot estimate noise"):
        load_pydoppler_dataset(
            **write_spectra(
                tmp_path, np.tile(2 + 0.003 * (wave - 5000), (2, 1)), error=False
            )
        )


def test_partial_exposure_information_is_rejected(tmp_path):
    kwargs = write_spectra(tmp_path, np.ones((2, 41)))
    (tmp_path / "phases").write_text("spec0.txt 0 .02\nspec1.txt .5\n")
    with pytest.raises(ValueError, match="every spectrum"):
        load_pydoppler_dataset(**kwargs)


def test_explicit_diagonal_approximation_is_recorded(tmp_path):
    kwargs = write_spectra(tmp_path, np.ones((2, 41)))
    prepared = load_pydoppler_dataset(**kwargs, propagate_covariance=False)
    assert prepared.spectra.covariance is None
    assert prepared.spectra.meta["covariance_policy"] == "diagonal_approximation"


def test_duplicate_wavelengths_require_noise_information(tmp_path):
    from pydoppler3d.pydoppler_compat import _read_spectrum

    path = tmp_path / "duplicates.txt"
    np.savetxt(path, [[5000, 1], [5000, 3], [5001, 2]])
    with pytest.raises(ValueError, match="duplicate"):
        _read_spectrum(path)
    np.savetxt(path, [[5000, 1, 1], [5000, 3, 1], [5001, 2, 1]])
    wave, flux, error = _read_spectrum(path)
    assert np.array_equal(wave, [5000, 5001])
    assert np.allclose(flux, [2, 2])
    assert np.allclose(error, [1 / np.sqrt(2), 1])
