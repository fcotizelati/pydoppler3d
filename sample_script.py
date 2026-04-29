import os
from pathlib import Path

import numpy as np

from pydoppler3d import (
    DopplerMap,
    MemConfig,
    TrailedSpectra,
    VelocityGrid3D,
    copy_test_data,
    mem_reconstruct,
    plot_average_spectrum,
    plot_map_projection,
    plot_map_slices,
    plot_map_volume_html,
    plot_reconstruction,
    plot_residuals,
    plot_trails,
    project_cube,
    save_volume_scatter_preview,
)
from pydoppler3d.pydoppler_compat import load_v834cen_dataset

# Set to True to save figures and run without GUI windows.
SAVE_PNGS = True
# Set to True to display figures interactively at the end.
SHOW_PLOTS = False


def main() -> None:
    workdir = Path.cwd() / "pydoppler3d-workdir"
    outdir = Path.cwd() / "output_images"
    outdir.mkdir(parents=True, exist_ok=True)

    if SAVE_PNGS and not SHOW_PLOTS:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplconfig"))

    # Import the bundled V834 Cen magnetic-CV dataset from the CDS doptomog set.
    copy_test_data(workdir, overwrite=True)
    prepared = load_v834cen_dataset(workdir / "v834cen")

    spectra = prepared.spectra
    spectra.to_npz(outdir / "v834cen_heii4686_trails.npz")
    plot_average_spectrum(
        prepared.wavelength,
        prepared.average_flux,
        prepared.continuum,
        spectra.velocity,
        prepared.average_line_flux,
        outdir / "Average_Spec.png",
        continuum_band=prepared.continuum_band,
    )
    plot_trails(
        spectra,
        outdir / "Trail.png",
        cmap="magma_r",
        cycles=2,
        title="V834 Cen He II 4686 trailed spectra",
    )

    # V834 Cen is a magnetic CV, so it is a better physical demonstration of a
    # 3D velocity cube than the U Gem comparison set. The Vz structure is still
    # regularized and should be checked against default-map sensitivity tests.
    positive_flux = np.clip(spectra.flux, 0.0, None)
    scale = float(np.nanpercentile(positive_flux, 99.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    positive_flux = positive_flux / scale
    error = spectra.error / scale if spectra.error is not None else None
    emission_spectra = TrailedSpectra(
        phases=spectra.phases,
        velocity=spectra.velocity,
        flux=positive_flux,
        error=error,
        exposure=spectra.exposure,
    )

    inclination_deg = 50.0
    grid = VelocityGrid3D.regular(vlim_xy=1_800.0, nxy=41, vlim_z=1_200.0, nz=25)
    result = mem_reconstruct(
        positive_flux,
        grid,
        spectra.phases,
        spectra.velocity,
        error=error,
        config=MemConfig(
            iterations=45,
            step=2e-3,
            alpha=5e-4,
            default="squeezed",
            default_fwhm_kms=200.0,
            squeeze_pull=0.45,
            squeeze_sigma_vz_kms=260.0,
            optimizer="lbfgsb",
            default_updates=2,
        ),
        inclination_deg=inclination_deg,
        gamma=0.0,
    )
    doppler_map = DopplerMap(result.image, grid)
    doppler_map.to_npz(outdir / "Doppler_Map.npz")

    model = project_cube(
        result.image,
        grid,
        spectra.phases,
        spectra.velocity,
        inclination_deg=inclination_deg,
    )
    np.savez_compressed(
        outdir / "Reconstruction.npz",
        phases=spectra.phases,
        velocity=spectra.velocity,
        observed=positive_flux,
        model=model,
        residual=positive_flux - model,
        chi2_history=result.chi2_history,
        objective_history=result.objective_history,
    )

    plot_map_slices(
        doppler_map,
        outdir / "Doppler_Map.png",
        vz_values=[-900.0, -500.0, 0.0, 500.0, 900.0],
        cmap="magma_r",
    )
    plot_map_projection(
        doppler_map,
        outdir / "Doppler_Map_Projection.png",
        method="sum",
        cmap="magma_r",
    )
    save_volume_scatter_preview(
        doppler_map,
        outdir / "Doppler_Map_3D_Preview.png",
    )
    plot_map_volume_html(
        doppler_map,
        outdir / "Doppler_Map_3D.html",
        percentile=97.5,
        surface_count=5,
    )
    plot_reconstruction(
        emission_spectra,
        model,
        outdir / "Reconstruction.png",
        cmap="magma_r",
        cycles=2,
    )
    plot_residuals(
        emission_spectra,
        model,
        outdir / "Residuals.png",
        cmap="magma_r",
        cycles=2,
    )
    print(f"Wrote V834 Cen 3D Doppler products to {outdir}")


if __name__ == "__main__":
    main()
