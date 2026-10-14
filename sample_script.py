"""Reproducible V834 Cen demonstration; run `python sample_script.py --help`."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pydoppler3d import (
    DopplerMap,
    MemConfig,
    ProjectionOperator,
    VelocityGrid3D,
    __version__,
    copy_test_data,
    load_v834cen_dataset,
    mem_reconstruct,
    plot_average_spectrum,
    plot_map_projection,
    plot_map_slices,
    plot_map_volume_html,
    plot_reconstruction,
    plot_residuals,
    plot_trails,
    save_volume_scatter_preview,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output_images"))
    parser.add_argument("--workdir", type=Path, default=Path("pydoppler3d-workdir"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--nxy", type=int, default=31)
    parser.add_argument("--nz", type=int, default=21)
    parser.add_argument(
        "--no-plots", action="store_true", help="Write numerical products only."
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return zero for a recorded partial demonstration fit.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(args.workdir.resolve() / ".mplconfig"))
    args.output.mkdir(parents=True, exist_ok=True)
    copy_test_data(args.workdir)
    prepared = load_v834cen_dataset(args.workdir / "v834cen")
    spectra = prepared.spectra
    input_path = args.output / "v834cen_heii4686_trails.npz"
    spectra.to_npz(input_path)
    grid = VelocityGrid3D.regular(
        vlim_xy=1800.0, nxy=args.nxy, vlim_z=1200.0, nz=args.nz
    )
    config = MemConfig(
        iterations=args.iterations,
        alpha=5e-4,
        default="squeezed",
        squeeze_pull=0.45,
        squeeze_sigma_vz_kms=260.0,
        default_updates=2,
    )
    response = dict(inclination_deg=50.0, gamma=0.0, exposure=spectra.exposure)
    result = mem_reconstruct(
        spectra.flux,
        grid,
        spectra.phases,
        spectra.velocity,
        error=spectra.error,
        covariance=spectra.covariance,
        mask=spectra.mask,
        config=config,
        **response,
    )
    diagnostics = {
        **result.diagnostics(),
        "package_version": __version__,
        "sample_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_metadata": spectra.meta,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    doppler_map = DopplerMap(result.image, grid, diagnostics)
    doppler_map.to_npz(args.output / "Doppler_Map.npz")
    model = ProjectionOperator(
        grid, spectra.phases, spectra.velocity, **response
    ).forward(result.image)
    np.savez_compressed(
        args.output / "Reconstruction.npz",
        phases=spectra.phases,
        velocity=spectra.velocity,
        observed=spectra.flux,
        model=model,
        residual=spectra.flux - model,
        chi2_history=result.chi2_history,
        objective_history=result.objective_history,
        history_stage=result.history_stage,
    )
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    )
    if not args.no_plots:
        plot_average_spectrum(
            prepared.wavelength,
            prepared.average_flux,
            prepared.continuum,
            spectra.velocity,
            prepared.average_line_flux,
            args.output / "Average_Spec.pdf",
            continuum_band=prepared.continuum_band,
        )
        plot_trails(spectra, args.output / "Trail.pdf", cycles=2)
        plot_map_slices(
            doppler_map,
            args.output / "Doppler_Map.pdf",
            vz_values=[-900, -500, 0, 500, 900],
        )
        plot_map_projection(doppler_map, args.output / "Doppler_Map_Projection.pdf")
        save_volume_scatter_preview(
            doppler_map, args.output / "Doppler_Map_3D_Preview.pdf"
        )
        plot_map_volume_html(doppler_map, args.output / "Doppler_Map_3D.html")
        plot_reconstruction(
            spectra,
            model,
            args.output / "Reconstruction.pdf",
            cycles=2,
            standardized=True,
        )
        plot_residuals(
            spectra, model, args.output / "Residuals.pdf", cycles=2, standardized=True
        )
    print(f"{result.status}: mean chi2={result.chi2_history[-1]:.6g}; {result.message}")
    print(f"Products and provenance: {args.output.resolve()}")
    if not result.success and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
