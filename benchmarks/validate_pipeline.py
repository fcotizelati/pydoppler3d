"""Independent simulations and cache benchmark; writes machine-readable evidence.

Run from an installed/editable checkout:
  python -m benchmarks.validate_pipeline --output docs/validation_results.json
"""

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtr

from pydoppler3d import (
    MemConfig,
    ProjectionOperator,
    VelocityGrid3D,
    __version__,
    mem_reconstruct,
)


def analytic_profiles(
    phases,
    velocity,
    *,
    inclination=55.0,
    gamma=0.0,
    width=0.04,
    sigma=25.0,
    modulation=0.0,
    planar=False,
):
    """Gaussian pixel integrals and dense exposure sampling; no package projector."""
    dv = velocity[1] - velocity[0]
    edges = np.r_[velocity - dv / 2, velocity[-1] + dv / 2]
    subphase = phases[:, None] + width * ((np.arange(301) + 0.5) / 301 - 0.5)[None, :]
    inc = np.deg2rad(inclination)
    profiles = np.zeros((len(phases), len(velocity)))
    # A compact stream-like component plus a broad planar disc (24 ring sites).
    sites = [(-120.0, 80.0, 0.0 if planar else 90.0, 2.5)]
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        sites.append((180 * np.cos(angle), 180 * np.sin(angle), 0.0, 0.1))
    for vx, vy, vz, flux in sites:
        center = (
            gamma
            + np.sin(inc)
            * (-vx * np.cos(2 * np.pi * subphase) + vy * np.sin(2 * np.pi * subphase))
            - vz * np.cos(inc)
        )
        bins = np.diff(
            ndtr((edges[None, None, :] - center[:, :, None]) / sigma), axis=2
        )
        profiles += flux * np.mean(
            bins * (1 + modulation * np.cos(2 * np.pi * subphase[:, :, None])), axis=1
        )
    return profiles


def run(output, iterations=800):
    rng = np.random.default_rng(901)
    phases = np.linspace(0, 1, 20, endpoint=False)
    velocity = np.linspace(-350, 350, 101)
    grid = VelocityGrid3D.regular(vlim_xy=300, nxy=17, vlim_z=180, nz=13)
    cases = [
        ("off_plane", "squeezed", 55.0, 25.0, 0.0, False, False),
        ("gaussian_default", "gaussian", 55.0, 25.0, 0.0, False, False),
        ("planar_source", "squeezed", 55.0, 25.0, 0.0, True, False),
        ("phase_gaps", "squeezed", 55.0, 25.0, 0.0, False, True),
        ("modulated_misspecification", "squeezed", 55.0, 25.0, 0.35, False, False),
        ("wrong_resolution", "squeezed", 55.0, 10.0, 0.0, False, False),
        ("2d_projected_baseline", "gaussian", 90.0, 25.0, 0.0, False, False),
    ]
    records = []
    # Reuse the same noise realization where the observations are identical.
    noise = rng.normal(0, 0.015, (len(phases), len(velocity)))
    for name, default, inc, fit_sigma, modulation, planar, gaps in cases:
        select = (phases < 0.3) | (phases > 0.6) if gaps else np.ones(len(phases), bool)
        p = phases[select]
        truth = analytic_profiles(
            phases, velocity, modulation=modulation, planar=planar
        )[select]
        data = truth + noise[select]
        used_grid = (
            VelocityGrid3D.regular(vlim_xy=300, nxy=17, vlim_z=180, nz=2)
            if inc == 90
            else grid
        )
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mem_reconstruct(
                data,
                used_grid,
                p,
                velocity,
                error=np.full_like(data, 0.015),
                inclination_deg=inc,
                exposure=0.04,
                instrumental_fwhm=fit_sigma * 2 * np.sqrt(2 * np.log(2)),
                phase_samples=15,
                config=MemConfig(
                    iterations=iterations,
                    default_updates=1,
                    default=default,
                    alpha=0.002,
                    default_fwhm_kms=80,
                    squeeze_sigma_vz_kms=60,
                ),
            )
        model = ProjectionOperator(
            used_grid,
            p,
            velocity,
            inclination_deg=inc,
            exposure=0.04,
            phase_samples=15,
            instrumental_fwhm=fit_sigma * 2 * np.sqrt(2 * np.log(2)),
        ).forward(result.image)
        total = result.image.sum()
        centroid = float(np.sum(result.image * used_grid.vz[None, None, :]) / total)
        records.append(
            {
                "case": name,
                "seconds": time.perf_counter() - started,
                "mean_chi2": float(result.chi2_history[-1]),
                "status": result.status,
                "success": result.success,
                "iterations": result.n_iterations,
                "model_vs_truth_rms_in_sigma": float(
                    np.sqrt(np.mean(((model - truth) / 0.015) ** 2))
                ),
                "vz_centroid_kms": None if inc == 90 else centroid,
                "vz_rms_kms": None
                if inc == 90
                else float(
                    np.sqrt(
                        np.sum(
                            result.image * (used_grid.vz[None, None, :] - centroid) ** 2
                        )
                        / total
                    )
                ),
            }
        )
    timings = {}
    for budget in (0, 64 * 1024**2):
        started = time.perf_counter()
        operator = ProjectionOperator(
            grid, phases, velocity, inclination_deg=55, cache_max_bytes=budget
        )
        setup = time.perf_counter() - started
        cube = np.ones(grid.shape)
        residual = np.ones(operator.profile_shape)
        started = time.perf_counter()
        for _ in range(20):
            operator.forward(cube)
            operator.adjoint(residual)
        timings[str(budget)] = {
            "setup_seconds": setup,
            "20_forward_adjoint_seconds": time.perf_counter() - started,
            "cache_bytes": operator.cache_bytes,
        }
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "pydoppler3d").glob("*.py"))
    }
    summary = {
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "seed": 901,
        "iteration_budget": iterations,
        "phase_quadrature_nodes": 15,
        "native_simulation_exposure_samples": 301,
        "noise_sigma": 0.015,
        "source_sha256": hashes,
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "cases": records,
        "cache_benchmark": timings,
        "interpretation": "Simulation diagnostics, not calibrated detection significances or unique cube recovery.",
        "external_reference_benchmark": "Not performed: MEMSYS/trm-doppler is a separate optional native toolchain.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": records, "cache_benchmark": timings}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/validation_results.json")
    )
    parser.add_argument("--iterations", type=int, default=800)
    args = parser.parse_args()
    run(args.output, args.iterations)
