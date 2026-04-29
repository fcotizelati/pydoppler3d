"""Command-line entry points for PyDoppler3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .data import DopplerMap, TrailedSpectra
from .geometry import VelocityGrid3D
from .reconstruct import MemConfig, mem_reconstruct
from .visualize import (
    plot_map_projection,
    plot_map_slices,
    plot_map_volume_html,
    plot_trails,
)


def info_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show PyDoppler3D package information.")
    parser.parse_args(argv)
    print(f"pydoppler3d {__version__}")
    print("Backend: pure Python / NumPy / SciPy")


def reconstruct_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run pure-Python maximum-entropy reconstruction.")
    parser.add_argument("input", type=Path, help="Input TrailedSpectra NPZ.")
    parser.add_argument("output", type=Path, help="Output DopplerMap NPZ.")
    parser.add_argument("--nxy", type=int, default=41)
    parser.add_argument("--nz", type=int, default=31)
    parser.add_argument("--vlim-xy", type=float, default=1500.0)
    parser.add_argument("--vlim-z", type=float, default=1000.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--inclination", type=float, default=75.0)
    parser.add_argument(
        "--default-updates",
        type=int,
        default=2,
        help="Number of MEM default-map refreshes for L-BFGS-B.",
    )
    parser.add_argument("--target-chi2", type=float, default=None)
    args = parser.parse_args(argv)

    spectra = TrailedSpectra.from_npz(args.input)
    grid = VelocityGrid3D.regular(
        vlim_xy=args.vlim_xy,
        nxy=args.nxy,
        vlim_z=args.vlim_z,
        nz=args.nz,
    )
    result = mem_reconstruct(
        spectra.flux,
        grid,
        spectra.phases,
        spectra.velocity,
        error=spectra.error,
        config=MemConfig(
            iterations=args.iterations,
            alpha=args.alpha,
            default_updates=args.default_updates,
            target_chi2=args.target_chi2,
        ),
        inclination_deg=args.inclination,
    )
    DopplerMap(result.image, grid).to_npz(args.output)


def plot_trails_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot phase-velocity trailed spectra.")
    parser.add_argument("input", type=Path, help="Input TrailedSpectra NPZ.")
    parser.add_argument("output", type=Path, help="Output image file.")
    args = parser.parse_args(argv)

    plot_trails(TrailedSpectra.from_npz(args.input), args.output)


def plot_map_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Vx-Vy slices through a 3D map.")
    parser.add_argument("input", type=Path, help="Input DopplerMap NPZ.")
    parser.add_argument("output", type=Path, help="Output image file.")
    parser.add_argument("--vz", nargs="*", type=float, help="Vz values to display.")
    parser.add_argument("--nslices", type=int, default=5)
    parser.add_argument(
        "--projection",
        choices=["none", "sum", "max"],
        default="none",
        help="Plot a Vz-collapsed projection instead of slices.",
    )
    args = parser.parse_args(argv)

    doppler_map = DopplerMap.from_npz(args.input)
    if args.projection == "none":
        plot_map_slices(doppler_map, args.output, vz_values=args.vz, nslices=args.nslices)
    else:
        plot_map_projection(doppler_map, args.output, method=args.projection)


def plot_volume_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot an interactive 3D Doppler map.")
    parser.add_argument("input", type=Path, help="Input DopplerMap NPZ.")
    parser.add_argument("output", type=Path, help="Output HTML file.")
    parser.add_argument("--percentile", type=float, default=97.0)
    parser.add_argument("--surface-count", type=int, default=5)
    args = parser.parse_args(argv)

    plot_map_volume_html(
        DopplerMap.from_npz(args.input),
        args.output,
        percentile=args.percentile,
        surface_count=args.surface_count,
    )
