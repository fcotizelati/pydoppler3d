"""Visualization helpers for Doppler trails and 3D map cubes."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .data import DopplerMap, TrailedSpectra


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Visualization requires matplotlib: pip install pydoppler3d[plot]"
        ) from exc
    return plt


def plot_trails(
    spectra: TrailedSpectra,
    output: str | Path,
    *,
    cmap: str = "magma",
    dpi: int = 180,
    cycles: int = 1,
    title: str = "Trailed spectra",
) -> Path:
    """Save a phase-velocity heatmap of trailed spectra."""

    plt = _pyplot()
    output = Path(output)
    cycles = max(1, int(cycles))
    order = np.argsort(spectra.phases)
    phases = spectra.phases[order]
    flux = spectra.flux[order]
    if cycles > 1:
        phases = np.concatenate([phases + cycle for cycle in range(cycles)])
        flux = np.vstack([flux for _cycle in range(cycles)])
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        spectra.velocity,
        phases,
        flux,
        shading="auto",
        cmap=cmap,
    )
    fig.colorbar(mesh, ax=ax, label="Flux")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Orbital phase")
    ax.set_title(title)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_average_spectrum(
    wavelength: np.ndarray,
    average_flux: np.ndarray,
    continuum: np.ndarray,
    velocity: np.ndarray,
    average_line_flux: np.ndarray,
    output: str | Path,
    *,
    continuum_band: tuple[float, float, float, float] | None = None,
    dpi: int = 180,
) -> Path:
    """Save a PyDoppler-style average spectrum and line-window diagnostic."""

    plt = _pyplot()
    wavelength = np.asarray(wavelength, dtype=float)
    average_flux = np.asarray(average_flux, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    output = Path(output)
    fig, axes = plt.subplots(2, 1, figsize=(6.57, 8.57), constrained_layout=True)
    axes[0].plot(wavelength, average_flux, color="black", lw=1.0)
    if continuum_band is not None:
        band = tuple(float(value) for value in continuum_band)
        local = (wavelength >= band[0]) & (wavelength <= band[3])
        axes[0].axvspan(
            band[0],
            band[1],
            color="0.85",
            alpha=0.55,
            label="Continuum windows",
        )
        axes[0].axvspan(band[2], band[3], color="0.85", alpha=0.55)
        axes[0].axvspan(
            band[1],
            band[2],
            color="tab:orange",
            alpha=0.10,
            label="Line window",
        )
        axes[0].plot(
            wavelength[local],
            continuum[local],
            color="crimson",
            lw=1.4,
            label="Local continuum fit",
        )
        for value in band:
            axes[0].axvline(value, color="0.3", ls="--", lw=0.8)
        margin = 0.12 * (band[3] - band[0])
        axes[0].set_xlim(band[0] - margin, band[3] + margin)
    else:
        axes[0].plot(wavelength, continuum, color="crimson", lw=1.2, label="Continuum fit")
    axes[0].set_xlabel(r"Wavelength / $\AA$")
    axes[0].set_ylabel("Input flux")
    axes[0].legend(loc="best")

    axes[1].plot(velocity, average_line_flux, color="black", lw=1.0)
    axes[1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[1].set_xlabel("Velocity (km/s)")
    axes[1].set_ylabel("Continuum-subtracted flux")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = [int(np.argmin(np.abs(axis - value))) for value in values]
    return np.asarray(sorted(set(indices)), dtype=int)


def _auto_slice_indices(axis: np.ndarray, nslices: int) -> np.ndarray:
    nslices = max(1, min(int(nslices), axis.size))
    return np.unique(np.linspace(0, axis.size - 1, nslices, dtype=int))


def plot_map_slices(
    doppler_map: DopplerMap,
    output: str | Path,
    *,
    vz_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
    nslices: int = 5,
    cmap: str = "viridis",
    dpi: int = 180,
    percentile: float = 99.5,
) -> Path:
    """Save a montage of ``Vx-Vy`` slices through a 3D Doppler cube."""

    plt = _pyplot()
    output = Path(output)
    image = doppler_map.image
    grid = doppler_map.grid
    if vz_values is None:
        indices = _auto_slice_indices(grid.vz, nslices)
    else:
        indices = _nearest_indices(grid.vz, np.asarray(vz_values, dtype=float))

    planes = [image[:, :, index].T for index in indices]
    finite = np.concatenate([plane[np.isfinite(plane)] for plane in planes])
    if finite.size == 0:
        raise ValueError("map contains no finite values to plot.")
    vmin = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 1))
    vmax = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    ncols = min(3, len(indices))
    nrows = int(ceil(len(indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    shown = None
    extent = (grid.vx[0], grid.vx[-1], grid.vy[0], grid.vy[-1])
    for ax, index, plane in zip(axes.flat, indices, planes, strict=False):
        shown = ax.imshow(
            plane,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Vz = {grid.vz[index]:.0f} km/s")
        ax.set_xlabel("Vx (km/s)")
        ax.set_ylabel("Vy (km/s)")
    for ax in list(axes.flat)[len(indices) :]:
        ax.set_visible(False)
    if shown is not None:
        fig.colorbar(shown, ax=axes.ravel().tolist(), label="Map intensity")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_map_projection(
    doppler_map: DopplerMap,
    output: str | Path,
    *,
    method: str = "sum",
    cmap: str = "viridis",
    dpi: int = 180,
) -> Path:
    """Save a ``Vz``-collapsed ``Vx-Vy`` view of a 3D Doppler cube."""

    plt = _pyplot()
    output = Path(output)
    image = doppler_map.image
    grid = doppler_map.grid
    if method == "sum":
        plane = np.sum(image, axis=2).T
        title = "Vz-integrated map"
    elif method == "max":
        plane = np.max(image, axis=2).T
        title = "Vz maximum-intensity projection"
    else:
        raise ValueError("method must be 'sum' or 'max'.")

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    shown = ax.imshow(
        plane,
        origin="lower",
        extent=(grid.vx[0], grid.vx[-1], grid.vy[0], grid.vy[-1]),
        aspect="equal",
        cmap=cmap,
    )
    fig.colorbar(shown, ax=ax, label="Map intensity")
    ax.set_xlabel("Vx (km/s)")
    ax.set_ylabel("Vy (km/s)")
    ax.set_title(title)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_reconstruction(
    spectra: TrailedSpectra,
    model_flux: np.ndarray,
    output: str | Path,
    *,
    cmap: str = "magma",
    dpi: int = 180,
    cycles: int = 1,
) -> Path:
    """Save input, reconstructed, and residual trailed spectra."""

    plt = _pyplot()
    output = Path(output)
    model = np.asarray(model_flux, dtype=float)
    if model.shape != spectra.flux.shape:
        raise ValueError("model_flux must have the same shape as spectra.flux.")
    residual = spectra.flux - model

    order = np.argsort(spectra.phases)
    phases = spectra.phases[order]
    observed = spectra.flux[order]
    model = model[order]
    residual = residual[order]
    if cycles > 1:
        phases = np.concatenate([phases + cycle for cycle in range(cycles)])
        observed = np.vstack([observed for _cycle in range(cycles)])
        model = np.vstack([model for _cycle in range(cycles)])
        residual = np.vstack([residual for _cycle in range(cycles)])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), constrained_layout=True)
    panels = [
        ("Observed trail", observed),
        ("Reconstructed trail", model),
        ("Residuals", residual),
    ]
    for ax, (title, data) in zip(axes, panels, strict=True):
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2.0, 98.0])
            if hi <= lo:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            lo, hi = 0.0, 1.0
        mesh = ax.pcolormesh(
            spectra.velocity,
            phases,
            data,
            shading="auto",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        fig.colorbar(mesh, ax=ax, label="Flux")
        ax.set_title(title)
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Orbital phase")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_residuals(
    spectra: TrailedSpectra,
    model_flux: np.ndarray,
    output: str | Path,
    *,
    cmap: str = "magma",
    dpi: int = 180,
    cycles: int = 1,
) -> Path:
    """Save a residual trailed-spectrum heatmap."""

    plt = _pyplot()
    output = Path(output)
    model = np.asarray(model_flux, dtype=float)
    if model.shape != spectra.flux.shape:
        raise ValueError("model_flux must have the same shape as spectra.flux.")
    order = np.argsort(spectra.phases)
    residual = (spectra.flux - model)[order]
    phases = spectra.phases[order]
    if cycles > 1:
        phases = np.concatenate([phases + cycle for cycle in range(cycles)])
        residual = np.vstack([residual for _cycle in range(cycles)])
    vmax = float(np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 98.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        spectra.velocity,
        phases,
        residual,
        shading="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label="Observed - reconstructed")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Orbital phase")
    ax.set_title("Residual trailed spectra")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_map_volume_html(
    doppler_map: DopplerMap,
    output: str | Path,
    *,
    percentile: float = 97.0,
    surface_count: int = 5,
    opacity: float = 0.18,
    max_voxels: int = 120_000,
) -> Path:
    """Save an interactive Plotly HTML isosurface/volume view of a 3D cube."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Interactive 3D visualization requires plotly: "
            "pip install pydoppler3d[volume]"
        ) from exc

    output = Path(output)
    image = np.asarray(doppler_map.image, dtype=float)
    grid = doppler_map.grid
    step = max(1, int(np.ceil((image.size / max_voxels) ** (1.0 / 3.0))))
    image = image[::step, ::step, ::step]
    vx = grid.vx[::step]
    vy = grid.vy[::step]
    vz = grid.vz[::step]

    values = image.ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("map contains no finite values to plot.")
    isomin = float(np.nanpercentile(finite, percentile))
    isomax = float(np.nanmax(finite))
    if not np.isfinite(isomin) or isomin >= isomax:
        isomin = float(np.nanpercentile(finite, 90.0))
    if isomin >= isomax:
        isomin = float(np.nanmin(finite))

    x, y, z = np.meshgrid(vx, vy, vz, indexing="ij")
    fig = go.Figure(
        data=[
            go.Isosurface(
                x=x.ravel(),
                y=y.ravel(),
                z=z.ravel(),
                value=values,
                isomin=isomin,
                isomax=isomax,
                opacity=opacity,
                surface_count=int(surface_count),
                colorscale="Magma",
                caps={"x_show": False, "y_show": False, "z_show": False},
            )
        ]
    )
    fig.update_layout(
        title="3D Doppler map",
        scene={
            "xaxis_title": "Vx (km/s)",
            "yaxis_title": "Vy (km/s)",
            "zaxis_title": "Vz (km/s)",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(output, include_plotlyjs="cdn")
    return output


def save_volume_scatter_preview(
    doppler_map: DopplerMap,
    output: str | Path,
    *,
    percentile: float = 99.9,
    point_percentile: float = 99.85,
    max_points: int = 90,
    cmap: str = "magma",
    dpi: int = 180,
) -> Path:
    """Save a static 3D overview with transparent projections and peak voxels."""

    plt = _pyplot()
    output = Path(output)
    image = np.asarray(doppler_map.image, dtype=float)
    grid = doppler_map.grid
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("map contains no finite values to plot.")

    vmin = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap_obj = plt.get_cmap(cmap)
    normalizer = plt.Normalize(vmin=vmin, vmax=vmax)

    def colors(values: np.ndarray, *, max_alpha: float) -> np.ndarray:
        scaled = np.clip(normalizer(values), 0.0, 1.0)
        out = cmap_obj(scaled)
        out[..., -1] = max_alpha * (0.08 + 0.92 * scaled)
        return out

    vx, vy, vz = grid.mesh()
    xy = np.nanmax(image, axis=2)
    xz = np.nanmax(image, axis=1)
    yz = np.nanmax(image, axis=0)

    fig = plt.figure(figsize=(8.3, 7.2), constrained_layout=True)
    ax: Any = fig.add_subplot(111, projection="3d")

    xx, yy = np.meshgrid(grid.vx, grid.vy, indexing="ij")
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, grid.vz[0]),
        facecolors=colors(xy, max_alpha=0.72),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    xx, zz = np.meshgrid(grid.vx, grid.vz, indexing="ij")
    ax.plot_surface(
        xx,
        np.full_like(xx, grid.vy[-1]),
        zz,
        facecolors=colors(xz, max_alpha=0.58),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    yy, zz = np.meshgrid(grid.vy, grid.vz, indexing="ij")
    ax.plot_surface(
        np.full_like(yy, grid.vx[0]),
        yy,
        zz,
        facecolors=colors(yz, max_alpha=0.58),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    threshold = float(np.nanpercentile(finite, point_percentile))
    peak_mask = np.isfinite(image) & (image >= threshold)
    peak_values = image[peak_mask]
    if peak_values.size:
        px = vx[peak_mask]
        py = vy[peak_mask]
        pz = vz[peak_mask]
        if peak_values.size > max_points:
            keep = np.argsort(peak_values)[-int(max_points) :]
            px = px[keep]
            py = py[keep]
            pz = pz[keep]
            peak_values = peak_values[keep]
        scaled = np.clip(normalizer(peak_values), 0.0, 1.0)
        ax.scatter(
            px,
            py,
            pz,
            c=peak_values,
            s=12.0 + 24.0 * scaled,
            cmap=cmap_obj,
            norm=normalizer,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.92,
            depthshade=False,
        )

    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap_obj)
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=ax,
        shrink=0.72,
        pad=0.08,
        label="Map intensity (percentile-clipped)",
    )
    ax.set_xlabel("Vx (km/s)")
    ax.set_ylabel("Vy (km/s)")
    ax.set_zlabel("Vz (km/s)")
    ax.set_xlim(grid.vx[0], grid.vx[-1])
    ax.set_ylim(grid.vy[0], grid.vy[-1])
    ax.set_zlim(grid.vz[0], grid.vz[-1])
    ax.set_box_aspect(
        (
            float(np.ptp(grid.vx)),
            float(np.ptp(grid.vy)),
            float(np.ptp(grid.vz)),
        )
    )
    ax.view_init(elev=24.0, azim=-55.0)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
