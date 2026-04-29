from pathlib import Path

import numpy as np

from pydoppler3d import VelocityGrid3D, project_cube


def main() -> None:
    grid = VelocityGrid3D.regular(vlim_xy=1_500.0, nxy=61, vlim_z=1_000.0, nz=41)
    velocities = np.linspace(-2_500.0, 2_500.0, 401)
    phases = np.linspace(0.0, 1.0, 80, endpoint=False)

    vx, vy, vz = grid.mesh()
    cube = np.exp(
        -0.5
        * (
            ((vx + 500.0) / 120.0) ** 2
            + ((vy - 700.0) / 120.0) ** 2
            + ((vz - 300.0) / 180.0) ** 2
        )
    )
    profiles = project_cube(
        cube,
        grid,
        phases,
        velocities,
        inclination_deg=75.0,
        instrumental_fwhm=25.0,
    )

    outdir = Path("output_images")
    outdir.mkdir(exist_ok=True)
    np.savez_compressed(
        outdir / "synthetic_3d_doppler_profiles.npz",
        phases=phases,
        velocities=velocities,
        profiles=profiles,
    )
    print(f"Wrote {profiles.shape} synthetic profiles to {outdir}")


if __name__ == "__main__":
    main()
