"""Prototype 3D Doppler tomography tools."""

from .components import (
    MapComponent,
    back_project_components,
    phase_weight,
    project_components,
)
from .convolution import convolve_profiles_fft, gaussian_kernel
from .data import DopplerMap, TrailedSpectra
from .defaults import gaussian_default, squeezed_default
from .geometry import VelocityGrid3D, line_velocity_axis, radial_velocity
from .project import back_project, project_cube
from .pydoppler_compat import load_v834cen_dataset
from .reconstruct import (
    LandweberConfig,
    MemConfig,
    ReconstructionResult,
    entropy,
    landweber_reconstruct,
    mem_reconstruct,
)
from .sample_data import copy_test_data, get_test_data_path
from .visualize import (
    plot_average_spectrum,
    plot_map_projection,
    plot_map_slices,
    plot_map_volume_html,
    plot_reconstruction,
    plot_residuals,
    plot_trails,
    save_volume_scatter_preview,
)

__all__ = [
    "DopplerMap",
    "LandweberConfig",
    "MapComponent",
    "MemConfig",
    "ReconstructionResult",
    "TrailedSpectra",
    "VelocityGrid3D",
    "back_project",
    "back_project_components",
    "convolve_profiles_fft",
    "copy_test_data",
    "entropy",
    "gaussian_default",
    "gaussian_kernel",
    "get_test_data_path",
    "landweber_reconstruct",
    "line_velocity_axis",
    "load_v834cen_dataset",
    "mem_reconstruct",
    "phase_weight",
    "plot_average_spectrum",
    "plot_map_projection",
    "plot_map_slices",
    "plot_map_volume_html",
    "plot_reconstruction",
    "plot_residuals",
    "plot_trails",
    "project_components",
    "project_cube",
    "radial_velocity",
    "save_volume_scatter_preview",
    "squeezed_default",
]

__version__ = "0.1.0"
