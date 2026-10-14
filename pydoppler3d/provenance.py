"""Runtime and source fingerprints for reproducible numerical products."""

import hashlib
import platform
from pathlib import Path

import numpy as np
import scipy

from ._version import __version__


def runtime_provenance():
    root = Path(__file__).resolve().parent
    return {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))
        },
    }
