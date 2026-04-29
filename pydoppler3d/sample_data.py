"""Bundled sample-data helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

try:
    from importlib.resources import files as resource_files
except ImportError:  # pragma: no cover - Python <3.9 fallback
    from importlib_resources import files as resource_files  # type: ignore


def get_test_data_path():
    """Return the package resource path containing bundled sample data."""

    return resource_files(__package__).joinpath("test_data")


def copy_test_data(destination: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Copy bundled sample data into ``destination``.

    This mirrors the convenience function in :mod:`pydoppler` and copies both
    the U Gem comparison set and the V834 Cen magnetic-CV example.
    """

    source = get_test_data_path()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    def walk(node, relative: Path = Path(".")) -> None:
        for item in node.iterdir():
            rel = relative / item.name
            target = destination / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                walk(item, rel)
                continue
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with item.open("rb") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied.append(target)

    walk(source)
    return copied
