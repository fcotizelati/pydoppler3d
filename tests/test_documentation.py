"""Run the actual README quick start in an empty directory."""

import re
from pathlib import Path

import pytest


def test_readme_quickstart(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    readme = Path(__file__).resolve().parents[1] / "README.md"
    examples = re.findall(r"```python\n(.*?)```", readme.read_text(), flags=re.S)
    assert len(examples) == 1
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(examples[0], str(readme), "exec"), namespace)
    assert (tmp_path / "output_images" / "Doppler_Map.npz").is_file()
    assert (tmp_path / "output_images" / "Reconstruction.pdf").is_file()
    assert namespace["result"].status != "not_assessed"
    assert (
        namespace["doppler_map"].meta["velocity_convention"] == "right_handed_minus_vz"
    )
