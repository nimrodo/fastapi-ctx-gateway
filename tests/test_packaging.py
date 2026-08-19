"""Verifies the built wheel actually ships the py.typed marker (PEP 561)."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_wheel_contains_py_typed_marker(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH to run this test"
    subprocess.run(
        [uv, "build", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "uv build produced no wheel"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    assert any(name.endswith("fastapi_ctx_gateway/py.typed") for name in names)
