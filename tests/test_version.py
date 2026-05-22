"""Sanity check: the version string is defined in three places (pyproject, package
init, FastAPI app) and they must agree. This test fails loudly if any drifts."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text()


def _pyproject_version() -> str:
    text = _read(Path("pyproject.toml"))
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no version line"
    return m.group(1)


def _package_version() -> str:
    import chorus
    return chorus.__version__


def _fastapi_version() -> str:
    text = _read(Path("chorus/server/app.py"))
    m = re.search(r'version\s*=\s*"([^"]+)"', text)
    assert m, "chorus/server/app.py has no FastAPI version kwarg"
    return m.group(1)


def test_version_triplet_agrees():
    pp = _pyproject_version()
    pkg = _package_version()
    fa = _fastapi_version()
    assert pp == pkg == fa, (
        f"Version drift detected: pyproject={pp!r}, package={pkg!r}, fastapi={fa!r}. "
        "Update all three together for any version bump."
    )
