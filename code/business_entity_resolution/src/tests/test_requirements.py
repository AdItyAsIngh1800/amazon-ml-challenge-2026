"""requirements.txt pins exactly the third-party packages src/ imports, at the installed versions."""

from __future__ import annotations

import ast
import sys
from importlib import metadata
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
# import name -> distribution name, where they differ
DIST = {"sklearn": "scikit-learn"}
# pinned for QA only (type stubs / tools), never imported by src/
DEV_ONLY = {"mypy", "pandas-stubs", "scipy-stubs", "types-psutil"}


def _imports() -> set[str]:
    """Distribution names of every non-stdlib top-level import under src/ (tests included)."""
    found: set[str] = set()
    for f in (PKG_ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found |= {n.split(".")[0] for n in names}
    return {DIST.get(m, m) for m in found - set(sys.stdlib_module_names) - {"src"}}


def _pins() -> dict[str, str]:
    """``name==version`` lines of requirements.txt."""
    lines = (PKG_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pairs = [ln.split("#")[0].strip().split("==") for ln in lines]
    return {p[0].lower(): p[1] for p in pairs if p != [""]}


def test_every_line_is_pinned() -> None:
    """No unpinned or range requirements."""
    lines = (PKG_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    bad = [ln for ln in lines if ln.split("#")[0].strip() and "==" not in ln]
    assert not bad


def test_pins_match_imports() -> None:
    """Every imported package is pinned; every pinned runtime package is imported."""
    pins = set(_pins())
    assert _imports() - pins == set()
    assert pins - DEV_ONLY - _imports() == set()


def test_pins_match_installed() -> None:
    """Pinned versions are the ones this environment runs (and was tested with)."""
    wrong = {n: (v, metadata.version(n)) for n, v in _pins().items() if metadata.version(n) != v}
    assert not wrong
