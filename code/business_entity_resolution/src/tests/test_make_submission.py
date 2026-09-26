"""Tests for make_submission.py on a fake repository with the real validator."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src import config, io_utils, make_submission

REAL_VALIDATOR = config.REPO_ROOT / "utils" / "validate_submission.py"
PKG = make_submission.PACKAGE


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake repo: package with source, tests, clutter; tiny test split; valid outputs."""
    root = tmp_path / "repo"
    pkg = root / PKG
    for rel, text in {"src/__init__.py": "", "src/run_pipeline.py": "x = 1\n", "src/tests/test_x.py": "",
                      "src/__pycache__/run_pipeline.cpython-313.pyc": "junk", "src/tests/fixture.parquet": "data",
                      "src/notes.ipynb": "{}", "README.md": "# r\n", "requirements.txt": "numpy==2.5.3\n",
                      "pyproject.toml": "", "CLAUDE.md": "private"}.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text, encoding="utf-8")
    for rel in ("CLAUDE.md", "artifacts/x.parquet", "Documentation_template.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n", encoding="utf-8")
    (root / "utils").mkdir()
    shutil.copy(REAL_VALIDATOR, root / "utils" / "validate_submission.py")
    cols = list(io_utils.SOURCE_COLUMNS)
    rows = {1: [("S1-1", "A", "x", "France"), ("S1-2", "B", "y", "US")], 2: [("S2-1", "A", "x", "France")],
            3: [("S3-1", "B", "y", "US")]}
    for s, r in rows.items():
        io_utils.write_source_tsv(pd.DataFrame(r, columns=cols), root / "dataset" / "test" / f"test_source{s}.tsv")
    out = root / "output"
    io_utils.write_id_list_tsv({"S1-1": ["S2-1"], "S1-2": []}, ["S1-1", "S1-2"], out / "matching_results.tsv",
                               io_utils.MATCHING_HEADER)
    io_utils.write_id_list_tsv({"S1-1": ["S2-1"], "S1-2": ["S3-1"]}, ["S1-1", "S1-2"], out / "candidate_pairs.tsv",
                               io_utils.CANDIDATE_HEADER)
    monkeypatch.setattr(config, "REPO_ROOT", root)
    return root


def _build(repo: Path) -> Path:
    """Build with the fake repo's defaults."""
    return make_submission.build("team_x", repo / "dataset", repo / "output", repo, repo / "Documentation_template.md")


def test_zip_matches_readme_tree(repo: Path) -> None:
    """Exactly outputs + src/*.py + README + requirements + doc; no pyc, data, notebooks, CLAUDE.md, pyproject."""
    out = _build(repo)
    assert out == repo / "team_x_submission.zip" and not (repo / "team_x_submission.zip.tmp").exists()
    names = set(zipfile.ZipFile(out).namelist())
    assert names == {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                     f"{PKG}/README.md", f"{PKG}/requirements.txt", f"{PKG}/src/__init__.py",
                     f"{PKG}/src/run_pipeline.py", f"{PKG}/src/tests/test_x.py"}


def test_refuses_when_validator_fails(repo: Path) -> None:
    """A missing S1 row fails validation: no zip is written."""
    io_utils.write_id_list_tsv({"S1-1": []}, ["S1-1"], repo / "output" / "matching_results.tsv",
                               io_utils.MATCHING_HEADER)
    with pytest.raises(RuntimeError, match="did NOT pass"):
        _build(repo)
    assert not list(repo.glob("*.zip*"))


def test_refuses_without_validator(repo: Path) -> None:
    """Packaging never skips validation."""
    (repo / "utils" / "validate_submission.py").unlink()
    with pytest.raises(FileNotFoundError, match="Validator"):
        _build(repo)


def test_bad_team_name_and_missing_doc(repo: Path) -> None:
    """Team names are path-safe; a missing methodology document is reported."""
    with pytest.raises(ValueError, match="Team name"):
        make_submission.build("../x", repo / "dataset", repo / "output", repo, repo / "Documentation_template.md")
    with pytest.raises(FileNotFoundError, match="nope.md"):
        make_submission.build("t", repo / "dataset", repo / "output", repo, repo / "nope.md")


def test_check_tree_rejects_extras(repo: Path, tmp_path: Path) -> None:
    """An unexpected file or a wrong output header fails the tree check."""
    files = make_submission.package_files(repo, repo / "output", repo / "Documentation_template.md")
    root = tmp_path / "unzipped"
    for name, src in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, root / name)
    make_submission.check_tree(root, set(files))
    (root / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        make_submission.check_tree(root, set(files))
    (root / "extra.txt").unlink()
    (root / "output" / "candidate_pairs.tsv").write_text("a\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        make_submission.check_tree(root, set(files))


def test_cli(repo: Path) -> None:
    """The CLI resolves the default dataset / output / doc / zip locations."""
    make_submission.main(["--team-name", "cli_team", "--data-dir", str(repo / "dataset"),
                          "--out-dir", str(repo / "output")])
    assert (repo / "cli_team_submission.zip").is_file()
