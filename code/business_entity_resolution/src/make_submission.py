"""Build ``<team_name>_submission.zip`` exactly per the challenge README tree.

Usage (from code/business_entity_resolution/):
    python -m src.make_submission --team-name TEAM [--data-dir PATH] [--out-dir PATH]
        [--zip-dir PATH] [--doc PATH]

Zip layout::

    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/**.py      (no __pycache__, no data files)
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md

Steps: run utils/validate_submission.py on the two output TSVs (refuse unless it
prints PASS), write the zip, unzip it into a temporary folder and check the tree
holds exactly the expected files, then move it into place. Only ``*.py`` files
are taken from src/, so artifacts, parquet/TSV data, notebooks and CLAUDE.md can
never end up in the zip.

Memory: the validator streams the TSVs (no --check-ids); zipping streams files.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

from src import config, decide, io_utils
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PACKAGE = "code/business_entity_resolution"
OUTPUT_FILES: tuple[str, ...] = ("matching_results.tsv", "candidate_pairs.tsv")
DOC = "Documentation_template.md"
PACKAGE_TOP: frozenset[str] = frozenset({"src", "README.md", "requirements.txt"})
TEAM_PATTERN = r"^[A-Za-z0-9_-]+$"
HEADERS: dict[str, tuple[str, ...]] = {"matching_results.tsv": io_utils.MATCHING_HEADER,
                                      "candidate_pairs.tsv": io_utils.CANDIDATE_HEADER}


def package_files(repo_root: Path, output_dir: Path, doc: Path) -> dict[str, Path]:
    """Map every zip member name to its source file.

    Args:
        repo_root: Repository root (holds ``code/business_entity_resolution``).
        output_dir: Folder with the two submission TSVs.
        doc: Filled-in methodology document.

    Returns:
        Member name (POSIX, relative to the zip root) -> source path.

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    pkg = repo_root / PACKAGE
    files = {f"output/{name}": output_dir / name for name in OUTPUT_FILES}
    files |= {f"{PACKAGE}/{name}": pkg / name for name in ("README.md", "requirements.txt")}
    for f in sorted((pkg / "src").rglob("*.py")):
        rel = f.relative_to(pkg)
        if "__pycache__" not in rel.parts:
            files[f"{PACKAGE}/{rel.as_posix()}"] = f
    files[DOC] = doc
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing file(s) for the submission: {missing}")
    return files


def check_tree(root: Path, expected: set[str]) -> None:
    """Check an unzipped package holds exactly the README tree and nothing else.

    Args:
        root: Folder the zip was extracted into.
        expected: Member names that must be present (``package_files`` keys).

    Raises:
        ValueError: On missing / extra files, a wrong top level, a forbidden
            file, or an output TSV with the wrong header.
    """
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Zip tree mismatch: missing {sorted(expected - actual)}, extra {sorted(actual - expected)}")
    top = {p.name for p in root.iterdir()}
    if top != {"output", "code", DOC}:
        raise ValueError(f"Zip top level must be output/, code/, {DOC}; got {sorted(top)}")
    if {p.name for p in (root / "output").iterdir()} != set(OUTPUT_FILES):
        raise ValueError(f"output/ must hold exactly {OUTPUT_FILES}")
    if {p.name for p in (root / "code").iterdir()} != {"business_entity_resolution"}:
        raise ValueError("code/ must hold only business_entity_resolution/")
    if {p.name for p in (root / PACKAGE).iterdir()} != PACKAGE_TOP:
        raise ValueError(f"{PACKAGE}/ must hold exactly {sorted(PACKAGE_TOP)}")
    bad = [n for n in actual if n.startswith(f"{PACKAGE}/src/") and not n.endswith(".py")]
    if bad:
        raise ValueError(f"Non-source files under src/: {bad}")
    for name, header in HEADERS.items():
        with (root / "output" / name).open(encoding="utf-8", newline="") as fh:
            first = fh.readline()
        if first != "\t".join(header) + "\n":
            raise ValueError(f"output/{name}: header {first!r} != {header}")


def build(team_name: str, data_dir: Path, output_dir: Path, zip_dir: Path, doc: Path) -> Path:
    """Validate the outputs, write the zip, re-check it unzipped, move it into place.

    Args:
        team_name: Team name for ``<team_name>_submission.zip``.
        data_dir: Folder with ``test/`` (the validator reads the test S1 IDs).
        output_dir: Folder with the two submission TSVs.
        zip_dir: Folder the zip is written to.
        doc: Methodology document placed at the zip root as ``Documentation_template.md``.

    Returns:
        Path of the finished zip.

    Raises:
        ValueError: On a bad team name or a wrong zip tree.
        FileNotFoundError: If the validator or a packaged file is missing.
        RuntimeError: If the validator does not PASS.
    """
    if not re.match(TEAM_PATTERN, team_name):
        raise ValueError(f"Team name must match {TEAM_PATTERN}: {team_name!r}")
    decide.run_validator(output_dir / OUTPUT_FILES[0], output_dir / OUTPUT_FILES[1], data_dir / "test", required=True)
    files = package_files(config.REPO_ROOT, output_dir, doc)
    zip_dir.mkdir(parents=True, exist_ok=True)
    out = zip_dir / f"{team_name}_submission.zip"
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, src in files.items():
            zf.write(src, name)
    try:
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(tmp) as zf:
            zf.extractall(td)
            check_tree(Path(td), set(files))
    except ValueError:
        tmp.unlink()
        raise
    tmp.replace(out)
    logger.info("Wrote %s: %d files, %.1f MB (validator PASS, tree checked)", out, len(files), out.stat().st_size / 1e6)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if None.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--team-name", required=True)
    p.add_argument("--data-dir", type=Path, default=None, help="folder with test/ (default: <repo>/dataset)")
    p.add_argument("--out-dir", type=Path, default=None, help="folder with the two TSVs (default: <repo>/output)")
    p.add_argument("--zip-dir", type=Path, default=None, help="where to write the zip (default: repo root)")
    p.add_argument("--doc", type=Path, default=None, help=f"methodology document (default: <repo>/{DOC})")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    paths = config.get_paths(args.data_dir, None, args.out_dir)
    setup_logging(args.log_level, paths.log_dir)
    command = "src.make_submission " + " ".join(sys.argv[1:] if argv is None else argv)
    with fulldata_lock(paths.data_dir, command):
        build(args.team_name, paths.data_dir, paths.output_dir,
              (args.zip_dir or config.REPO_ROOT).resolve(), (args.doc or config.REPO_ROOT / DOC).resolve())


if __name__ == "__main__":
    main()
