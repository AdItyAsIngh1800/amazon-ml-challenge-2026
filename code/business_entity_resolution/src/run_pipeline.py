"""Single entry point for the entity-resolution pipeline.

Usage (from code/business_entity_resolution/):
    python -m src.run_pipeline --stage {prep,block,feat,train,predict,decide,write,all}
        --split {train,test} --data-dir PATH --out-dir PATH
        [--artifacts-dir PATH] [--force] [--log-level INFO] [config overrides]

Full reproduction: ``--split train --stage all`` then ``--split test --stage all``.
  train split, all = prep, block, feat, train, decide
  test split,  all = prep, block, feat, predict, write

Each stage is skipped when all its outputs already exist (resume after a crash)
unless ``--force``. Runtime and peak RSS are logged per stage. Stage artifacts
go to ``--artifacts-dir``; the two submission TSVs go to ``--out-dir``. An
artifacts folder is tied to the ``--data-dir`` that produced it, so dev-sample
artifacts are never reused for a full-data run.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.config import Paths
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

STAGE_ORDER: tuple[str, ...] = ("prep", "block", "feat", "train", "predict", "decide", "write")
RUN_META = "run_meta.json"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes:
        name: CLI stage name.
        owner: Owning module and team member (from the plan).
        splits: Splits this stage runs for.
        outputs: Maps (paths, split) to the files/folders the stage produces;
            the stage is skipped when all of them exist.
        run: Stage implementation taking (paths, split).
    """

    name: str
    owner: str
    splits: tuple[str, ...]
    outputs: Callable[[Paths, str], list[Path]]
    run: Callable[[Paths, str], None]


def _not_implemented(name: str, owner: str) -> Callable[[Paths, str], None]:
    """Placeholder run function that names the owning module."""

    def _run(paths: Paths, split: str) -> None:
        """Raise NotImplementedError for this stage."""
        raise NotImplementedError(f"Stage '{name}' ({split}) is not implemented yet; owner: {owner}")

    return _run


def _stage(name: str, owner: str, splits: tuple[str, ...], outputs: Callable[[Paths, str], list[Path]]) -> Stage:
    """Build a Stage whose run function is still a placeholder."""
    return Stage(name, owner, splits, outputs, _not_implemented(name, owner))


BOTH = ("train", "test")
STAGES: dict[str, Stage] = {
    s.name: s
    for s in (
        _stage("prep", "normalize.py (Member 2)", BOTH,
               lambda p, sp: [p.artifacts_dir / f"records_{sp}.parquet"]),
        _stage("block", "blocking.py (Member 3)", BOTH,
               lambda p, sp: [p.artifacts_dir / f"candidates_{sp}.parquet"]),
        _stage("feat", "features.py (Member 4)", BOTH,
               lambda p, sp: [p.artifacts_dir / f"features_{sp}"]),
        _stage("train", "model.py (Member 4)", ("train",),
               lambda p, sp: [p.artifacts_dir / "oof_train.parquet"]),
        _stage("predict", "model.py (Member 4)", ("test",),
               lambda p, sp: [p.artifacts_dir / "pred_test.parquet"]),
        _stage("decide", "decide.py (Member 1)", ("train",),
               lambda p, sp: [p.artifacts_dir / "decision_config.json"]),
        _stage("write", "run_pipeline.py + io_utils.py (Member 1)", ("test",),
               lambda p, sp: [p.output_dir / "matching_results.tsv", p.output_dir / "candidate_pairs.tsv"]),
    )
}

# CLI flag -> config attribute. Stage modules must read ``config.<NAME>`` at
# call time (not ``from src.config import NAME``) so overrides take effect.
CONFIG_OVERRIDES: dict[str, str] = {
    "block_chunk_s1_rows": "BLOCK_CHUNK_S1_ROWS",
    "block_top_k": "BLOCK_TOP_K",
    "max_candidates_per_s1": "MAX_CANDIDATES_PER_S1",
    "feature_chunk_pairs": "FEATURE_CHUNK_PAIRS",
    "lgbm_num_threads": "LGBM_NUM_THREADS",
    "rapidfuzz_workers": "RAPIDFUZZ_WORKERS",
}


def stages_for(stage: str, split: str) -> list[Stage]:
    """Resolve ``--stage`` for a split into an ordered list of stages.

    Args:
        stage: A name from ``STAGE_ORDER`` or ``"all"``.
        split: ``"train"`` or ``"test"``.

    Returns:
        Stages to run, in pipeline order.

    Raises:
        ValueError: If a single named stage does not apply to ``split``.
    """
    if stage == "all":
        return [STAGES[n] for n in STAGE_ORDER if split in STAGES[n].splits]
    s = STAGES[stage]
    if split not in s.splits:
        raise ValueError(f"Stage '{stage}' only runs for split(s) {s.splits}, not '{split}'")
    return [s]


def _check_run_meta(paths: Paths) -> None:
    """Tie the artifacts folder to one data folder; raise on a mismatch.

    Raises:
        ValueError: If the artifacts folder was produced from another data dir.
    """
    meta_file = paths.artifacts_dir / RUN_META
    data_dir = str(paths.data_dir)
    if meta_file.exists():
        previous = json.loads(meta_file.read_text(encoding="utf-8"))["data_dir"]
        if previous != data_dir:
            raise ValueError(
                f"{paths.artifacts_dir} holds artifacts built from {previous}, not {data_dir}; "
                "use a different --artifacts-dir"
            )
        return
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps({"data_dir": data_dir}, indent=2) + "\n", encoding="utf-8", newline="\n")


def run(stage: str, split: str, paths: Paths, force: bool = False) -> None:
    """Run (or skip) the requested stages in order.

    Args:
        stage: Stage name or ``"all"``.
        split: ``"train"`` or ``"test"``.
        paths: Resolved run directories.
        force: Rerun stages even if their outputs exist.

    Raises:
        ValueError: On an invalid stage/split combination or artifacts-folder
            mismatch.
        NotImplementedError: When a stage without an implementation is reached.
    """
    todo = stages_for(stage, split)
    _check_run_meta(paths)
    logger.info("Run: stage=%s split=%s force=%s", stage, split, force)
    logger.info("Paths: data=%s artifacts=%s output=%s", paths.data_dir, paths.artifacts_dir, paths.output_dir)
    for s in todo:
        done = all(o.exists() for o in s.outputs(paths, split))
        logger.info("  %-8s owner=%-42s outputs %s", s.name, s.owner, "exist" if done else "missing")
    for s in todo:
        if not force and all(o.exists() for o in s.outputs(paths, split)):
            logger.info("Stage %s: outputs exist, skipping (use --force to rerun)", s.name)
            continue
        with track_stage(f"{s.name}[{split}]"):
            s.run(paths, split)


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=(*STAGE_ORDER, "all"))
    p.add_argument("--split", required=True, choices=BOTH)
    p.add_argument("--data-dir", type=Path, default=None, help="folder with train/ and test/")
    p.add_argument("--out-dir", type=Path, default=None, help="folder for the submission TSVs")
    p.add_argument("--artifacts-dir", type=Path, default=None, help="folder for stage artifacts and logs")
    p.add_argument("--force", action="store_true", help="rerun stages whose outputs exist")
    p.add_argument("--log-level", default="INFO")
    for flag, name in CONFIG_OVERRIDES.items():
        p.add_argument(f"--{flag.replace('_', '-')}", type=int, default=None,
                       help=f"override config.{name} (default {getattr(config, name)})")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, configure logging and run the pipeline.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if None.
    """
    args = build_parser().parse_args(argv)
    paths = config.get_paths(args.data_dir, args.artifacts_dir, args.out_dir)
    setup_logging(args.log_level, paths.log_dir)
    for flag, name in CONFIG_OVERRIDES.items():
        value = getattr(args, flag)
        if value is not None:
            setattr(config, name, value)
        logger.info("config.%s = %s", name, getattr(config, name))
    run(args.stage, args.split, paths, args.force)


if __name__ == "__main__":
    main()
