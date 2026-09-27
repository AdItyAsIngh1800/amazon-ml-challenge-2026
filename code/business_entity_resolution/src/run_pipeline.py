"""Single entry point for the entity-resolution pipeline.

Usage (from code/business_entity_resolution/):
    python -m src.run_pipeline --stage {prep,block,feat,train,predict,decide,write,baseline,compare,all}
        --split {train,test} --data-dir PATH --out-dir PATH
        [--artifacts-dir PATH] [--force] [--log-level INFO] [config overrides]

Full reproduction: ``--split train --stage all`` then ``--split test --stage all``.
  train split, all = prep, block, feat, train, decide
  test split,  all = prep, block, feat, predict, write
M1 rule baseline (not part of all): ``--stage baseline`` on either split writes
oof_train / pred_test from blocking's rrf_score in place of train / predict;
then decide with ``--decide-score-column score`` and write as usual.
``--stage compare --split train`` tunes every decision variant on the same
oof_train and writes decide_compare.tsv (overall / singleton / per-country F0.5).

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
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src import blocking, config, decide, features, model, normalize
from src.config import Paths
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

STAGE_ORDER: tuple[str, ...] = ("prep", "block", "feat", "train", "predict", "decide", "write")
EXTRA_STAGES: tuple[str, ...] = ("baseline", "compare")  # runnable by name only, never part of "all"
RUN_META = "run_meta.json"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes:
        name: CLI stage name.
        owner: Owning module and lane.
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


BOTH = ("train", "test")
# Stage wiring only. Implementations live in each owner's module as
# run_stage(paths, split); teammates never edit this file. model.py and
# decide.py own two stages each and branch on the split.
STAGES: dict[str, Stage] = {
    s.name: s
    for s in (
        Stage("prep", normalize.OWNER, BOTH,
              lambda p, sp: [p.artifacts_dir / f"records_{sp}.parquet"], normalize.run_stage),
        Stage("block", blocking.OWNER, BOTH,
              lambda p, sp: [p.artifacts_dir / f"candidates_{sp}.parquet"], blocking.run_stage),
        Stage("feat", features.OWNER, BOTH,
              lambda p, sp: [p.artifacts_dir / f"features_{sp}"], features.run_stage),
        Stage("train", model.OWNER, ("train",),
              lambda p, sp: [p.artifacts_dir / "oof_train.parquet"], model.run_stage),
        Stage("predict", model.OWNER, ("test",),
              lambda p, sp: [p.artifacts_dir / "pred_test.parquet"], model.run_stage),
        Stage("decide", decide.OWNER, ("train",),
              lambda p, sp: [p.artifacts_dir / "decision_config.json"], decide.run_stage),
        Stage("write", decide.OWNER, ("test",),
              lambda p, sp: [p.output_dir / "matching_results.tsv", p.output_dir / "candidate_pairs.tsv"],
              decide.run_stage),
        Stage("baseline", decide.OWNER, BOTH,
              lambda p, sp: [p.artifacts_dir / ("oof_train.parquet" if sp == "train" else "pred_test.parquet")],
              decide.run_baseline),
        Stage("compare", decide.OWNER, ("train",),
              lambda p, sp: [p.artifacts_dir / decide.COMPARE_FILE], decide.run_compare),
    )
}

# CLI flag -> config attribute. Stage modules must read ``config.<NAME>`` at
# call time (not ``from src.config import NAME``) so overrides take effect.
CONFIG_OVERRIDES: dict[str, str] = {
    "block_chunk_s1_rows": "BLOCK_CHUNK_S1_ROWS",
    "block_top_k": "BLOCK_TOP_K",
    "block_workers": "BLOCK_WORKERS",
    "max_candidates_per_s1": "MAX_CANDIDATES_PER_S1",
    "feature_chunk_pairs": "FEATURE_CHUNK_PAIRS",
    "lgbm_num_threads": "LGBM_NUM_THREADS",
    "rapidfuzz_workers": "RAPIDFUZZ_WORKERS",
    "train_max_rows": "TRAIN_MAX_ROWS",
}


def stages_for(stage: str, split: str) -> list[Stage]:
    """Resolve ``--stage`` for a split into an ordered list of stages.

    Args:
        stage: A name from ``STAGE_ORDER`` / ``EXTRA_STAGES`` or ``"all"``.
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
    p.add_argument("--stage", required=True, choices=(*STAGE_ORDER, *EXTRA_STAGES, "all"))
    p.add_argument("--split", required=True, choices=BOTH)
    p.add_argument("--data-dir", type=Path, default=None, help="folder with train/ and test/")
    p.add_argument("--out-dir", type=Path, default=None, help="folder for the submission TSVs")
    p.add_argument("--artifacts-dir", type=Path, default=None, help="folder for stage artifacts and logs")
    p.add_argument("--force", action="store_true", help="rerun stages whose outputs exist")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--decide-score-column", choices=("prob", "score"), default=None,
                   help=f"decision input column: model 'prob' or rule-baseline 'score' "
                        f"(default {config.DECIDE_SCORE_COLUMN})")
    p.add_argument("--decision-config", type=Path, default=None,
                   help="write (test): apply this decision_config.json instead of <artifacts>/decision_config.json")
    p.add_argument("--decide-method", choices=decide.METHODS, default=None,
                   help=f"decision rule tuned by decide (default {config.DECIDE_METHOD})")
    p.add_argument("--decide-one-owner-auto", action="store_true",
                   help="decide: keep whichever one-owner setting scores higher")
    p.add_argument("--decide-one-owner-delta", type=float, default=None,
                   help="decide: soft one-owner, drop a candidate from an S1 only if another S1 scores it more "
                        "than this higher (default: hard one-owner rule)")
    for flag, name in CONFIG_OVERRIDES.items():
        p.add_argument(f"--{flag.replace('_', '-')}", type=int, default=None,
                       help=f"override config.{name} (default {getattr(config, name)})")
    p.add_argument("--block-s1-fraction", type=float, default=None,
                   help="train only: block a stratified (country x singleton) fraction of S1; the rest get no "
                        f"candidates and are excluded from training and decide (default {config.BLOCK_S1_FRACTION})")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, configure logging and run the pipeline.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.block_s1_fraction is not None and args.split != "train":
        parser.error("--block-s1-fraction is train-only; the test split always blocks every S1")
    paths = config.get_paths(args.data_dir, args.artifacts_dir, args.out_dir)
    setup_logging(args.log_level, paths.log_dir)
    for flag, name in CONFIG_OVERRIDES.items():
        value = getattr(args, flag)
        if value is not None:
            setattr(config, name, value)
        logger.info("config.%s = %s", name, getattr(config, name))
    if args.block_s1_fraction is not None:
        config.BLOCK_S1_FRACTION = args.block_s1_fraction
    logger.info("config.BLOCK_S1_FRACTION = %s", config.BLOCK_S1_FRACTION)
    if args.decide_score_column is not None:
        config.DECIDE_SCORE_COLUMN = args.decide_score_column
    logger.info("config.DECIDE_SCORE_COLUMN = %s", config.DECIDE_SCORE_COLUMN)
    if args.decide_method is not None:
        config.DECIDE_METHOD = args.decide_method
    if args.decide_one_owner_auto:
        config.DECIDE_ONE_OWNER_AUTO = True
    if args.decide_one_owner_delta is not None:
        config.DECIDE_ONE_OWNER_DELTA = args.decide_one_owner_delta
    logger.info("config.DECIDE_METHOD = %s, DECIDE_ONE_OWNER_AUTO = %s, DECIDE_ONE_OWNER_DELTA = %s",
                config.DECIDE_METHOD, config.DECIDE_ONE_OWNER_AUTO, config.DECIDE_ONE_OWNER_DELTA)
    if args.decision_config is not None:
        config.DECISION_CONFIG_PATH = args.decision_config.resolve()
        logger.info("config.DECISION_CONFIG_PATH = %s", config.DECISION_CONFIG_PATH)
    command = "src.run_pipeline " + " ".join(sys.argv[1:] if argv is None else argv)
    with fulldata_lock(paths.data_dir, command):
        run(args.stage, args.split, paths, args.force)


if __name__ == "__main__":
    main()
