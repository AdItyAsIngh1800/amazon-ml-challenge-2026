"""Build a small, fixed-seed development sample of the training split.

About ``frac`` of train S1 are drawn, stratified by country x singleton flag.
All their true S2/S3 matches are kept, plus random other S2/S3 records as
distractors so that (S2+S3) : S1 matches the full training data. Output is
written in the original TSV format under ``<out_dir>/train/`` so any stage
can point at it with ``--data-dir <out_dir>``.

The sample has fewer near-duplicate distractors than the real data, so it
overstates precision: use it for debugging only, never for scores or
thresholds.

Usage (from code/business_entity_resolution/):
    python -m src.make_dev_sample --data-dir PATH [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, io_utils
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

DEV_SAMPLE_FRAC = 0.10


def _source_path(root: Path, s: int) -> Path:
    """Path of ``train_source{s}.tsv`` under ``root/train``."""
    return root / "train" / f"train_source{s}.tsv"


def build_dev_sample(data_dir: Path, out_dir: Path, frac: float = DEV_SAMPLE_FRAC, seed: int = config.SEED) -> None:
    """Sample train S1 with their matches and distractors; write TSVs.

    Loads the full train split into memory (about 2 GB for the real data).

    Args:
        data_dir: Folder containing ``train/train_source{1,2,3}.tsv`` and
            ``train/train_ground_truth.tsv``.
        out_dir: Destination; files are written to ``out_dir/train/`` with
            the original names.
        frac: Fraction of S1 to sample within each country x singleton stratum.
        seed: Random seed for S1 and distractor sampling.

    Raises:
        ValueError: If ``frac`` is not in (0, 1] or an input file fails the
            ``io_utils`` checks.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    s1 = io_utils.read_source(_source_path(data_dir, 1))
    s2 = io_utils.read_source(_source_path(data_dir, 2))
    s3 = io_utils.read_source(_source_path(data_dir, 3))
    truth = io_utils.read_ground_truth(data_dir / "train" / "train_ground_truth.tsv")

    singleton = s1["entity_id"].map(lambda s: not truth[s])
    strata = s1["country"] + "|" + singleton.map({True: "singleton", False: "matched"})
    picked = s1.groupby(strata, sort=True).sample(frac=frac, random_state=seed).index
    s1_keep = s1.index.isin(picked)  # keep original file order
    s1_out = s1[s1_keep]
    s1_ids = s1_out["entity_id"].tolist()

    matched: set[str] = set().union(*(truth[s] for s in s1_ids))
    ids23 = pd.concat([s2["entity_id"], s3["entity_id"]], ignore_index=True)
    is_matched = ids23.isin(matched).to_numpy()
    ratio = len(ids23) / len(s1)
    n_extra = max(0, round(ratio * len(s1_out)) - int(is_matched.sum()))
    pool = np.flatnonzero(~is_matched)
    rng = np.random.default_rng(seed)
    keep23 = is_matched.copy()
    keep23[rng.choice(pool, size=min(n_extra, len(pool)), replace=False)] = True
    s2_out = s2[keep23[: len(s2)]]
    s3_out = s3[keep23[len(s2) :]]

    io_utils.write_source_tsv(s1_out, _source_path(out_dir, 1))
    io_utils.write_source_tsv(s2_out, _source_path(out_dir, 2))
    io_utils.write_source_tsv(s3_out, _source_path(out_dir, 3))
    io_utils.write_id_list_tsv(
        truth, s1_ids, out_dir / "train" / "train_ground_truth.tsv", io_utils.GROUND_TRUTH_COLUMNS
    )

    n23 = len(s2_out) + len(s3_out)
    logger.info("Dev sample S1: %d of %d (%.1f%%)", len(s1_out), len(s1), 100 * len(s1_out) / len(s1))
    logger.info("Dev sample S2: %d | S3: %d | true matches kept: %d | distractors: %d",
                len(s2_out), len(s3_out), int(is_matched.sum()), n23 - int(is_matched.sum()))
    logger.info("(S2+S3):S1 ratio sample %.3f vs full %.3f", n23 / max(len(s1_out), 1), ratio)
    logger.info("Singleton rate sample %.4f vs full %.4f", singleton[s1_keep].mean(), singleton.mean())
    full_c = s1["country"].value_counts(normalize=True)
    samp_c = s1_out["country"].value_counts(normalize=True)
    for c in full_c.index:
        logger.info("Country %-8s share sample %.4f vs full %.4f", c, samp_c.get(c, 0.0), full_c[c])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="folder with train/ (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <artifacts>/dev_sample")
    parser.add_argument("--frac", type=float, default=DEV_SAMPLE_FRAC)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    paths = config.get_paths(data_dir=args.data_dir)
    setup_logging(args.log_level, paths.log_dir)
    out_dir = args.out_dir.resolve() if args.out_dir else paths.artifacts_dir / "dev_sample"
    with fulldata_lock(paths.data_dir, "src.make_dev_sample " + " ".join(sys.argv[1:])), track_stage("make_dev_sample"):
        build_dev_sample(paths.data_dir, out_dir, args.frac, args.seed)


if __name__ == "__main__":
    main()
