"""Error report (Phase 3): where the tuned decision loses F0.5 on train OOF.

Usage (from code/business_entity_resolution/, after ``--stage decide --split train``):
    python -m src.error_report --data-dir PATH --artifacts-dir PATH --n-examples 15

Reads ``<artifacts>/records_train.parquet``, ``<artifacts>/oof_train.parquet``,
``<artifacts>/decision_config.json``, ``<data>/train/train_ground_truth.tsv`` and,
if present, ``blocking_s1_subset_train.parquet`` (unblocked S1 are ignored,
exactly as in decide.py). The decision is re-applied with decide.py's own
functions (one-owner rule, t / t_empty or per-source thresholds). Writes
``<artifacts>/error_report/report.txt``.

Buckets (one per wrong pair):
- ``B1``: predicted match for a singleton S1 (false merge).
- ``B2``: wrong extra match on an S1 that has true matches.
- ``B4_threshold`` / ``B4_one_owner``: true pair among the candidates but not
  predicted, because the rule rejected it (score below t / t_empty gate) or the
  one-owner rule gave the candidate to another S1.
Blocking misses (true pairs never scored) are in ``src.blocking_misses``.

F0.5 points lost: per S1, the macro-F0.5 gain from fixing that bucket alone
(dropping all its wrong matches for B1/B2, adding all its in-candidate true
matches for B4), split evenly over the S1's pairs in the bucket and divided by
the number of S1, so bucket rows add up across countries and sources.

Memory: as decide.py (~13 bytes per OOF pair) plus a table of the wrong pairs;
names/addresses are read only for the printed examples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from src import config, decide, evaluate, io_utils
from src.config import Paths
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

BUCKETS: tuple[str, ...] = ("B1", "B2", "B4_threshold", "B4_one_owner")


def _decision(paths: Paths) -> tuple[decide.SplitIds, decide.Pairs, NDArray[np.bool_], NDArray[np.bool_],
                                     NDArray[np.int64]]:
    """Re-apply decision_config.json to oof_train.parquet with decide.py's functions.

    Returns:
        (ids, labelled pairs, one-owner keep mask, selection mask, true matches per S1).

    Raises:
        ValueError: On missing columns, unknown IDs or an invalid method.
    """
    cfg = json.loads((paths.artifacts_dir / decide.DECISION_CONFIG).read_text(encoding="utf-8"))
    ids = decide.load_split_ids(paths, "train")  # drops S1 outside the blocking subset
    n_all = pq.read_table(paths.artifacts_dir / "records_train.parquet", columns=["source"],
                          filters=[("source", "=", "S1")]).num_rows
    logger.info("error_report: analysing %d of %d train S1; %d unblocked S1 excluded",
                len(ids.s1), n_all, n_all - len(ids.s1))
    pairs = decide.load_pairs(paths.artifacts_dir / "oof_train.parquet", str(cfg["score_column"]), ids,
                              with_label=True)
    keep = decide.one_owner_mask(pairs, len(ids.cand)) if cfg["one_owner"] else np.ones(len(pairs.s1), bool)
    sel = decide.apply_rule(pairs, ids, keep, cfg)
    return ids, pairs, keep, sel, decide.truth_counts(paths.data_dir, ids)


def error_pairs(ids: decide.SplitIds, pairs: decide.Pairs, keep: NDArray[np.bool_], sel: NDArray[np.bool_],
                n_truth: NDArray[np.int64]) -> tuple[pd.DataFrame, float]:
    """Bucket every wrong pair and attribute the F0.5 points it costs.

    Args:
        ids: Split IDs.
        pairs: Labelled OOF pairs.
        keep: One-owner mask used by the decision (all True if off).
        sel: Predicted pairs.
        n_truth: True matches per S1 code.

    Returns:
        (errors, macro F0.5). ``errors`` has one row per wrong pair: ``s1``,
        ``cand`` (int32 codes), ``bucket`` (one of ``BUCKETS``), ``prob``
        (float32), ``points`` (float64 macro-F0.5 points), ``country``, ``source``
        ("S2"/"S3").

    Raises:
        ValueError: If pairs are unlabelled.
    """
    if pairs.label is None:
        raise ValueError("error_report needs labelled pairs")
    n = len(n_truth)
    lab = pairs.label == 1
    tp = np.bincount(pairs.s1[sel & lab], minlength=n)
    n_pred = np.bincount(pairs.s1[sel], minlength=n)
    fn = np.bincount(pairs.s1[~sel & lab], minlength=n)
    cur = evaluate.f05_from_counts(tp, n_pred, n_truth)
    gain_fp = (evaluate.f05_from_counts(tp, tp, n_truth) - cur) / np.maximum(n_pred - tp, 1)
    gain_fn = (evaluate.f05_from_counts(tp + fn, n_pred + fn, n_truth) - cur) / np.maximum(fn, 1)

    wrong_fp, missed = sel & ~lab, ~sel & lab
    idx = np.flatnonzero(wrong_fp | missed)
    s1, fp = pairs.s1[idx], wrong_fp[idx]
    bucket = np.where(fp, np.where(n_truth[s1] == 0, "B1", "B2"),
                      np.where(keep[idx], "B4_threshold", "B4_one_owner"))
    errors = pd.DataFrame({
        "s1": s1, "cand": pairs.cand[idx], "bucket": pd.Categorical(bucket, categories=BUCKETS),
        "prob": pairs.score[idx], "points": np.where(fp, gain_fp[s1], gain_fn[s1]) / max(n, 1),
        "country": ids.s1_country[s1],
        "source": np.where(decide.cand_is_s3(ids)[pairs.cand[idx]], "S3", "S2"),
    })
    return errors, float(cur.mean()) if n else 1.0


def bucket_table(errors: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Wrong pairs, affected S1 and F0.5 points lost per bucket (x ``by`` column), with a B4 total."""
    keys = ["bucket"] + ([by] if by else [])
    g = errors.groupby(keys, observed=True).agg(n_pairs=("s1", "size"), n_s1=("s1", "nunique"),
                                                f05_points_lost=("points", "sum"))
    b4 = errors[errors["bucket"].str.startswith("B4")].assign(bucket="B4")
    g4 = b4.groupby(keys, observed=True).agg(n_pairs=("s1", "size"), n_s1=("s1", "nunique"),
                                             f05_points_lost=("points", "sum"))
    return pd.concat([g, g4]).round({"f05_points_lost": 4})


def _examples(paths: Paths, ids: decide.SplitIds, pairs: decide.Pairs, errors: pd.DataFrame,
              n_examples: int) -> list[str]:
    """Random wrong pairs per bucket side by side (names from records, only for the sampled IDs)."""
    samples = {b: errors[errors["bucket"].str.startswith(b)] for b in ("B1", "B2", "B4")}
    samples = {b: d.sample(n=min(n_examples, len(d)), random_state=config.SEED) for b, d in samples.items()}
    ex = pd.concat(samples.values())
    if ex.empty:
        return ["(no errors)"]
    # rank of each example within its S1 (1 = highest score)
    sub = np.isin(pairs.s1, ex["s1"].unique())
    rk = pd.DataFrame({"s1": pairs.s1[sub], "cand": pairs.cand[sub], "p": pairs.score[sub]})
    rk["rank"] = rk.groupby("s1")["p"].rank(method="min", ascending=False).astype(int)
    rk["n_cands"] = rk.groupby("s1")["s1"].transform("size")
    gt = io_utils.read_ground_truth_pairs(paths.data_dir / "train" / "train_ground_truth.tsv")
    s1_str = pd.Series(ids.s1[samples["B2"]["s1"].to_numpy()])
    truth = gt[gt["s1_id"].isin(s1_str) & (gt["cand_id"] != "")].groupby("s1_id")["cand_id"].apply(list)
    need = set(ids.s1[ex["s1"].to_numpy()]) | set(ids.cand[ex["cand"].to_numpy()]) | {c for v in truth for c in v}
    rec = pq.read_table(paths.artifacts_dir / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw"],
                        filters=[("entity_id", "in", sorted(need))]).to_pandas().set_index("entity_id")

    def show(label: str, eid: str) -> list[str]:
        """Two lines: name and address of one record."""
        r = rec.loc[eid]
        return [f"    {label:<5} {eid:<16} name: {r['name_raw']}", f"    {'':<22} addr: {r['addr_raw']}"]

    lines: list[str] = []
    for b, d in samples.items():
        lines += ["", f"== Examples {b} ({len(d)} random, seed {config.SEED})"]
        d = d.merge(rk, on=["s1", "cand"], how="left").assign(
            s1_id=lambda x: ids.s1[x["s1"].to_numpy()], cand_id=lambda x: ids.cand[x["cand"].to_numpy()])
        for i, r in enumerate(d.itertuples(), 1):
            s1_id, cand_id = str(r.s1_id), str(r.cand_id)
            lines.append(f"[{i}] {s1_id} -> {cand_id}  {r.bucket}  {r.country} {r.source}  prob={r.prob:.4f}  "
                         f"rank={r.rank}/{r.n_cands}  points={r.points:.2e}")
            lines += show("S1", s1_id) + show("cand", cand_id)
            for t in truth.get(s1_id, [])[:3] if b == "B2" else []:
                lines += show("true", t)
    return lines


def run(paths: Paths, n_examples: int) -> Path:
    """Build the error report for the tuned train decision.

    Args:
        paths: Run directories (artifacts + ``data_dir/train`` ground truth).
        n_examples: Random examples printed per bucket.

    Returns:
        Path of ``<artifacts_dir>/error_report/report.txt``.

    Raises:
        ValueError: On missing columns, unknown IDs or an invalid decision config.
    """
    ids, pairs, keep, sel, n_truth = _decision(paths)
    errors, f05 = error_pairs(ids, pairs, keep, sel, n_truth)
    head = [
        "Error report: train OOF under decision_config.json",
        f"S1 analysed {len(ids.s1)}; macro F0.5 {f05:.4f}; wrong pairs {len(errors)}",
        "f05_points_lost = macro-F0.5 gain if that bucket alone were fixed (B4 = B4_threshold + B4_one_owner)",
        "", "== Overall", bucket_table(errors, None).to_string(),
        "", "== Per country", bucket_table(errors, "country").to_string(),
        "", "== Per candidate source", bucket_table(errors, "source").to_string(),
    ]
    report = "\n".join(head + _examples(paths, ids, pairs, errors, n_examples)) + "\n"
    out = paths.artifacts_dir / "error_report" / "report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8", newline="\n")
    for line in head:
        logger.info("%s", line)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train/train_ground_truth.tsv")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="folder with oof_train + decision_config")
    parser.add_argument("--n-examples", type=int, default=15, help="random examples per bucket")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    paths = config.get_paths(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    setup_logging(args.log_level, paths.log_dir)
    with fulldata_lock(paths.data_dir, "src.error_report " + " ".join(sys.argv[1:])), track_stage("error_report"):
        run(paths, args.n_examples)


if __name__ == "__main__":
    main()
