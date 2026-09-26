"""Stages `decide` (split=train), `write` (split=test) and `baseline` (both).

baseline: candidates_{split}.parquet rrf_score -> oof_train.parquet (train, with
        labels) / pred_test.parquet (test) with a "score" column: the M1 rule
        baseline standing in for train/predict until the model exists.

decide: <artifacts>/oof_train.parquet (s1_id, cand_id, <score col>, label) for ALL
        train pairs -> <artifacts>/decision_config.json. Optional global
        one-owner rule (each S2/S3 ID kept only for its highest-scoring S1),
        then per S1 keep score >= t, empty list if the S1's max score < t_empty.
        t and t_empty are grid-searched for macro F0.5 over ALL train S1 at once;
        the grid is config.DECIDE_GRID_QUANTILES quantiles of the score column.
write:  <artifacts>/pred_test.parquet + decision_config.json (applied unchanged)
        -> <out>/matching_results.tsv and <out>/candidate_pairs.tsv, one row per
        test S1 (France included), then utils/validate_submission.py must PASS.

The score column is ``config.DECIDE_SCORE_COLUMN``: "prob" (model) or "score"
(rule baseline, until the model exists).

Memory: pairs are held as int32 S1/candidate indices + float32 scores (+ int8
labels), ~13 bytes per pair (~1.4 GB for 110M pairs), preallocated and filled
from parquet in PAIR_BATCH_ROWS batches using only the needed columns. The
one-owner rule uses ufunc.at (no sort); write groups by S1 with one stable
argsort (+8 bytes per pair transient).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from src import config, contracts, evaluate, io_utils
from src.config import Paths
from src.experiment_log import log_experiment

logger = logging.getLogger(__name__)

OWNER = "decide.py (lane A, lead)"
PAIR_BATCH_ROWS = 1_048_576  # = default parquet row-group size
DECISION_CONFIG = "decision_config.json"
ID_PATTERN = r"^S[1-3]-[0-9]{1,12}$"
BASELINE_SOURCE_COLUMN = "rrf_score"  # candidates_{split} column added by blocking (Lane B)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class SplitIds:
    """Entity IDs of one split, from records_{split}.parquet.

    Attributes:
        s1: S1 IDs in file order (index position = S1 code).
        cand: S2/S3 IDs in file order (index position = candidate code).
        s1_country: Country label per S1 code.
        s1_keys, cand_keys: Sorted int64 ID keys (see ``id_keys``).
        s1_order, cand_order: Code of each sorted key.
    """

    s1: pd.Index
    cand: pd.Index
    s1_country: NDArray[np.object_]
    s1_keys: NDArray[np.int64]
    s1_order: NDArray[np.int64]
    cand_keys: NDArray[np.int64]
    cand_order: NDArray[np.int64]


@dataclass(frozen=True)
class Pairs:
    """Scored (S1, candidate) pairs as compact arrays, one element per pair.

    Attributes:
        s1: int32 S1 codes.
        cand: int32 candidate codes.
        score: float32 model probability or baseline score.
        label: int8 1 if the candidate is a true match (train only), else None.
    """

    s1: NDArray[np.int32]
    cand: NDArray[np.int32]
    score: NDArray[np.float32]
    label: NDArray[np.int8] | None


@dataclass(frozen=True)
class DecisionConfig:
    """Tuned decision rule, saved as decision_config.json and applied unchanged to test."""

    t: float
    t_empty: float
    one_owner: bool
    score_column: str
    train_f05: float


def load_split_ids(paths: Paths, split: str) -> SplitIds:
    """Read S1 and S2/S3 IDs (and S1 countries) from records_{split}.parquet.

    Args:
        paths: Run directories.
        split: ``"train"`` or ``"test"``.

    Returns:
        ``SplitIds`` (~1 GB for the full data: 12.5M IDs plus hash indexes).

    Raises:
        ValueError: If a required column is missing or IDs are duplicated.
    """
    rec = io_utils.load_parquet(paths.artifacts_dir / f"records_{split}.parquet",
                                columns=["entity_id", "source", "country"])
    is_s1 = (rec["source"] == "S1").to_numpy()
    s1 = pd.Index(rec.loc[is_s1, "entity_id"])
    cand = pd.Index(rec.loc[~is_s1, "entity_id"])
    k1, k2 = id_keys(pa.array(s1)), id_keys(pa.array(cand))
    o1, o2 = np.argsort(k1, kind="stable"), np.argsort(k2, kind="stable")
    if (np.diff(k1[o1]) == 0).any() or (np.diff(k2[o2]) == 0).any():
        raise ValueError(f"records_{split}.parquet: duplicate entity_id values")
    return SplitIds(s1=s1, cand=cand, s1_country=rec.loc[is_s1, "country"].to_numpy(dtype=object),
                    s1_keys=k1[o1], s1_order=o1, cand_keys=k2[o2], cand_order=o2)


def id_keys(ids: pa.Array | pa.ChunkedArray) -> NDArray[np.int64]:
    """Map IDs ``S<n>-<digits>`` to int64 keys ``n * 10**12 + digits``, computed in Arrow.

    Avoids Python string objects entirely (they bloat RSS on 100M-row inputs).

    Raises:
        ValueError: If an ID does not match ``ID_PATTERN``.
    """
    ok = pc.all(pc.match_substring_regex(ids, ID_PATTERN)).as_py()
    if len(ids) and not ok:
        raise ValueError(f"IDs must match {ID_PATTERN}")
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64()).to_numpy()
    out: NDArray[np.int64] = src * 10**12 + num
    return out


def encode(ids: pa.Array | pa.ChunkedArray, sorted_keys: NDArray[np.int64], order: NDArray[np.int64],
           what: str) -> NDArray[np.int32]:
    """Codes of ``ids`` via binary search in sorted keys (no hashing, no Python strings).

    Raises:
        ValueError: If an ID is malformed or not present.
    """
    k = id_keys(ids)
    pos = np.minimum(np.searchsorted(sorted_keys, k), max(len(sorted_keys) - 1, 0))
    missing = int((sorted_keys[pos] != k).sum()) if len(sorted_keys) else len(k)
    if missing:
        raise ValueError(f"{missing} {what} value(s) missing from records")
    return order[pos].astype(np.int32)


def load_pairs(path: Path, score_column: str, ids: SplitIds, with_label: bool) -> Pairs:
    """Read scored pairs into int32/float32 arrays, PAIR_BATCH_ROWS rows at a time.

    Args:
        path: Parquet file or folder with one row per (S1, candidate) pair and
            string columns ``s1_id``, ``cand_id``, float ``<score_column>``
            and, if ``with_label``, int/bool ``label``.
        score_column: ``"prob"`` or ``"score"``.
        ids: Split IDs used to encode the string IDs.
        with_label: Read the ``label`` column (train).

    Returns:
        ``Pairs`` arrays in file order.

    Raises:
        ValueError: If a column is missing or an ID is not in records_{split}.
    """
    cols = ["s1_id", "cand_id", score_column] + (["label"] if with_label else [])
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path}: no parquet files")
    readers = [pq.ParquetFile(f, pre_buffer=False, buffer_size=1 << 20) for f in files]
    for f, r in zip(files, readers, strict=True):
        io_utils.require_columns(r.schema_arrow.names, cols, str(f))
    n = sum(r.metadata.num_rows for r in readers)
    pairs = Pairs(s1=np.empty(n, np.int32), cand=np.empty(n, np.int32), score=np.empty(n, np.float32),
                  label=np.empty(n, np.int8) if with_label else None)
    pos = 0
    # Sequential reads without pre-buffering: with pre_buffer (and the dataset
    # scanner's read-ahead) RSS grew ~70 bytes per pair; this stays flat (~0.35 GB).
    batches = (b for r in readers for b in r.iter_batches(batch_size=PAIR_BATCH_ROWS, columns=cols))
    for batch in batches:
        end = pos + batch.num_rows
        pairs.s1[pos:end] = encode(batch.column("s1_id"), ids.s1_keys, ids.s1_order, f"{path.name} s1_id")
        pairs.cand[pos:end] = encode(batch.column("cand_id"), ids.cand_keys, ids.cand_order, f"{path.name} cand_id")
        pairs.score[pos:end] = batch.column(score_column).to_numpy(zero_copy_only=False)
        if pairs.label is not None:
            pairs.label[pos:end] = batch.column("label").to_numpy(zero_copy_only=False)
        pos = end
    logger.info("Loaded %d scored pairs from %s (%s)", len(pairs.s1), path, score_column)
    return pairs


# --------------------------------------------------------------------------- decision rule


def one_owner_mask(pairs: Pairs, n_cand: int) -> NDArray[np.bool_]:
    """Keep each candidate only for its highest-scoring S1 across the whole split.

    Ties go to the lowest S1 code (deterministic). No sort: a per-candidate
    max score and min tied S1 code via ``ufunc.at`` (two n_cand-sized arrays
    plus a few bytes of masks per pair).

    Args:
        pairs: Scored pairs.
        n_cand: Number of candidate codes (``len(SplitIds.cand)``).

    Returns:
        Boolean mask over pairs.
    """
    best = np.full(n_cand, -np.inf, dtype=np.float32)
    np.maximum.at(best, pairs.cand, pairs.score)
    tie = pairs.score == best[pairs.cand]
    owner = np.full(n_cand, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(owner, pairs.cand[tie], pairs.s1[tie])
    keep: NDArray[np.bool_] = tie & (pairs.s1 == owner[pairs.cand])
    logger.info("One-owner rule keeps %d of %d pairs", int(keep.sum()), len(keep))
    return keep


def max_score_per_s1(pairs: Pairs, keep: NDArray[np.bool_], n_s1: int) -> NDArray[np.float32]:
    """Highest kept score per S1 (-inf for an S1 with no kept pair)."""
    out = np.full(n_s1, -np.inf, dtype=np.float32)
    np.maximum.at(out, pairs.s1[keep], pairs.score[keep])
    return out


def select(pairs: Pairs, keep: NDArray[np.bool_], max_s1: NDArray[np.float32], t: float, t_empty: float) -> NDArray[np.bool_]:
    """Pairs predicted as matches: kept, score >= t, and the S1's max score >= t_empty."""
    return keep & (pairs.score >= t) & (max_s1[pairs.s1] >= t_empty)


def per_s1_scores(pairs: Pairs, sel: NDArray[np.bool_], n_truth: NDArray[np.int64]) -> NDArray[np.float64]:
    """Per-S1 F0.5 of a selection (train pairs must carry labels)."""
    if pairs.label is None:
        raise ValueError("per_s1_scores needs labelled pairs")
    n = len(n_truth)
    n_pred = np.bincount(pairs.s1[sel], minlength=n)
    tp = np.bincount(pairs.s1[sel & (pairs.label == 1)], minlength=n)
    return evaluate.f05_from_counts(tp, n_pred, n_truth)


def threshold_grid(scores: NDArray[np.float32], n_quantiles: int) -> tuple[float, ...]:
    """Candidate thresholds: ``n_quantiles`` quantiles of the scores, deduplicated, ascending.

    ``inverted_cdf`` quantiles are actual score values, so ``score >= t`` is exact
    in float32 and the grid follows whatever range the score column has
    (rrf_score ~0.017-0.18 or probabilities). Memory: one float32 copy of
    ``scores`` for the partition.

    Args:
        scores: All candidate scores being decided on.
        n_quantiles: Number of evenly spaced quantile levels in [0, 1].

    Returns:
        Sorted unique thresholds.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if len(scores) == 0:
        raise ValueError("threshold_grid needs at least one score")
    q = np.quantile(scores, np.linspace(0.0, 1.0, n_quantiles), method="inverted_cdf")
    return tuple(float(v) for v in np.unique(q.astype(np.float32)))


def grid_search(pairs: Pairs, keep: NDArray[np.bool_], n_truth: NDArray[np.int64],
                t_grid: tuple[float, ...]) -> tuple[float, float, float]:
    """Best (t, t_empty, macro F0.5) over the grid; t_empty also tries 0 (no gate).

    One pass over the pairs per t; each t_empty is O(n_s1). The first best
    in ascending (t, t_empty) order wins ties.
    """
    if pairs.label is None:
        raise ValueError("grid_search needs labelled pairs")
    n = len(n_truth)
    max_s1 = max_score_per_s1(pairs, keep, n)
    keep_true = keep & (pairs.label == 1)
    best = (t_grid[0], 0.0, -1.0)
    for t in t_grid:
        above = pairs.score >= t
        n_pred = np.bincount(pairs.s1[keep & above], minlength=n)
        tp = np.bincount(pairs.s1[keep_true & above], minlength=n)
        for t_empty in (0.0, *t_grid):
            gate = max_s1 >= t_empty
            f = float(evaluate.f05_from_counts(np.where(gate, tp, 0), np.where(gate, n_pred, 0), n_truth).mean())
            if f > best[2]:
                best = (t, t_empty, f)
    return best


def truth_counts(data_dir: Path, ids: SplitIds) -> NDArray[np.int64]:
    """True matches per S1 code from the ground truth (includes matches blocking missed).

    Raises:
        ValueError: If the ground truth references an S1 not in records_train.
    """
    gt = io_utils.read_ground_truth_pairs(data_dir / "train" / "train_ground_truth.tsv")
    gt = gt[gt["cand_id"] != ""]
    codes = encode(pa.array(gt["s1_id"]), ids.s1_keys, ids.s1_order, "ground-truth s1_id")
    return np.bincount(codes, minlength=len(ids.s1)).astype(np.int64)


# --------------------------------------------------------------------------- stages


def run_decide(paths: Paths, log_row: bool = True) -> DecisionConfig:
    """Tune t / t_empty on all train pairs, report segments, save decision_config.json.

    Args:
        paths: Run directories (reads artifacts records_train + oof_train and the
            train ground truth).
        log_row: Append the result to the experiment log.

    Returns:
        The saved ``DecisionConfig``.

    Raises:
        ValueError: On missing columns, unknown IDs or an unlabelled input.
    """
    score_col = config.DECIDE_SCORE_COLUMN
    ids = load_split_ids(paths, "train")
    pairs = load_pairs(paths.artifacts_dir / "oof_train.parquet", score_col, ids, with_label=True)
    n_truth = truth_counts(paths.data_dir, ids)
    all_kept = np.ones(len(pairs.s1), dtype=bool)
    owner_kept = one_owner_mask(pairs, len(ids.cand))
    grid = threshold_grid(pairs.score, config.DECIDE_GRID_QUANTILES)
    logger.info("Threshold grid: %d values from %d quantiles of %s, %.4g .. %.4g",
                len(grid), config.DECIDE_GRID_QUANTILES, score_col, grid[0], grid[-1])
    results = {flag: grid_search(pairs, mask, n_truth, grid)
               for flag, mask in ((True, owner_kept), (False, all_kept))}
    for flag, (t, te, f) in results.items():
        logger.info("Grid best with one_owner=%s: t=%.6g t_empty=%.6g macro F0.5=%.4f", flag, t, te, f)
    one_owner = config.DECIDE_ONE_OWNER
    t, t_empty, f05 = results[one_owner]
    keep = owner_kept if one_owner else all_kept
    scores = per_s1_scores(pairs, select(pairs, keep, max_score_per_s1(pairs, keep, len(n_truth)), t, t_empty), n_truth)
    table = evaluate.segment_table(scores, n_truth == 0, ids.s1_country)

    cfg = DecisionConfig(t=t, t_empty=t_empty, one_owner=one_owner, score_column=score_col, train_f05=round(f05, 6))
    out = paths.artifacts_dir / DECISION_CONFIG
    out.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8", newline="\n")
    logger.info("Saved %s: %s", out, asdict(cfg))
    if log_row:
        seg = dict(zip(table["segment"], table["f05"], strict=True))
        other = results[not one_owner][2]
        log_experiment(
            f"A-decide-{pd.Timestamp.now():%Y%m%d-%H%M%S}", "A", f"decide v1 quantile grid ({score_col})",
            f05_overall=seg["overall"], f05_singleton=seg.get("singleton"),
            f05_non_singleton=seg.get("non_singleton"), f05_us=seg.get("country=US"),
            f05_india=seg.get("country=India"),
            notes=f"t={t} t_empty={t_empty} one_owner={one_owner} (other setting {other:.4f}); "
                  f"data={paths.data_dir.name}; pairs={len(pairs.s1)}",
        )
    return cfg


class _SlicedIds(Mapping[str, list[str]]):
    """Read-only S1 -> candidate-ID list view over pairs sorted by S1 code (no copies per S1)."""

    def __init__(self, s1: pd.Index, s1_sorted: NDArray[np.int32], cand_sorted: NDArray[np.int32],
                 names: NDArray[np.object_]) -> None:
        """Index the sorted pairs by S1 code."""
        self._s1 = s1
        self._starts = np.searchsorted(s1_sorted, np.arange(len(s1) + 1))
        self._cand = cand_sorted
        self._names = names

    def __getitem__(self, key: str) -> list[str]:
        """Candidate IDs for one S1 (empty list if none)."""
        i = self._s1.get_loc(key)
        if not isinstance(i, int):
            raise KeyError(key)
        return list(self._names[self._cand[self._starts[i] : self._starts[i + 1]]])

    def __iter__(self) -> Iterator[str]:
        """All S1 IDs."""
        return iter(self._s1)

    def __len__(self) -> int:
        """Number of S1."""
        return len(self._s1)


def _run_validator(matching: Path, candidates: Path, test_dir: Path) -> None:
    """Run utils/validate_submission.py; raise unless it prints PASS.

    Raises:
        RuntimeError: If the validator does not exit 0 with a PASS line.
    """
    script = config.REPO_ROOT / "utils" / "validate_submission.py"
    if not script.exists():
        logger.warning("Validator %s not found (e.g. running from the submission zip); skipping", script)
        return
    res = subprocess.run(
        [sys.executable, str(script), "--matching", str(matching), "--candidate", str(candidates),
         "--test-dir", str(test_dir)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    for line in res.stdout.splitlines():
        logger.info("validator | %s", line)
    passed = res.returncode == 0 and any(line.startswith("PASS") for line in res.stdout.splitlines())
    if not passed:
        raise RuntimeError(f"validate_submission.py did NOT pass (exit {res.returncode}):\n{res.stdout}{res.stderr}")


def run_write(paths: Paths) -> None:
    """Apply decision_config.json to test predictions and write both submission TSVs.

    The config is ``config.DECISION_CONFIG_PATH`` if set (``--decision-config``,
    e.g. tuned in another artifacts dir), else ``<artifacts>/decision_config.json``.
    Files are written as ``*.tmp``, validated, then renamed, so a failed
    validation never leaves outputs that the runner would skip on rerun.

    Raises:
        FileNotFoundError: If the decision config file does not exist.
        ValueError: If decision_config.json lacks a contract key or inputs are malformed.
        RuntimeError: If the validator does not PASS.
    """
    cfg_path = config.DECISION_CONFIG_PATH or paths.artifacts_dir / DECISION_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Decision config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    io_utils.require_columns(cfg.keys(), contracts.DECISION_CONFIG_KEYS, str(cfg_path))
    logger.info("Using decision config %s (%s): %s", cfg_path.resolve(),
                "--decision-config" if config.DECISION_CONFIG_PATH else "artifacts default", cfg)
    ids = load_split_ids(paths, "test")
    pairs = load_pairs(paths.artifacts_dir / "pred_test.parquet", cfg["score_column"], ids, with_label=False)
    keep = one_owner_mask(pairs, len(ids.cand)) if cfg["one_owner"] else np.ones(len(pairs.s1), dtype=bool)
    sel = select(pairs, keep, max_score_per_s1(pairs, keep, len(ids.s1)), cfg["t"], cfg["t_empty"])
    s1_list = ids.s1.tolist()
    n_pairs, n_sel = len(pairs.s1), int(sel.sum())
    n_empty = int((np.bincount(pairs.s1[sel], minlength=len(s1_list)) == 0).sum())
    del keep

    # Group by S1 (stable: candidates keep file order); drop inputs as soon as possible.
    order = np.argsort(pairs.s1, kind="stable")
    s1_sorted, cand_sorted, sel_sorted = pairs.s1[order], pairs.cand[order], sel[order]
    del order, sel, pairs
    names = ids.cand.to_numpy(dtype=object)
    cand_map = _SlicedIds(ids.s1, s1_sorted, cand_sorted, names)
    match_map = _SlicedIds(ids.s1, s1_sorted[sel_sorted], cand_sorted[sel_sorted], names)

    out = {"candidate_pairs.tsv": (cand_map, io_utils.CANDIDATE_HEADER),
           "matching_results.tsv": (match_map, io_utils.MATCHING_HEADER)}
    tmp = {name: paths.output_dir / f"{name}.tmp" for name in out}
    for name, (mapping, header) in out.items():
        io_utils.write_id_list_tsv(mapping, s1_list, tmp[name], header)
    logger.info("Test: %d S1, %d candidate pairs, %d matches, %d S1 with empty match list",
                len(s1_list), n_pairs, n_sel, n_empty)
    _run_validator(tmp["matching_results.tsv"], tmp["candidate_pairs.tsv"], paths.data_dir / "test")
    for name, path in tmp.items():
        path.replace(paths.output_dir / name)
    logger.info("Wrote %s and %s", paths.output_dir / "matching_results.tsv", paths.output_dir / "candidate_pairs.tsv")


def run_baseline(paths: Paths, split: str) -> None:
    """M1 rule baseline: blocking's RRF score becomes the decision score (no model).

    Streams candidates_{split}.parquet (``s1_id``, ``cand_id``, float
    ``BASELINE_SOURCE_COLUMN``; one row per pair) PAIR_BATCH_ROWS rows at a
    time and writes the file ``decide``/``write`` read with
    ``--decide-score-column score``:

    * train -> oof_train.parquet: s1_id, cand_id (str), fold (int8, -1 = no
      CV), score (float32), label (int8, 1 if the pair is in the ground truth).
    * test  -> pred_test.parquet: s1_id, cand_id (str), score (float32).

    Memory: one batch of pairs plus, on train, the records ID index (~1 GB at
    full scale) and one int64 key per true pair; independent of pair count.

    Args:
        paths: Run directories (reads artifacts candidates_{split}, and on
            train records_train + the train ground truth).
        split: ``"train"`` or ``"test"``.

    Raises:
        ValueError: If ``rrf_score`` (or another needed column) is missing, or
            a ground-truth ID is not in records_train.
    """
    src = paths.artifacts_dir / f"candidates_{split}.parquet"
    reader = pq.ParquetFile(src, pre_buffer=False, buffer_size=1 << 20)
    if BASELINE_SOURCE_COLUMN not in reader.schema_arrow.names:
        raise ValueError(f"{src} has no '{BASELINE_SOURCE_COLUMN}' column: Lane B's blocking PR that adds "
                         f"{BASELINE_SOURCE_COLUMN} must be merged and the block stage rerun (--force)")
    io_utils.require_columns(reader.schema_arrow.names, ["s1_id", "cand_id"], str(src))
    train = split == "train"
    fields = [pa.field("s1_id", pa.string()), pa.field("cand_id", pa.string())]
    if train:
        ids = load_split_ids(paths, "train")
        n_cand = len(ids.cand)
        gt = io_utils.read_ground_truth_pairs(paths.data_dir / "train" / "train_ground_truth.tsv")
        gt = gt[gt["cand_id"] != ""]
        s1c = encode(pa.array(gt["s1_id"]), ids.s1_keys, ids.s1_order, "ground-truth s1_id")
        cc = encode(pa.array(gt["cand_id"]), ids.cand_keys, ids.cand_order, "ground-truth cand_id")
        truth = np.sort(s1c.astype(np.int64) * n_cand + cc)
        del gt, s1c, cc
        fields.append(pa.field("fold", pa.int8()))
    fields.append(pa.field("score", pa.float32()))
    if train:
        fields.append(pa.field("label", pa.int8()))
    schema = pa.schema(fields)

    out = paths.artifacts_dir / ("oof_train.parquet" if train else "pred_test.parquet")
    tmp = out.with_name(out.name + ".tmp")
    n = n_pos = 0
    with pq.ParquetWriter(tmp, schema) as writer:
        for b in reader.iter_batches(batch_size=PAIR_BATCH_ROWS, columns=["s1_id", "cand_id", BASELINE_SOURCE_COLUMN]):
            cols: dict[str, pa.Array] = {"s1_id": pc.cast(b.column("s1_id"), pa.string()),
                                         "cand_id": pc.cast(b.column("cand_id"), pa.string())}
            if train:
                key = (encode(b.column("s1_id"), ids.s1_keys, ids.s1_order, "candidates s1_id").astype(np.int64)
                       * n_cand + encode(b.column("cand_id"), ids.cand_keys, ids.cand_order, "candidates cand_id"))
                pos = np.minimum(np.searchsorted(truth, key), max(len(truth) - 1, 0))
                label = (truth[pos] == key).astype(np.int8) if len(truth) else np.zeros(len(key), np.int8)
                n_pos += int(label.sum())
                cols["fold"] = pa.array(np.full(b.num_rows, -1, np.int8))
            cols["score"] = pc.cast(b.column(BASELINE_SOURCE_COLUMN), pa.float32())
            if train:
                cols["label"] = pa.array(label)
            writer.write_table(pa.table(cols, schema=schema))
            n += b.num_rows
            del b, cols
            pa.default_memory_pool().release_unused()  # mimalloc keeps freed pages: 3.1 -> 2.6 GB peak at 60M pairs
    tmp.replace(out)
    if train:
        logger.info("Baseline: %d train pairs, %d positive (%.1f%% of %d true pairs) -> %s",
                    n, n_pos, 100 * n_pos / max(len(truth), 1), len(truth), out)
    else:
        logger.info("Baseline: %d test pairs -> %s", n, out)


def run_stage(paths: Paths, split: str) -> None:
    """Tune thresholds (split=train) or write the submission TSVs (split=test).

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.
    """
    if split == "train":
        run_decide(paths)
    else:
        run_write(paths)
