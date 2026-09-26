"""Stages `decide` (split=train), `write` (split=test), `baseline` (both), `compare` (train).

baseline: candidates_{split}.parquet rrf_score -> oof_train.parquet (train, with
        labels) / pred_test.parquet (test) with a "score" column: the M1 rule
        baseline standing in for train/predict until the model exists.

decide: <artifacts>/oof_train.parquet (s1_id, cand_id, <score col>, label) for ALL
        train pairs -> <artifacts>/decision_config.json. Optional global
        one-owner rule (each S2/S3 ID kept only for its highest-scoring S1),
        then per S1 keep score >= t, empty list if the S1's max score < t_empty.
        t and t_empty are grid-searched for macro F0.5 over ALL train S1 at once;
        the grid is config.DECIDE_GRID_QUANTILES quantiles of the score column
        plus a fixed config.DECIDE_GRID_STEP grid over (0, 1).
        config.DECIDE_METHOD swaps the rule: "per_source" (t_s2 / t_s3 by ID
        prefix, coordinate descent) or "expected_f05" (per-S1 expected-F0.5
        prefix of probability-sorted candidates). config.DECIDE_ONE_OWNER_AUTO
        keeps whichever one-owner setting scores higher.
compare: every method x one-owner on/off on the same oof_train.parquet
        -> <artifacts>/decide_compare.tsv (+ logged table), per-segment F0.5.
write:  <artifacts>/pred_test.parquet + decision_config.json (applied unchanged)
        -> <out>/matching_results.tsv and <out>/candidate_pairs.tsv, one row per
        test S1 (France included), then utils/validate_submission.py must PASS.

The score column is ``config.DECIDE_SCORE_COLUMN``: "prob" (model) or "score"
(rule baseline, until the model exists).

Memory: pairs are held as int32 S1/candidate indices + float32 scores (+ int8
labels), ~13 bytes per pair (~1.4 GB for 110M pairs), preallocated and filled
from parquet in PAIR_BATCH_ROWS batches using only the needed columns. The
one-owner rule uses ufunc.at (no sort); write groups by S1 with one stable
argsort (+8 bytes per pair transient). Threshold search sorts the kept pairs by
score once (~25 bytes per pair transient, 9 retained) and sweeps the grid
incrementally; expected_f05 lexsorts kept pairs by (S1, -score).
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
from src.logging_utils import track_stage

logger = logging.getLogger(__name__)

OWNER = "decide.py (lane A, lead)"
PAIR_BATCH_ROWS = 1_048_576  # = default parquet row-group size
DECISION_CONFIG = "decision_config.json"
ID_PATTERN = r"^S[1-3]-[0-9]{1,12}$"
BASELINE_SOURCE_COLUMN = "rrf_score"  # candidates_{split} column added by blocking (Lane B)
METHODS: tuple[str, ...] = ("threshold", "per_source", "expected_f05")
COMPARE_FILE = "decide_compare.tsv"


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

    On train, when blocking wrote ``blocking_s1_subset_train.parquet``
    (``--block-s1-fraction`` < 1), only those S1 are kept: the others were
    never blocked, so they are excluded from decide (logged).

    Returns:
        ``SplitIds`` (~1 GB for the full data: 12.5M IDs plus hash indexes).

    Raises:
        ValueError: If a required column is missing or IDs are duplicated.
    """
    rec = io_utils.load_parquet(paths.artifacts_dir / f"records_{split}.parquet",
                                columns=["entity_id", "source", "country"])
    is_s1 = (rec["source"] == "S1").to_numpy()
    subset = paths.artifacts_dir / f"blocking_s1_subset_{split}.parquet"
    if split == "train" and subset.exists():
        n_all = int(is_s1.sum())
        is_s1 = is_s1 & rec["entity_id"].isin(io_utils.load_parquet(subset, ["s1_id"])["s1_id"]).to_numpy()
        logger.warning("%s: decide uses %d of %d train S1; %d unblocked S1 excluded",
                       subset.name, int(is_s1.sum()), n_all, n_all - int(is_s1.sum()))
    s1 = pd.Index(rec.loc[is_s1, "entity_id"])
    cand = pd.Index(rec.loc[(rec["source"] != "S1").to_numpy(), "entity_id"])
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


def select(pairs: Pairs, keep: NDArray[np.bool_], max_s1: NDArray[np.float32],
           t: float | NDArray[np.float32], t_empty: float) -> NDArray[np.bool_]:
    """Pairs predicted as matches: kept, score >= t, and the S1's max score >= t_empty.

    ``t`` is one threshold or one per pair (per-source rule).
    """
    return keep & (pairs.score >= t) & (max_s1[pairs.s1] >= t_empty)


def per_s1_scores(pairs: Pairs, sel: NDArray[np.bool_], n_truth: NDArray[np.int64]) -> NDArray[np.float64]:
    """Per-S1 F0.5 of a selection (train pairs must carry labels)."""
    if pairs.label is None:
        raise ValueError("per_s1_scores needs labelled pairs")
    n = len(n_truth)
    n_pred = np.bincount(pairs.s1[sel], minlength=n)
    tp = np.bincount(pairs.s1[sel & (pairs.label == 1)], minlength=n)
    return evaluate.f05_from_counts(tp, n_pred, n_truth)


def threshold_grid(scores: NDArray[np.float32], n_quantiles: int,
                   step: float | None = None) -> tuple[float, ...]:
    """Candidate thresholds: score quantiles UNION a fixed step grid, deduplicated, ascending.

    ``inverted_cdf`` quantiles are actual score values, so ``score >= t`` is exact
    in float32 and the grid follows whatever range the score column has
    (rrf_score ~0.017-0.18). Probabilities pile up near 0, leaving no quantile
    between ~0.02 and ~0.99, so the fixed grid step, 2*step, ..., 1 - step
    (float32) covers that range. Memory: one float32 copy of ``scores`` for
    the partition.

    Args:
        scores: All candidate scores being decided on.
        n_quantiles: Number of evenly spaced quantile levels in [0, 1].
        step: Fixed grid spacing; None reads config.DECIDE_GRID_STEP, 0 disables.

    Returns:
        Sorted unique thresholds.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if len(scores) == 0:
        raise ValueError("threshold_grid needs at least one score")
    step = config.DECIDE_GRID_STEP if step is None else step
    q = np.quantile(scores, np.linspace(0.0, 1.0, n_quantiles), method="inverted_cdf")
    fixed = np.arange(1, round(1 / step)) * step if step > 0 else np.empty(0)
    return tuple(float(v) for v in np.unique(np.concatenate([q, fixed]).astype(np.float32)))


@dataclass(frozen=True)
class _Desc:
    """Labelled pairs of one group sorted by descending score, so ``score >= t`` is a prefix.

    Attributes:
        s1: int32 S1 codes. neg: float32 ``-score`` (ascending). label: int8 labels.
    """

    s1: NDArray[np.int32]
    neg: NDArray[np.float32]
    label: NDArray[np.int8]

    def n_above(self, t: float) -> int:
        """Number of pairs with score >= t."""
        return int(np.searchsorted(self.neg, np.float32(-t), side="right"))

    def counts(self, t: float, n: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(true positives, predictions) per S1 when keeping score >= t."""
        end = self.n_above(t)
        s = self.s1[:end]
        return np.bincount(s[self.label[:end] == 1], minlength=n), np.bincount(s, minlength=n)


def _sort_desc(pairs: Pairs, mask: NDArray[np.bool_]) -> _Desc:
    """Masked labelled pairs sorted by descending score (stable).

    Memory: ~25 bytes per masked pair transient, 9 retained.
    """
    if pairs.label is None:
        raise ValueError("threshold search needs labelled pairs")
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(-pairs.score[idx], kind="stable")]
    return _Desc(s1=pairs.s1[idx], neg=-pairs.score[idx], label=pairs.label[idx])


class _Gate:
    """Best t_empty for given per-S1 counts in O(n_s1): no per-t_empty rescoring.

    An S1 is open (keeps its selection) iff its max kept score >= t_empty, else
    it predicts empty (F0.5 = 1 iff it is a singleton). With S1 sorted by max
    score, macro F0.5 at every t_empty is a suffix sum of (open - closed) gains.
    """

    def __init__(self, max_s1: NDArray[np.float32], n_truth: NDArray[np.int64], te_grid: tuple[float, ...]) -> None:
        """Sort S1 by max kept score once; ``te_grid`` is tried with 0.0 (no gate) prepended."""
        self.order = np.argsort(max_s1, kind="stable")
        self.n_truth = n_truth
        self.closed = (n_truth == 0).astype(np.float64)
        self.te = np.array((0.0, *te_grid), dtype=np.float32)
        # first S1 (in max order) passing each t_empty
        self.first_open = np.searchsorted(max_s1[self.order], self.te, side="left")

    def best(self, tp: NDArray[np.int64], n_pred: NDArray[np.int64]) -> tuple[float, float]:
        """(t_empty, macro F0.5) maximising F0.5; the lowest t_empty wins ties."""
        gain = evaluate.f05_from_counts(tp, n_pred, self.n_truth)[self.order] - self.closed[self.order]
        suffix = np.concatenate((np.cumsum(gain[::-1])[::-1], [0.0]))
        f = (self.closed.sum() + suffix[self.first_open]) / len(self.n_truth)
        i = int(np.argmax(f))
        return float(self.te[i]), float(f[i])


def _sweep(group: _Desc, grid: tuple[float, ...], base_tp: NDArray[np.int64], base_pred: NDArray[np.int64],
           gate: _Gate) -> tuple[float, float, float]:
    """Best (t, t_empty, macro F0.5) for ``group``'s threshold, other counts fixed at ``base_*``.

    Walks t from high to low adding each newly passing slice of the sorted
    group, so the whole grid costs one pass over the group plus O(n_s1) per t.
    The lowest (t, t_empty) wins ties (same as scanning ascending with ``>``).
    """
    n = len(base_tp)
    tp, n_pred = base_tp.copy(), base_pred.copy()
    prev = 0
    best = (grid[0], 0.0, -1.0)
    for t in reversed(grid):
        end = group.n_above(t)
        s = group.s1[prev:end]
        n_pred += np.bincount(s, minlength=n)
        tp += np.bincount(s[group.label[prev:end] == 1], minlength=n)
        prev = end
        te, f = gate.best(tp, n_pred)
        if f >= best[2]:
            best = (t, te, f)
    return best


def grid_search(pairs: Pairs, keep: NDArray[np.bool_], n_truth: NDArray[np.int64],
                t_grid: tuple[float, ...]) -> tuple[float, float, float]:
    """Best (t, t_empty, macro F0.5) over the grid; t_empty also tries 0 (no gate).

    Sorts the kept pairs by score once, then one incremental sweep: O(n log n)
    + O(len(grid) * n_s1). The first best in ascending (t, t_empty) order wins ties.
    """
    n = len(n_truth)
    gate = _Gate(max_score_per_s1(pairs, keep, n), n_truth, t_grid)
    zeros = np.zeros(n, dtype=np.int64)
    return _sweep(_sort_desc(pairs, keep), t_grid, zeros, zeros, gate)


def cand_is_s3(ids: SplitIds) -> NDArray[np.bool_]:
    """True per candidate code whose ID starts with ``S3-`` (source from the ID prefix)."""
    out = np.zeros(len(ids.cand), dtype=bool)
    out[ids.cand_order] = ids.cand_keys // 10**12 == 3
    return out


def per_source_search(pairs: Pairs, keep: NDArray[np.bool_], is_s3: NDArray[np.bool_], n_truth: NDArray[np.int64],
                      t_grid: tuple[float, ...], start: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Best (t_s2, t_s3, t_empty, macro F0.5) by coordinate descent from the global optimum.

    Starting at t_s2 = t_s3 = ``start``'s t, alternately re-sweeps the S2 and
    the S3 threshold (each with t_empty) with the other held fixed, accepting
    only strict improvements, for up to config.DECIDE_CD_ROUNDS rounds. The
    result is never worse than ``start``. Each sweep is one pass over that
    source's pairs + O(len(grid) * n_s1).

    Args:
        pairs: Labelled pairs.
        keep: Pairs eligible (e.g. one-owner mask).
        is_s3: Per pair, True if the candidate is an S3 record.
        n_truth: True matches per S1.
        t_grid: Threshold grid (also used for t_empty).
        start: (t, t_empty, F0.5) from ``grid_search`` on the same ``keep``.

    Returns:
        (t_s2, t_s3, t_empty, macro F0.5).
    """
    n = len(n_truth)
    gate = _Gate(max_score_per_s1(pairs, keep, n), n_truth, t_grid)
    groups = {2: _sort_desc(pairs, keep & ~is_s3), 3: _sort_desc(pairs, keep & is_s3)}
    t = {2: start[0], 3: start[0]}
    te, f = start[1], start[2]
    for rnd in range(config.DECIDE_CD_ROUNDS):
        improved = False
        for src, other in ((2, 3), (3, 2)):
            base_tp, base_pred = groups[other].counts(t[other], n)
            t_new, te_new, f_new = _sweep(groups[src], t_grid, base_tp, base_pred, gate)
            if f_new > f + 1e-12:
                t[src], te, f, improved = t_new, te_new, f_new, True
        logger.info("Per-source round %d: t_s2=%.6g t_s3=%.6g t_empty=%.6g F0.5=%.4f", rnd + 1, t[2], t[3], te, f)
        if not improved:
            break
    return t[2], t[3], te, f


def expected_f05_select(pairs: Pairs, keep: NDArray[np.bool_], n_s1: int) -> NDArray[np.bool_]:
    """Per S1, the probability-sorted prefix of kept candidates with the highest expected F0.5.

    Treating probabilities as independent calibrated Bernoullis, predicting
    nothing scores P(no match) = prod(1 - p); the top-k prefix scores
    1.25 * sum_{i<=k} p_i / (0.25 * sum_all p + k). The empty set wins ties.

    Memory: a lexsort of the kept pairs (~20 bytes per kept pair) plus
    float64 work arrays for config.DECIDE_EXPECTED_CHUNK_PAIRS pairs at a time
    (chunks end on S1 boundaries).

    Args:
        pairs: Pairs whose ``score`` is a probability in [0, 1].
        keep: Pairs eligible (e.g. one-owner mask).
        n_s1: Number of S1 codes (unused beyond validation of codes).

    Returns:
        Boolean selection mask over pairs.
    """
    # ponytail: ratio-of-expectations approximation and truths outside the candidates
    # ignored; an exact Poisson-binomial DP (O(m^2) per S1) if calibration proves good.
    sel = np.zeros(len(pairs.s1), dtype=bool)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return sel
    idx = idx[np.lexsort((-pairs.score[idx], pairs.s1[idx]))]
    s1 = pairs.s1[idx]
    if s1[-1] >= n_s1:
        raise ValueError("S1 code out of range")
    starts = np.flatnonzero(np.concatenate(([True], s1[1:] != s1[:-1])))
    bounds = np.append(starts, len(idx))
    step = max(1, config.DECIDE_EXPECTED_CHUNK_PAIRS)
    g0 = 0
    while g0 < len(starts):
        g1 = max(g0 + 1, int(np.searchsorted(bounds, bounds[g0] + step, side="right")) - 1)
        g1 = min(g1, len(starts))
        lo, hi = int(bounds[g0]), int(bounds[g1])
        p = np.clip(pairs.score[idx[lo:hi]].astype(np.float64), 0.0, 1.0)
        local = bounds[g0:g1] - lo
        grp = np.repeat(np.arange(g1 - g0), np.diff(np.append(local, hi - lo)))
        csum = np.cumsum(p)
        offset = csum[local] - p[local]
        cum = csum - offset[grp]
        total = np.add.reduceat(p, local)
        k = np.arange(hi - lo) - local[grp] + 1
        exp_f = 1.25 * cum / (0.25 * total[grp] + k)
        best = np.maximum.reduceat(exp_f, local)
        k_best = np.minimum.reduceat(np.where(exp_f == best[grp], k, np.iinfo(np.int64).max), local)
        p_empty = np.exp(np.add.reduceat(np.log1p(-np.minimum(p, 1.0 - 1e-12)), local))
        take = (best > p_empty)[grp] & (k <= k_best[grp])
        sel[idx[lo:hi][take]] = True
        g0 = g1
    logger.info("Expected-F0.5 selection keeps %d of %d kept pairs", int(sel.sum()), len(idx))
    return sel


def calibration_table(prob: NDArray[np.float32], label: NDArray[np.int8], n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: pairs in equal-width probability bins (deciles of [0, 1]), logged.

    Args:
        prob: Predicted probabilities, one per pair.
        label: 1 for a true pair.
        n_bins: Number of equal-width bins.

    Returns:
        DataFrame, one row per non-empty bin: ``bin`` (str), ``n`` (int),
        ``mean_prob``, ``pos_rate`` (float). Also logs the expected calibration error.
    """
    b = np.minimum((np.clip(prob, 0.0, 1.0) * n_bins).astype(np.int64), n_bins - 1)
    n = np.bincount(b, minlength=n_bins)
    mean_p = np.bincount(b, weights=prob.astype(np.float64), minlength=n_bins) / np.maximum(n, 1)
    rate = np.bincount(b, weights=label.astype(np.float64), minlength=n_bins) / np.maximum(n, 1)
    table = pd.DataFrame({"bin": [f"[{i / n_bins:.1f},{(i + 1) / n_bins:.1f})" for i in range(n_bins)],
                          "n": n, "mean_prob": mean_p, "pos_rate": rate})[n > 0].reset_index(drop=True)
    for r in table.itertuples():
        logger.info("Calibration %s n=%-10d mean_prob=%.4f pos_rate=%.4f", r.bin, r.n, r.mean_prob, r.pos_rate)
    ece = float((table["n"] * (table["mean_prob"] - table["pos_rate"]).abs()).sum() / max(len(prob), 1))
    logger.info("Calibration: expected calibration error %.4f over %d pairs", ece, len(prob))
    return table


def tune(pairs: Pairs, ids: SplitIds, keep: NDArray[np.bool_], n_truth: NDArray[np.int64], method: str,
         grid: tuple[float, ...]) -> dict[str, float]:
    """Tune one decision method on labelled pairs.

    Returns:
        Rule parameters (``t``, ``t_empty``, plus ``t_s2`` / ``t_s3`` for
        per_source) and ``f05`` (train macro F0.5).

    Raises:
        ValueError: On an unknown method.
    """
    if method == "expected_f05":
        sel = expected_f05_select(pairs, keep, len(n_truth))
        return {"t": 0.0, "t_empty": 0.0, "f05": float(per_s1_scores(pairs, sel, n_truth).mean())}
    if method not in METHODS:
        raise ValueError(f"Unknown decision method {method!r}; expected one of {METHODS}")
    t, te, f = grid_search(pairs, keep, n_truth, grid)
    if method == "threshold":
        return {"t": t, "t_empty": te, "f05": f}
    is_s3 = cand_is_s3(ids)[pairs.cand]
    t2, t3, te2, f2 = per_source_search(pairs, keep, is_s3, n_truth, grid, (t, te, f))
    return {"t": t, "t_empty": te2, "t_s2": t2, "t_s3": t3, "f05": f2}


def apply_rule(pairs: Pairs, ids: SplitIds, keep: NDArray[np.bool_], rule: Mapping[str, object]) -> NDArray[np.bool_]:
    """Selection mask for a tuned rule (a decision_config dict or ``tune`` output).

    Raises:
        ValueError: On an unknown method.
    """
    method = str(rule.get("method", "threshold"))
    if method == "expected_f05":
        return expected_f05_select(pairs, keep, len(ids.s1))
    if method not in METHODS:
        raise ValueError(f"Unknown decision method {method!r}; expected one of {METHODS}")
    t: float | NDArray[np.float32] = float(rule["t"])  # type: ignore[arg-type]  # JSON value
    if method == "per_source":
        t = np.where(cand_is_s3(ids)[pairs.cand], np.float32(rule["t_s3"]), np.float32(rule["t_s2"]))  # type: ignore[arg-type]
    t_empty = float(rule["t_empty"])  # type: ignore[arg-type]
    return select(pairs, keep, max_score_per_s1(pairs, keep, len(ids.s1)), t, t_empty)


def truth_counts(data_dir: Path, ids: SplitIds) -> NDArray[np.int64]:
    """True matches per S1 code from the ground truth (includes matches blocking missed).

    Ground-truth rows of S1 not in ``ids`` (excluded by a blocking S1 subset)
    are ignored.
    """
    gt = io_utils.read_ground_truth_pairs(data_dir / "train" / "train_ground_truth.tsv")
    gt = gt[(gt["cand_id"] != "") & gt["s1_id"].isin(ids.s1)]  # isin: S1 outside a blocking subset
    codes = encode(pa.array(gt["s1_id"]), ids.s1_keys, ids.s1_order, "ground-truth s1_id")
    return np.bincount(codes, minlength=len(ids.s1)).astype(np.int64)


# --------------------------------------------------------------------------- stages


def _load_train(paths: Paths) -> tuple[SplitIds, Pairs, NDArray[np.int64]]:
    """Train IDs, labelled OOF pairs (config.DECIDE_SCORE_COLUMN) and true-match counts per S1."""
    ids = load_split_ids(paths, "train")
    pairs = load_pairs(paths.artifacts_dir / "oof_train.parquet", config.DECIDE_SCORE_COLUMN, ids, with_label=True)
    return ids, pairs, truth_counts(paths.data_dir, ids)


def _grid(pairs: Pairs) -> tuple[float, ...]:
    """Threshold grid (quantiles of the decided score column + fixed step grid), logged."""
    grid = threshold_grid(pairs.score, config.DECIDE_GRID_QUANTILES)
    logger.info("Threshold grid: %d values from %d quantiles of %s + step %g, %.4g .. %.4g",
                len(grid), config.DECIDE_GRID_QUANTILES, config.DECIDE_SCORE_COLUMN, config.DECIDE_GRID_STEP,
                grid[0], grid[-1])
    return grid


def _check_method(method: str, score_col: str) -> None:
    """expected_f05 needs probabilities.

    Raises:
        ValueError: On an unknown method or expected_f05 on a non-probability score.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown decision method {method!r}; expected one of {METHODS}")
    if method == "expected_f05" and score_col != "prob":
        raise ValueError("DECIDE_METHOD=expected_f05 needs probabilities (--decide-score-column prob)")


def run_decide(paths: Paths, log_row: bool = True) -> DecisionConfig:
    """Tune the decision rule on all train pairs, report segments, save decision_config.json.

    The rule is config.DECIDE_METHOD, tuned with and without the one-owner rule;
    config.DECIDE_ONE_OWNER picks one, or with config.DECIDE_ONE_OWNER_AUTO the
    better one (ties keep one-owner), recorded as ``one_owner_auto`` and
    ``f05_by_one_owner``. Non-default methods add ``method`` (and ``t_s2`` /
    ``t_s3``) to decision_config.json; the default output is unchanged.

    Args:
        paths: Run directories (reads artifacts records_train + oof_train and the
            train ground truth).
        log_row: Append the result to the experiment log.

    Returns:
        The saved ``DecisionConfig`` (required keys only).

    Raises:
        ValueError: On missing columns, unknown IDs, an unlabelled input or an
            invalid method.
    """
    score_col, method = config.DECIDE_SCORE_COLUMN, config.DECIDE_METHOD
    _check_method(method, score_col)
    ids, pairs, n_truth = _load_train(paths)
    if score_col == "prob" and pairs.label is not None:
        calibration_table(pairs.score, pairs.label)
    masks = {True: one_owner_mask(pairs, len(ids.cand)), False: np.ones(len(pairs.s1), dtype=bool)}
    grid = _grid(pairs)
    results = {flag: tune(pairs, ids, mask, n_truth, method, grid) for flag, mask in masks.items()}
    for flag, r in results.items():
        logger.info("Best %s with one_owner=%s: %s", method, flag, r)
    auto = config.DECIDE_ONE_OWNER_AUTO
    one_owner = results[True]["f05"] >= results[False]["f05"] if auto else config.DECIDE_ONE_OWNER
    if auto:
        logger.info("Auto one-owner: chose one_owner=%s (F0.5 %.4f vs %.4f)", one_owner,
                    results[one_owner]["f05"], results[not one_owner]["f05"])
    rule = results[one_owner]
    f05 = rule["f05"]
    scores = per_s1_scores(pairs, apply_rule(pairs, ids, masks[one_owner], {**rule, "method": method}), n_truth)
    table = evaluate.segment_table(scores, n_truth == 0, ids.s1_country)

    cfg = DecisionConfig(t=rule["t"], t_empty=rule["t_empty"], one_owner=bool(one_owner), score_column=score_col,
                         train_f05=round(f05, 6))
    saved: dict[str, object] = dict(asdict(cfg))
    if method != "threshold":
        saved["method"] = method
    if method == "per_source":
        saved.update(t_s2=rule["t_s2"], t_s3=rule["t_s3"])
    if auto:
        saved.update(one_owner_auto=True,
                     f05_by_one_owner={str(k).lower(): round(v["f05"], 6) for k, v in results.items()})
    out = paths.artifacts_dir / DECISION_CONFIG
    out.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8", newline="\n")
    logger.info("Saved %s: %s", out, saved)
    if log_row:
        seg = dict(zip(table["segment"], table["f05"], strict=True))
        other = results[not one_owner]["f05"]
        desc = "quantile grid" if method == "threshold" else method
        log_experiment(
            f"A-decide-{pd.Timestamp.now():%Y%m%d-%H%M%S}", "A", f"decide v1 {desc} ({score_col})",
            f05_overall=seg["overall"], f05_singleton=seg.get("singleton"),
            f05_non_singleton=seg.get("non_singleton"), f05_us=seg.get("country=US"),
            f05_india=seg.get("country=India"),
            notes=f"{ {k: v for k, v in saved.items() if k != 'train_f05'} } (other one_owner setting {other:.4f}); "
                  f"data={paths.data_dir.name}; pairs={len(pairs.s1)}",
        )
    return cfg


def run_compare(paths: Paths, split: str = "train") -> None:
    """Tune every decision variant on the same OOF file and tabulate train F0.5 by segment.

    Variants: each method in ``METHODS`` x one-owner on/off (expected_f05 only
    when the score column is "prob"). Writes <artifacts>/decide_compare.tsv
    and logs the table. Does not touch decision_config.json.

    decide_compare.tsv has one row per variant: ``variant`` (str), ``params``
    (str), then one F0.5 column per segment (``overall``, ``singleton``,
    ``non_singleton``, ``country=<label>`` ...).

    Memory: as ``run_decide`` (~13 bytes per pair held, plus sort buffers of
    ~25 bytes per pair while a variant is tuned).

    Args:
        paths: Run directories (as for ``run_decide``).
        split: Must be ``"train"`` (runner signature).

    Raises:
        ValueError: As ``run_decide``.
    """
    score_col = config.DECIDE_SCORE_COLUMN
    ids, pairs, n_truth = _load_train(paths)
    methods = [m for m in METHODS if m != "expected_f05" or score_col == "prob"]
    if score_col == "prob" and pairs.label is not None:
        calibration_table(pairs.score, pairs.label)
    else:
        logger.warning("Score column %r is not a probability: skipping expected_f05", score_col)
    grid = _grid(pairs)
    rows: list[dict[str, object]] = []
    for flag in (True, False):
        keep = one_owner_mask(pairs, len(ids.cand)) if flag else np.ones(len(pairs.s1), dtype=bool)
        for method in methods:
            with track_stage(f"compare {method} one_owner={flag}"):
                rule = tune(pairs, ids, keep, n_truth, method, grid)
                scores = per_s1_scores(pairs, apply_rule(pairs, ids, keep, {**rule, "method": method}), n_truth)
            seg = evaluate.segment_table(scores, n_truth == 0, ids.s1_country)
            params = " ".join(f"{k}={v:.6g}" for k, v in rule.items() if k != "f05" and method != "expected_f05")
            rows.append({"variant": f"{method}{'+one_owner' if flag else ''}", "params": params,
                         **dict(zip(seg["segment"], seg["f05"].round(4), strict=True))})
        del keep
    table = pd.DataFrame(rows)
    out = paths.artifacts_dir / COMPARE_FILE
    table.to_csv(out, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    logger.info("Decision variants on %s (%d pairs, %s):\n%s", paths.artifacts_dir / "oof_train.parquet",
                len(pairs.s1), score_col, table.to_string(index=False))
    logger.info("Wrote %s", out)


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


def run_validator(matching: Path, candidates: Path, test_dir: Path, required: bool = False) -> None:
    """Run utils/validate_submission.py; raise unless it prints PASS.

    Args:
        matching: matching_results.tsv to check.
        candidates: candidate_pairs.tsv to check.
        test_dir: Folder with the test source TSVs.
        required: Raise if the validator script is absent (else warn and skip,
            e.g. when running from the submission zip).

    Raises:
        FileNotFoundError: If ``required`` and the validator script is missing.
        RuntimeError: If the validator does not exit 0 with a PASS line.
    """
    script = config.REPO_ROOT / "utils" / "validate_submission.py"
    if not script.exists():
        if required:
            raise FileNotFoundError(f"Validator not found: {script}")
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
    sel = apply_rule(pairs, ids, keep, cfg)
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
    run_validator(tmp["matching_results.tsv"], tmp["candidate_pairs.tsv"], paths.data_dir / "test")
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
