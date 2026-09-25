"""Official macro F0.5 scorer, segment breakdowns and blocking recall report.

Per-S1 rules (challenge README):
    truth empty, pred empty         -> 1.0
    truth empty, pred non-empty     -> 0.0
    truth non-empty, pred empty     -> 0.0
    otherwise F0.5 = 1.25PR / (0.25P + R), 0 if P = R = 0
Scores are macro-averaged over ALL requested S1 IDs, singletons included.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.io_utils import require_columns

logger = logging.getLogger(__name__)

_EMPTY: frozenset[str] = frozenset()
RECALL_CHUNK_ROWS = 1_000_000


def f05_single(pred: Collection[str], truth: Collection[str]) -> float:
    """F0.5 for one S1 entity under the official rules.

    Args:
        pred: Predicted S2/S3 IDs (a set or frozenset for O(1) lookups).
        truth: True S2/S3 IDs.

    Returns:
        Score in [0, 1].
    """
    if not truth:
        return 0.0 if pred else 1.0
    if not pred:
        return 0.0
    tp = len(set(pred) & set(truth))
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def _check_known(truth: Mapping[str, Collection[str]], s1_ids: Sequence[str]) -> None:
    """Raise if any requested S1 has no ground-truth entry."""
    unknown = set(s1_ids) - truth.keys()
    if unknown:
        raise ValueError(f"{len(unknown)} S1 id(s) missing from truth, e.g. {sorted(unknown)[:5]}")


def per_s1_f05(
    pred: Mapping[str, Collection[str]],
    truth: Mapping[str, Collection[str]],
    s1_ids: Sequence[str],
) -> NDArray[np.float64]:
    """F0.5 for each S1 in ``s1_ids`` order. An S1 missing from ``pred`` is empty.

    Memory: one float64 per S1.

    Args:
        pred: ``{s1_id: predicted IDs}``.
        truth: ``{s1_id: true IDs}``; must contain every ID in ``s1_ids``.
        s1_ids: S1 IDs to score.

    Returns:
        Array of per-S1 scores aligned with ``s1_ids``.

    Raises:
        ValueError: If an S1 in ``s1_ids`` is missing from ``truth``.
    """
    _check_known(truth, s1_ids)
    return np.fromiter(
        (f05_single(pred.get(s, _EMPTY), truth[s]) for s in s1_ids),
        dtype=np.float64,
        count=len(s1_ids),
    )


def macro_f05(
    pred: Mapping[str, Collection[str]],
    truth: Mapping[str, Collection[str]],
    s1_ids: Sequence[str],
) -> float:
    """Macro-averaged F0.5 over ``s1_ids`` (the leaderboard metric).

    Args:
        pred: ``{s1_id: predicted IDs}``; missing S1 = empty prediction.
        truth: ``{s1_id: true IDs}``; must contain every ID in ``s1_ids``.
        s1_ids: S1 IDs to average over (all S1 of the evaluation set).

    Returns:
        Mean per-S1 F0.5; 0.0 if ``s1_ids`` is empty.

    Raises:
        ValueError: If an S1 in ``s1_ids`` is missing from ``truth``.
    """
    scores = per_s1_f05(pred, truth, s1_ids)
    return float(scores.mean()) if len(scores) else 0.0


def f05_breakdown(
    pred: Mapping[str, Collection[str]],
    truth: Mapping[str, Collection[str]],
    s1_records: pd.DataFrame,
) -> pd.DataFrame:
    """Macro F0.5 overall, for singleton / non-singleton S1, and per country.

    Args:
        pred: ``{s1_id: predicted IDs}``; missing S1 = empty prediction.
        truth: ``{s1_id: true IDs}``; must contain every scored S1.
        s1_records: One row per S1 to score, string columns ``entity_id`` and
            ``country``.

    Returns:
        DataFrame with one row per segment: ``segment`` (str: ``overall``,
        ``singleton``, ``non_singleton``, ``country=<label>``), ``n_s1`` (int),
        ``f05`` (float; NaN for an empty segment).

    Raises:
        ValueError: If a column is missing or an S1 is missing from ``truth``.
    """
    require_columns(s1_records.columns, ("entity_id", "country"), "f05_breakdown s1_records")
    ids = s1_records["entity_id"].tolist()
    scores = per_s1_f05(pred, truth, ids)
    singleton = np.fromiter((not truth[s] for s in ids), dtype=bool, count=len(ids))
    countries = s1_records["country"].to_numpy()

    segments: list[tuple[str, NDArray[np.bool_]]] = [
        ("overall", np.ones(len(ids), dtype=bool)),
        ("singleton", singleton),
        ("non_singleton", ~singleton),
    ]
    segments += [(f"country={c}", countries == c) for c in sorted(set(countries))]
    rows = [
        {"segment": name, "n_s1": int(m.sum()), "f05": float(scores[m].mean()) if m.any() else float("nan")}
        for name, m in segments
    ]
    out = pd.DataFrame(rows)
    for r in rows:
        logger.info("F0.5 %-16s n=%-9d %.4f", r["segment"], r["n_s1"], r["f05"])
    return out


@dataclass(frozen=True)
class RecallReport:
    """Blocking quality over a set of S1 entities.

    Attributes:
        n_true_pairs: True (S1, S2/S3) pairs for the scored S1.
        n_found_pairs: True pairs present in the candidates.
        pair_recall: ``n_found_pairs / n_true_pairs`` (1.0 if no true pairs).
        pct_s1_fully_covered: % of S1 whose true matches are all candidates
            (singletons count as covered).
        avg_candidates_per_s1: Candidate rows per scored S1.
    """

    n_true_pairs: int
    n_found_pairs: int
    pair_recall: float
    pct_s1_fully_covered: float
    avg_candidates_per_s1: float


def blocking_recall_report(
    candidates: pd.DataFrame,
    truth: Mapping[str, Collection[str]],
    s1_ids: Sequence[str],
    chunk_rows: int = RECALL_CHUNK_ROWS,
) -> RecallReport:
    """Measure how many true pairs the blocking candidates contain.

    Candidate rows whose ``s1_id`` is not in ``s1_ids`` are ignored. Memory:
    an index of all true-pair keys (one string per true pair) plus one
    ``chunk_rows`` slice of candidate keys at a time.

    Args:
        candidates: One row per unique (S1, candidate) pair, string columns
            ``s1_id`` and ``cand_id``.
        truth: ``{s1_id: true IDs}``; must contain every ID in ``s1_ids``.
        s1_ids: S1 IDs to report on.
        chunk_rows: Candidate rows processed per step.

    Returns:
        A ``RecallReport``.

    Raises:
        ValueError: If a column is missing or an S1 is missing from ``truth``.
    """
    require_columns(candidates.columns, ("s1_id", "cand_id"), "blocking_recall_report candidates")
    _check_known(truth, s1_ids)
    true_keys = pd.Index([f"{s}\t{c}" for s in s1_ids for c in truth[s]])
    true_sizes = pd.Series({s: len(truth[s]) for s in s1_ids}, dtype="int64")
    s1_index = pd.Index(s1_ids)

    n_cands = 0
    found = pd.Series(dtype="int64")
    for start in range(0, len(candidates), chunk_rows):
        part = candidates.iloc[start : start + chunk_rows]
        part = part[part["s1_id"].isin(s1_index)]
        n_cands += len(part)
        keys = pd.Index(part["s1_id"] + "\t" + part["cand_id"])
        hit = true_keys.get_indexer(keys) >= 0
        found = found.add(part.loc[hit, "s1_id"].value_counts(), fill_value=0)

    found = found.reindex(true_sizes.index, fill_value=0)
    n_true = int(true_sizes.sum())
    n_found = int(found.sum())
    n_s1 = len(s1_ids)
    report = RecallReport(
        n_true_pairs=n_true,
        n_found_pairs=n_found,
        pair_recall=n_found / n_true if n_true else 1.0,
        pct_s1_fully_covered=100.0 * float((found >= true_sizes).mean()) if n_s1 else 0.0,
        avg_candidates_per_s1=n_cands / n_s1 if n_s1 else 0.0,
    )
    logger.info(
        "Blocking recall: pair recall %.4f (%d/%d) | S1 fully covered %.2f%% | avg cands/S1 %.1f",
        report.pair_recall, n_found, n_true, report.pct_s1_fully_covered, report.avg_candidates_per_s1,
    )
    return report
