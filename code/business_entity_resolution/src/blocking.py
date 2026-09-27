"""Stage `block`: TF-IDF candidate passes, union, cap and reverse features.

Reads:  <artifacts>/records_{split}.parquet (contracts.RECORDS_COLUMNS)
Writes: <artifacts>/candidates_{split}.parquet (contracts.CANDIDATES_COLUMNS)
        <artifacts>/blocking_recall_{split}.tsv (train only, recall report)
        <artifacts>/blocking_capped_out_{split}.parquet (train only: ``s1_id,
        cand_id`` of true pairs in the uncapped union but beyond the cap, for
        blocking_misses.py)

Passes are data (``PASSES``): each is a TF-IDF vectorizer over one or more
records columns joined with " | " (``pass_text``), fitted per country on that country's S1+S2+S3 records, returning the
top-K S2/S3 records per S1 within the same country (EDA E5: 100% of true pairs
share the country label, so there is no cross-country pass). Adding a pass
is one ``PassSpec`` entry; it adds the columns
``pass_<name>_score`` / ``pass_<name>_rank``. Contract passes with no spec
(C, F) are written as all-NaN columns.

Top-K without a full similarity matrix (sparse_dot_topn is not an allowed
dependency, so plain scipy):
1. Query-side pruning: each S1 row keeps only its n-grams with document
   frequency <= ``max_df`` x pool size, but always its ``min_query_terms``
   rarest ones, so names made only of common n-grams still get candidates.
   Candidate rows are not pruned.
2. The pruned S1 rows are multiplied with the candidate matrix in chunks sized
   by an nnz upper bound (sum of posting lengths per row), so each product
   holds at most ``BLOCK_MAX_PRODUCT_NNZ`` non-zeros whatever the data size.
3. Per S1 the ``rescore_k`` best partial scores are rescored with the exact
   cosine of the full vectors and the top K kept (the pass's ``top_k``, or
   ``config.BLOCK_TOP_K`` for every pass when that is set).

Output columns (one row per (S1, candidate) pair, at most
``config.MAX_CANDIDATES_PER_S1`` per S1, kept by reciprocal rank fusion of
the pass ranks, see ``union_passes``):
    s1_id, cand_id, cand_source (str); country_match (bool);
    pass_<X>_score (float32 cosine, NaN if pass X did not return the pair);
    pass_<X>_rank (float32, 1 = best within the S1, NaN if not returned);
    n_passes (int8); best_block_score (float32, max pass score);
    rrf_score (float32, reciprocal rank fusion value the cap ranks by);
    rev_n_s1 (int32, S1s that kept this candidate), rev_rank (int32, 1 = this
    S1 has the candidate's highest best_block_score), rev_gap (float32, that
    highest score minus this S1's score). Reverse features use blocking scores
    over all S1 of the split, never labels.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
import multiprocessing.pool
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, contracts, evaluate, experiment_log, io_utils, logging_utils
from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "blocking.py (lane block)"


@dataclass(frozen=True)
class PassSpec:
    """One TF-IDF blocking pass.

    Attributes:
        name: Pass letter; output columns are ``pass_<name>_score/_rank``.
        columns: records columns joined with " | " into the pass text; rows
            empty in every column are skipped.
        analyzer: ``"char"`` (character n-grams) or ``"word"`` (whitespace
            tokens of the normalised text).
        ngram: n-gram length.
        top_k: Candidates kept per S1 (overridden by ``config.BLOCK_TOP_K``).
        max_df: Query n-grams with document frequency above this share of the
            candidate pool are dropped from the S1 side ...
        min_query_terms: ... unless they are among the row's this-many rarest.
        rescore_k: Best partial scores rescored with the exact cosine.
    """

    name: str
    columns: tuple[str, ...]
    analyzer: str
    ngram: int
    top_k: int
    max_df: float
    min_query_terms: int
    rescore_k: int = 200


# Settings from the dev-sample sweep (see PR): name char3 max_df/min terms
# 0.001/4 keeps ~99% of the best recall at half the product size; address word
# tokens beat address char 3-grams (India 0.900 vs 0.853, US 0.928 vs 0.883).
# Pass A text is name_norm | name_norm | name_key (Lane C, experiment norm-001:
# beats name_key alone and name_norm alone on both countries). The 3x longer
# text needs min_query_terms 16: at 4, pruning drops the key n-grams and pass A
# recall falls below name_norm alone (US 0.752 vs 0.792); at 16 it is 0.867.
PASSES: tuple[PassSpec, ...] = (
    PassSpec("A", ("name_norm", "name_norm", "name_key"), "char", 3, top_k=50, max_df=0.001, min_query_terms=16),
    PassSpec("B", ("addr_norm",), "word", 1, top_k=50, max_df=0.003, min_query_terms=3),
)
CONTRACT_PASSES: tuple[str, ...] = ("A", "B", "C", "F")
BLOCK_MAX_PRODUCT_NNZ: int = 50_000_000  # ~0.6 GB of float32 + int32 per product
REPORT_CAPS: tuple[int, ...] = (20, 30, 50, 80)
# The cap keeps candidates by reciprocal rank fusion of the pass ranks: on the
# dev sample union@50 pair recall is 0.977 with RRF vs 0.951 by max score
# (name and address cosines are on different scales).
RRF_K: float = 10.0
EXPERIMENT_ID = "block-v1"

RECORD_COLUMNS: tuple[str, ...] = (
    "entity_id", "source", "country", *sorted({c for p in PASSES for c in p.columns}),
)
_GB = 1024**3


@dataclass
class PassResult:
    """Top-K pairs of one pass for one country, sorted by (s1, rank).

    Attributes:
        s1: Global record index of the S1 (int32).
        cand: Global record index of the candidate (int32).
        score: Exact cosine (float32).
        rank: 1-based rank within the S1 (int32).
    """

    s1: NDArray[np.int32]
    cand: NDArray[np.int32]
    score: NDArray[np.float32]
    rank: NDArray[np.int32]


@dataclass
class _Recall:
    """Recall counters summed over chunks (inputs of evaluate.RecallReport)."""

    n_s1: int = 0
    n_true: int = 0
    n_found: int = 0
    n_covered: float = 0.0
    n_cands: float = 0.0

    def add(self, rep: evaluate.RecallReport, n_s1: int) -> None:
        """Accumulate one chunk's report covering ``n_s1`` S1."""
        self.n_s1 += n_s1
        self.n_true += rep.n_true_pairs
        self.n_found += rep.n_found_pairs
        self.n_covered += rep.pct_s1_fully_covered * n_s1 / 100.0
        self.n_cands += rep.avg_candidates_per_s1 * n_s1

    def row(self) -> dict[str, float | int]:
        """Aggregated metrics as a report row."""
        n = max(self.n_s1, 1)
        return {
            "n_s1": self.n_s1, "n_true_pairs": self.n_true, "n_found_pairs": self.n_found,
            "pair_recall": self.n_found / self.n_true if self.n_true else 1.0,
            "pct_s1_fully_covered": 100.0 * self.n_covered / n,
            "avg_cands_per_s1": self.n_cands / n,
        }


@dataclass
class _Report:
    """Recall counters per (country, variant); variant = pass, union or cap."""

    cells: dict[tuple[str, str], _Recall] = field(default_factory=dict)

    def add(self, country: str, variant: str, rep: evaluate.RecallReport, n_s1: int) -> None:
        """Add a chunk's report to its country and to ``ALL``."""
        for c in (country, "ALL"):
            self.cells.setdefault((c, variant), _Recall()).add(rep, n_s1)

    def frame(self) -> pd.DataFrame:
        """One row per (country, variant) with the aggregated metrics."""
        return pd.DataFrame([{"country": c, "variant": v, **r.row()} for (c, v), r in self.cells.items()])


# --------------------------------------------------------------------------- top-K core


def prune_query(q: sp.csr_matrix, df: NDArray[np.int64], max_df_count: float, min_terms: int) -> sp.csr_matrix:
    """Drop common n-grams from S1 rows, keeping each row's rarest ``min_terms``.

    Args:
        q: S1 TF-IDF rows (float32 CSR).
        df: Document frequency of every column in the candidate pool.
        max_df_count: Columns with ``df`` above this are dropped ...
        min_terms: ... unless among the row's ``min_terms`` lowest-df columns.

    Returns:
        A pruned copy of ``q`` (same shape, weights unchanged).
    """
    q = q.tocsr()
    lengths = np.diff(q.indptr)
    row = np.repeat(np.arange(q.shape[0]), lengths)
    d = df[q.indices]
    order = np.lexsort((q.indices, d, row))
    pos = np.arange(q.nnz) - np.repeat(q.indptr[:-1], lengths)
    keep = np.empty(q.nnz, dtype=bool)
    keep[order] = (d[order] <= max_df_count) | (pos < min_terms)
    indptr = np.r_[0, np.cumsum(np.bincount(row[keep], minlength=q.shape[0]))]
    return sp.csr_matrix((q.data[keep], q.indices[keep], indptr), shape=q.shape)


def _chunk_bounds(row_nnz: NDArray[np.float64], budget: int, max_rows: int) -> list[tuple[int, int]]:
    """Split rows into consecutive chunks with sum(row_nnz) <= budget (min 1 row)."""
    bounds: list[tuple[int, int]] = []
    csum = np.cumsum(row_nnz)
    start, n = 0, len(row_nnz)
    while start < n:
        base = csum[start - 1] if start else 0.0
        end = int(np.searchsorted(csum, base + budget, side="right"))
        end = min(max(end, start + 1), start + max_rows, n)
        bounds.append((start, end))
        start = end
    return bounds


def pass_top_k(spec: PassSpec) -> int:
    """Effective K of a pass: ``config.BLOCK_TOP_K`` if set, else ``spec.top_k``."""
    return spec.top_k if config.BLOCK_TOP_K is None else config.BLOCK_TOP_K


def pass_text(rec: pd.DataFrame, spec: PassSpec) -> pd.Series:
    """Pass input text: ``spec.columns`` joined with " | " (as name_recall.py).

    Args:
        rec: Records with every column in ``spec.columns`` (str).

    Returns:
        str Series aligned with ``rec``; "" where every column is empty, so
        those rows are skipped by the pass.
    """
    cols = list(spec.columns)
    text = rec[cols[0]]
    for c in cols[1:]:
        text = text + " | " + rec[c]
    return text.where((rec[cols] != "").any(axis=1), "")


def top_k_pairs(q: sp.csr_matrix, c: sp.csr_matrix, spec: PassSpec, country: str = "") -> tuple[
    NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]
]:
    """Top-K candidates per query row by cosine, without a full similarity matrix.

    Memory: one sparse product of at most ``BLOCK_MAX_PRODUCT_NNZ`` non-zeros
    (or one row) plus ``rescore_k`` gathered pairs per row of that chunk, per
    worker. With ``config.BLOCK_WORKERS`` > 1 the S1 chunks run in a fork-based
    process pool: workers inherit ``q``, ``c`` and their pruned/transposed
    forms copy-on-write (read-only, never rebuilt per chunk), and results are
    collected in chunk order, so the output is identical to the serial run.

    Args:
        q: L2-normalised query rows (S1), float32 CSR, shape (n_q, V).
        c: L2-normalised candidate rows, float32 CSR, shape (n_c, V).
        spec: Pass settings (top_k, max_df, min_query_terms, rescore_k);
            ``config.BLOCK_TOP_K`` replaces ``top_k`` when not None.
        country: Country label for the progress log lines only.

    Returns:
        ``(q_idx, c_idx, score, rank)``: local row indices into ``q`` and
        ``c``, exact cosine and 1-based rank, sorted by (q_idx, rank). Pairs
        with cosine 0 are never returned; ties break on the lower ``c_idx``.
    """
    top_k = pass_top_k(spec)
    rescore_k = max(spec.rescore_k, top_k)
    empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.int32))
    if q.shape[0] == 0 or c.shape[0] == 0:
        return empty
    df = np.bincount(c.indices, minlength=c.shape[1]).astype(np.int64)
    qp = prune_query(q, df, spec.max_df * c.shape[0], spec.min_query_terms)
    ct = c.T.tocsr()
    row_nnz = np.asarray(qp.astype(bool).astype(np.float64) @ df.astype(np.float64)).ravel()
    chunks = _chunk_bounds(row_nnz, BLOCK_MAX_PRODUCT_NNZ, config.BLOCK_CHUNK_S1_ROWS)
    out: list[_Chunk] = []
    max_nnz = 0
    every = max(1, len(chunks) // 20)  # progress line about every 5% of chunks
    workers = min(config.BLOCK_WORKERS, len(chunks))
    global _WORK
    _WORK = _ChunkWork(q, c, qp, ct, top_k, rescore_k)
    start = time.perf_counter()
    pool: multiprocessing.pool.Pool | None = None
    results: Iterator[tuple[_Chunk, int]]
    try:
        if workers > 1:
            gc.freeze()  # keep the GC from touching (and so copying) inherited objects in the workers
            pool = multiprocessing.get_context("fork").Pool(workers)
            results = pool.imap(_chunk_top_k, chunks)  # imap yields in chunk order = serial order
        else:
            results = map(_chunk_top_k, chunks)
        for i, (res, nnz) in enumerate(results, 1):
            max_nnz = max(max_nnz, nnz)
            if len(res[0]):
                out.append(res)
            if i % every == 0 or i == len(chunks):
                elapsed = time.perf_counter() - start
                logger.info("pass %s country %s: chunk %d/%d, elapsed %.0fs, ETA %.0fs (%d workers)", spec.name,
                            country, i, len(chunks), elapsed, elapsed / i * (len(chunks) - i), workers)
    finally:
        if pool is not None:
            pool.terminate()
        gc.unfreeze()
        _WORK = None
    logger.info("Pass %s: %d S1 x %d candidates, %d product chunks, max product nnz %.1fM, RSS %.2f GB",
                spec.name, q.shape[0], c.shape[0], len(chunks), max_nnz / 1e6, psutil.Process().memory_info().rss / _GB)
    if not out:
        return empty
    qi, ci, sc, rk = (np.concatenate(x) for x in zip(*out, strict=True))
    return qi, ci, sc, rk


_Chunk = tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]


@dataclass(frozen=True)
class _ChunkWork:
    """Read-only inputs of ``_chunk_top_k``, set before the pool forks."""

    q: sp.csr_matrix
    c: sp.csr_matrix
    qp: sp.csr_matrix
    ct: sp.csr_matrix
    top_k: int
    rescore_k: int


_WORK: _ChunkWork | None = None  # module global so forked workers inherit it without pickling


def _chunk_top_k(bounds: tuple[int, int]) -> tuple[_Chunk, int]:
    """Top-K of S1 rows ``[a, b)`` of ``_WORK`` (see ``top_k_pairs``) and the product nnz.

    Runs in the parent (serial) or a forked worker; the same code either way.
    """
    w = _WORK
    assert w is not None
    a, b = bounds
    prod = (w.qp[a:b] @ w.ct).tocsr()
    nnz = int(prod.nnz)
    rows, cols = _best_per_row(prod, w.rescore_k)
    del prod
    if not len(rows):
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.int32)), nnz
    rows += a
    exact = np.asarray(w.q[rows].multiply(w.c[cols]).sum(axis=1), dtype=np.float32).ravel()
    order = np.lexsort((cols, -exact, rows))
    rows, cols, exact = rows[order], cols[order], exact[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
    rank = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)])) + 1
    keep = (rank <= w.top_k) & (exact > 0)
    return (rows[keep].astype(np.int32), cols[keep].astype(np.int32), exact[keep], rank[keep].astype(np.int32)), nnz


def _best_per_row(prod: sp.csr_matrix, k: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(row, col) of the up-to-``k`` largest entries of each CSR row (unordered)."""
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    indptr, indices, data = prod.indptr, prod.indices, prod.data
    for i in range(prod.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if b == a:
            continue
        idx = indices[a:b]
        if b - a > k:
            idx = idx[np.argpartition(-data[a:b], k - 1)[:k]]
        rows.append(np.full(len(idx), i, dtype=np.int64))
        cols.append(idx.astype(np.int64))
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def _vectorizer(spec: PassSpec) -> TfidfVectorizer:
    """TF-IDF vectorizer for a pass (float32, l2 norm, smooth idf)."""
    if spec.analyzer == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(spec.ngram, spec.ngram),
                               lowercase=False, dtype=np.float32)
    return TfidfVectorizer(analyzer=spec.analyzer, ngram_range=(spec.ngram, spec.ngram),
                           lowercase=False, dtype=np.float32)


def run_pass(texts: pd.Series, s1_pos: NDArray[np.int32], cand_pos: NDArray[np.int32], spec: PassSpec,
             country: str = "") -> PassResult:
    """Run one pass for one country.

    Args:
        texts: The pass text (``pass_text``) for the country's records, indexed by global
            record index (str; empty = no signal, skipped on both sides).
        s1_pos: Global record indices of the country's S1 (ascending).
        cand_pos: Global record indices of the country's S2/S3.
        spec: Pass settings.
        country: Country label for log lines only.

    Returns:
        Pairs with global indices, sorted by (s1, rank).
    """
    ok = texts.to_numpy() != ""
    all_pos = texts.index.to_numpy()
    with logging_utils.track_stage(f"pass {spec.name} country {country} TF-IDF fit"):
        x = _vectorizer(spec).fit_transform(texts[ok].tolist()).tocsr()
    row_of = pd.Series(np.arange(int(ok.sum())), index=all_pos[ok])
    q_pos = s1_pos[ok[np.searchsorted(all_pos, s1_pos)]]
    c_pos = cand_pos[ok[np.searchsorted(all_pos, cand_pos)]]
    q, c = x[row_of[q_pos].to_numpy()], x[row_of[c_pos].to_numpy()]
    del x  # keep only the S1 / candidate slices during top-K
    qi, ci, sc, rk = top_k_pairs(q, c, spec, country)
    return PassResult(q_pos[qi].astype(np.int32), c_pos[ci].astype(np.int32), sc, rk)


# --------------------------------------------------------------------------- union


def union_passes(results: dict[str, PassResult], n_records: int) -> pd.DataFrame:
    """Union pass pairs; add per-pass score/rank, n_passes, best score, union rank.

    Args:
        results: ``{pass name: PassResult}`` (any S1 subset).
        n_records: Total records (for the int64 pair key).

    Returns:
        One row per unique (s1, cand): int32 ``s1, cand``; float32
        ``pass_<X>_score, pass_<X>_rank`` for every pass in ``results`` (NaN
        when absent); int8 ``n_passes``; float32 ``best_block_score``; float32
        ``rrf_score`` (reciprocal rank fusion sum_X 1 / (RRF_K + pass_X_rank));
        int32 ``union_rank`` (1 = best in the S1 by ``rrf_score``, ties by best
        score then cand; ranked on the float64 sum before the float32 cast).
        Sorted by (s1, union_rank).
    """
    keys = {n: r.s1.astype(np.int64) * n_records + r.cand for n, r in results.items()}
    union = np.unique(np.concatenate(list(keys.values()))) if keys else np.empty(0, np.int64)
    out = pd.DataFrame({"s1": (union // n_records).astype(np.int32), "cand": (union % n_records).astype(np.int32)})
    scores = np.full((len(union), max(len(results), 1)), np.nan, dtype=np.float32)
    rrf = np.zeros(len(union), dtype=np.float64)
    for j, (n, r) in enumerate(results.items()):
        pos = np.searchsorted(union, keys[n])
        scores[pos, j] = r.score
        rank = np.full(len(union), np.nan, dtype=np.float32)
        rank[pos] = r.rank
        rrf[pos] += 1.0 / (RRF_K + r.rank)
        out[f"pass_{n}_score"] = scores[:, j]
        out[f"pass_{n}_rank"] = rank
    out["n_passes"] = (~np.isnan(scores)).sum(axis=1).astype(np.int8)
    out["best_block_score"] = np.fmax.reduce(scores, axis=1)
    out["rrf_score"] = rrf.astype(np.float32)
    order = np.lexsort((out["cand"].to_numpy(), -out["best_block_score"].to_numpy(), -rrf, out["s1"].to_numpy()))
    out = out.iloc[order].reset_index(drop=True)
    s1 = out["s1"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(s1)) + 1] if len(s1) else np.empty(0, np.int64)
    out["union_rank"] = (np.arange(len(s1)) - np.repeat(starts, np.diff(np.r_[starts, len(s1)])) + 1).astype(np.int32)
    return out


def reverse_features(s1: NDArray[np.int32], cand: NDArray[np.int32], score: NDArray[np.float32]) -> tuple[
    NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]
]:
    """Reverse features of each pair over all S1 that kept the same candidate.

    Memory: one int64 permutation plus a few arrays of len(pairs).

    Args:
        s1: S1 index per pair.
        cand: Candidate index per pair.
        score: ``best_block_score`` per pair.

    Returns:
        ``(rev_n_s1, rev_rank, rev_gap)`` aligned with the input: number of
        S1 holding the candidate, 1-based rank of this S1 among them by score
        (ties by S1 index), and the candidate's best score minus this score.
    """
    n = len(cand)
    order = np.lexsort((s1, -score, cand))
    c = cand[order]
    starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1] if n else np.empty(0, np.int64)
    sizes = np.diff(np.r_[starts, n])
    rev_n = np.empty(n, np.int32)
    rev_rank = np.empty(n, np.int32)
    rev_gap = np.empty(n, np.float32)
    rev_n[order] = np.repeat(sizes, sizes)
    rev_rank[order] = np.arange(n) - np.repeat(starts, sizes) + 1
    rev_gap[order] = np.repeat(score[order][starts], sizes) - score[order]
    return rev_n, rev_rank, rev_gap


# --------------------------------------------------------------------------- stage


def _output_schema() -> pa.Schema:
    """Arrow schema of candidates_{split}.parquet (contract passes + extra passes)."""
    names = [*CONTRACT_PASSES, *(p.name for p in PASSES if p.name not in CONTRACT_PASSES)]
    fields = [("s1_id", pa.string()), ("cand_id", pa.string()), ("cand_source", pa.string()),
              ("country_match", pa.bool_())]
    for n in names:
        fields += [(f"pass_{n}_score", pa.float32()), (f"pass_{n}_rank", pa.float32())]
    fields += [("n_passes", pa.int8()), ("best_block_score", pa.float32()), ("rrf_score", pa.float32()),
               ("rev_n_s1", pa.int32()), ("rev_rank", pa.int32()), ("rev_gap", pa.float32())]
    return pa.schema(fields)


def _load_records(path: Path) -> pd.DataFrame:
    """Load the needed records columns, sorted by (country, source, entity_id).

    Sorting makes every country's S1 a contiguous block of global indices.

    Raises:
        ValueError: If a required column is missing or a column is not str.
    """
    rec = io_utils.load_parquet(path, RECORD_COLUMNS)
    for col in RECORD_COLUMNS:
        if not (pd.api.types.is_string_dtype(rec[col]) or pd.api.types.is_object_dtype(rec[col])):
            raise ValueError(f"{path}: column {col} must be str, got {rec[col].dtype}")
    rec = rec.sort_values(["country", "source", "entity_id"], kind="stable", ignore_index=True)
    for col in sorted({c for p in PASSES for c in p.columns}):
        n_empty = int(rec[col].eq("").sum())
        if n_empty:
            logger.warning("%d records with empty %s are skipped by passes on that column", n_empty, col)
    return rec


def _truth_index(paths: Paths, split: str, entity_ids: pd.Series) -> pd.DataFrame | None:
    """True pairs as global indices sorted by s1 (``s1, cand`` int64), or None.

    Only the train split has ground truth; pairs whose IDs are not in the
    records are dropped with a WARNING.
    """
    gt = paths.data_dir / split / f"{split}_ground_truth.tsv"
    if not gt.exists():
        logger.info("No ground truth at %s: recall report skipped", gt)
        return None
    pairs = io_utils.read_ground_truth_pairs(gt)
    ids = pd.Index(entity_ids)  # hash index freed on return
    s1 = ids.get_indexer(pd.Index(pairs["s1_id"]))
    cand = ids.get_indexer(pd.Index(pairs["cand_id"]))  # "" (singleton) -> -1
    unknown_s1 = int((s1 < 0).sum())
    if unknown_s1:
        logger.warning("%d ground-truth rows with an S1 id not in records", unknown_s1)
    t = pd.DataFrame({"s1": s1, "cand": cand})
    t = t[t["s1"] >= 0].sort_values("s1", kind="stable", ignore_index=True)
    return t


def _chunk_recall(
    report: _Report, country: str, union: pd.DataFrame, s1_idx: NDArray[np.int64],
    truth: pd.DataFrame, ids: pa.StringArray,
) -> None:
    """Add one S1 chunk's recall (per pass, union, union at caps) via evaluate.py."""
    ts = truth["s1"].to_numpy()
    part = truth.iloc[np.searchsorted(ts, s1_idx[0]): np.searchsorted(ts, s1_idx[-1], side="right")]
    def _take(idx: NDArray[np.integer]) -> list[str]:
        """Entity IDs of global record indices."""
        return ids.take(pa.array(idx)).to_pylist()  # type: ignore[no-any-return]  # pyarrow is untyped

    s1_ids = _take(s1_idx)
    tr: dict[str, set[str]] = {s: set() for s in s1_ids}
    found = part[part["cand"] >= 0]
    for s, c in zip(_take(found["s1"].to_numpy()), _take(found["cand"].to_numpy()), strict=True):
        tr[s].add(c)
    cands = pd.DataFrame({"s1_id": _take(union["s1"].to_numpy()), "cand_id": _take(union["cand"].to_numpy())})
    variants: dict[str, NDArray[np.bool_]] = {
        f"pass_{p.name}": union[f"pass_{p.name}_score"].notna().to_numpy() for p in PASSES
    }
    variants["union"] = np.ones(len(union), dtype=bool)
    for cap in REPORT_CAPS:
        variants[f"union@{cap}"] = (union["union_rank"] <= cap).to_numpy()
    level = evaluate.logger.level
    evaluate.logger.setLevel(logging.WARNING)  # its per-call INFO line would repeat per chunk x variant
    try:
        for name, mask in variants.items():
            rep = evaluate.blocking_recall_report(cands[mask], tr, s1_ids)
            report.add(country, name, rep, len(s1_ids))
    finally:
        evaluate.logger.setLevel(level)


def _capped_out_true(union: pd.DataFrame, truth: pd.DataFrame, cap: int, n_records: int) -> pd.DataFrame:
    """True pairs of one S1 chunk that are in the union but beyond the cap.

    Args:
        union: ``union_passes`` output for the chunk (int32 ``s1, cand``, ``union_rank``).
        truth: ``_truth_index`` output (int64 ``s1, cand``, sorted by s1).
        cap: Candidates kept per S1.
        n_records: Total records (for the int64 pair key).

    Returns:
        int64 ``s1, cand`` of the capped-out true pairs.
    """
    out = union.loc[union["union_rank"] > cap, ["s1", "cand"]].astype(np.int64)
    t = truth[truth["cand"] >= 0]  # singletons have cand -1
    true_key = t["s1"].to_numpy() * n_records + t["cand"].to_numpy()
    return out[np.isin(out["s1"].to_numpy() * n_records + out["cand"].to_numpy(), true_key)]


def select_s1_fraction(strata: pd.Series, fraction: float) -> NDArray[np.bool_]:
    """Deterministic stratified sample: keep ``round(fraction * n)`` rows of every stratum.

    Args:
        strata: Stratum label per S1 (e.g. "India|singleton"), in a fixed order.
        fraction: Share to keep, in (0, 1].

    Returns:
        Boolean keep mask aligned with ``strata``; strata are visited in sorted
        order with one ``config.SEED`` generator, so the result is reproducible.
    """
    keep = np.zeros(len(strata), dtype=bool)
    rng = np.random.default_rng(config.SEED)
    for _, idx in sorted(strata.groupby(strata, sort=True).indices.items()):
        keep[rng.permutation(idx)[: int(round(fraction * len(idx)))]] = True
    return keep


def _subset_s1(rec: pd.DataFrame, truth: pd.DataFrame | None, split: str, out: Path) -> NDArray[np.bool_] | None:
    """Apply ``config.BLOCK_S1_FRACTION``: pick the S1 to block, write their IDs to ``out``.

    Args:
        rec: ``_load_records`` output.
        truth: ``_truth_index`` output (needed for the singleton strata).
        split: Must be ``"train"`` when the fraction is below 1.
        out: ``blocking_s1_subset_{split}.parquet`` (``s1_id`` str): the S1
            that were blocked; decide.py restricts train S1 to it. Removed when
            the fraction is 1 so a stale subset never outlives a full run.

    Returns:
        Keep mask over ``rec`` rows (True for every non-S1 row), or None when
        the fraction is 1.

    Raises:
        ValueError: If the fraction is outside (0, 1], used on the test split,
            or there is no ground truth to stratify by.
    """
    f = config.BLOCK_S1_FRACTION
    if not 0.0 < f <= 1.0:
        raise ValueError(f"BLOCK_S1_FRACTION must be in (0, 1], got {f}")
    if f == 1.0:
        out.unlink(missing_ok=True)
        return None
    if split != "train":
        raise ValueError(f"BLOCK_S1_FRACTION={f} is train-only; the {split} split always blocks every S1")
    if truth is None:
        raise ValueError("BLOCK_S1_FRACTION < 1 needs the train ground truth for the singleton strata")
    is_s1 = (rec["source"] == "S1").to_numpy()
    s1_idx = np.flatnonzero(is_s1)
    matched = np.isin(s1_idx, truth.loc[truth["cand"] >= 0, "s1"].to_numpy())
    strata = rec["country"].to_numpy()[s1_idx] + np.where(matched, "|matched", "|singleton")
    keep_s1 = select_s1_fraction(pd.Series(strata), f)
    keep = ~is_s1
    keep[s1_idx[keep_s1]] = True
    pd.DataFrame({"s1_id": rec["entity_id"].to_numpy()[s1_idx[keep_s1]]}).to_parquet(out, index=False)
    for st, n in pd.Series(strata).value_counts().sort_index().items():
        logger.info("S1 fraction %.3f: stratum %s keeps %d of %d", f, st, int(keep_s1[strata == st].sum()), n)
    logger.warning("S1 fraction %.3f: blocking %d of %d train S1; the other %d get no candidates and are "
                   "excluded from training, decide and the recall report (%s)", f, int(keep_s1.sum()), len(s1_idx),
                   int((~keep_s1).sum()), out.name)
    return keep


def _write_part(union: pd.DataFrame, path: Path) -> None:
    """Spill capped union rows (index columns, pass columns) to a parquet part."""
    union.drop(columns="union_rank").to_parquet(path, index=False)


def _finalize(parts: list[Path], rec: pd.DataFrame, out: Path) -> int:
    """Add reverse features and string IDs; write ``out`` via .tmp + rename.

    Memory: s1/cand/best of all pairs (12 bytes per pair) plus the reverse
    feature arrays; parts are then streamed one at a time.

    Returns:
        Number of rows written.
    """
    cols = ["s1", "cand", "best_block_score"]
    allp = pa.concat_tables([pq.read_table(p, columns=cols) for p in parts]) if parts else None
    if allp is None or allp.num_rows == 0:
        s1 = np.empty(0, np.int32); cand = np.empty(0, np.int32); best = np.empty(0, np.float32)
    else:
        s1 = allp["s1"].to_numpy(); cand = allp["cand"].to_numpy(); best = allp["best_block_score"].to_numpy()
    del allp
    rev_n, rev_rank, rev_gap = reverse_features(s1, cand, best)
    del s1, cand, best
    gc.collect()

    ids = pa.array(rec["entity_id"], pa.string())
    source = pa.array(rec["source"], pa.string())
    country = rec["country"].astype("category").cat.codes.to_numpy()
    schema = _output_schema()
    tmp = out.with_suffix(".parquet.tmp")
    offset = 0
    with pq.ParquetWriter(tmp, schema) as writer:
        for p in parts:
            u = pq.read_table(p).to_pandas()
            n = len(u)
            s1_i = pa.array(u["s1"].to_numpy())
            c_i = pa.array(u["cand"].to_numpy())
            data: dict[str, object] = {
                "s1_id": ids.take(s1_i), "cand_id": ids.take(c_i), "cand_source": source.take(c_i),
                "country_match": pa.array(country[u["s1"].to_numpy()] == country[u["cand"].to_numpy()]),
            }
            for f in schema.names[4:]:
                if f in u:
                    data[f] = pa.array(u[f].to_numpy(), schema.field(f).type)
                elif f.startswith("pass_"):
                    data[f] = pa.array(np.full(n, np.nan, dtype=np.float32))
            data["rev_n_s1"] = pa.array(rev_n[offset: offset + n])
            data["rev_rank"] = pa.array(rev_rank[offset: offset + n])
            data["rev_gap"] = pa.array(rev_gap[offset: offset + n])
            writer.write_table(pa.table({f: data[f] for f in schema.names}, schema=schema))
            offset += n
    tmp.replace(out)
    return offset


def _log_report(report: _Report, path: Path, split: str) -> None:
    """Write the recall TSV, log it, and append a row to the experiment log."""
    df = report.frame().sort_values(["country", "variant"], ignore_index=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.4f", lineterminator="\n", encoding="utf-8")
    for r in df.itertuples(index=False):
        logger.info("Recall %-6s %-10s pair recall %.4f | S1 fully covered %6.2f%% | avg cands/S1 %5.1f",
                    r.country, r.variant, r.pair_recall, r.pct_s1_fully_covered, r.avg_cands_per_s1)
    final = df[(df["country"] == "ALL") & (df["variant"] == f"union@{config.MAX_CANDIDATES_PER_S1}")]
    if final.empty:
        final = df[(df["country"] == "ALL") & (df["variant"] == "union")]
    at = df.set_index(["country", "variant"])["pair_recall"]
    notes = "; ".join(f"{c} {v}={at[(c, v)]:.4f}" for c, v in at.index if c != "ALL" or v.startswith("pass_"))
    passes = ", ".join(f"{p.name}={'|'.join(p.columns)}/{p.analyzer}{p.ngram} K={pass_top_k(p)} max_df={p.max_df} "
                       f"min_terms={p.min_query_terms}" for p in PASSES)
    experiment_log.log_experiment(
        EXPERIMENT_ID, "block",
        f"blocking v1 on {split}: passes {passes}; cap {config.MAX_CANDIDATES_PER_S1} by RRF (k={RRF_K:g})",
        pair_recall=float(final["pair_recall"].iloc[0]) if not final.empty else None,
        avg_cands_per_s1=float(final["avg_cands_per_s1"].iloc[0]) if not final.empty else None,
        notes=notes,
    )
    logger.info("Wrote recall report %s", path)


def run_stage(paths: Paths, split: str) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet.

    Per country: each pass is fitted on the country's records and returns its
    top-K per S1 (see module docstring). Pairs are then unioned in S1 chunks of
    ``config.BLOCK_CHUNK_S1_ROWS``, capped at ``config.MAX_CANDIDATES_PER_S1``
    per S1 by ``union_rank`` (RRF) and spilled to parquet parts; reverse
    features are computed over all S1 at the end. On the train split a recall
    report (per pass, union, union at ``REPORT_CAPS``; per country) is built
    with ``evaluate.blocking_recall_report`` and logged to the experiment log.

    Memory: the records' ID/country/pass columns, one pass's TF-IDF matrix and
    its top-K pairs for one country, one bounded sparse product, and at the end
    12 bytes per output pair for the reverse features.

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.

    Raises:
        ValueError: If records columns are missing or not strings.
    """
    art = paths.artifacts_dir
    out = art / f"candidates_{split}.parquet"
    parts_dir = art / f"candidates_{split}.parts.tmp"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    rec = _load_records(art / f"records_{split}.parquet")
    ids = pa.array(rec["entity_id"], pa.string())
    truth = _truth_index(paths, split, rec["entity_id"])
    report = _Report()
    parts: list[Path] = []
    capped_out: list[pd.DataFrame] = []
    cap = config.MAX_CANDIDATES_PER_S1
    n_records = len(rec)
    keep = _subset_s1(rec, truth, split, io_utils.blocked_s1_path(art, split))
    if keep is not None and truth is not None:
        truth = truth[keep[truth["s1"].to_numpy()]].reset_index(drop=True)

    for country, grp in rec.groupby("country", sort=True):
        pos = grp.index.to_numpy()
        is_s1 = (grp["source"] == "S1").to_numpy()
        if keep is not None:
            is_s1 = is_s1 & keep[pos]  # unselected S1 are neither queries nor candidates
        s1_pos = pos[is_s1].astype(np.int32)
        cand_pos = pos[(grp["source"] != "S1").to_numpy()].astype(np.int32)
        logger.info("Country %s: %d S1, %d S2/S3", country, len(s1_pos), len(cand_pos))
        results = {p.name: run_pass(pass_text(grp, p), s1_pos, cand_pos, p, str(country)) for p in PASSES}
        gc.collect()
        for a in range(0, len(s1_pos), config.BLOCK_CHUNK_S1_ROWS):
            chunk_s1 = s1_pos[a: a + config.BLOCK_CHUNK_S1_ROWS]
            lo, hi = int(chunk_s1[0]), int(chunk_s1[-1])
            sub = {}
            for n, r in results.items():
                i, j = np.searchsorted(r.s1, lo), np.searchsorted(r.s1, hi, side="right")
                sub[n] = PassResult(r.s1[i:j], r.cand[i:j], r.score[i:j], r.rank[i:j])
            union = union_passes(sub, n_records)
            if truth is not None:
                _chunk_recall(report, str(country), union, chunk_s1.astype(np.int64), truth, ids)
                ts = truth["s1"].to_numpy()
                part_truth = truth.iloc[np.searchsorted(ts, lo): np.searchsorted(ts, hi, side="right")]
                capped_out.append(_capped_out_true(union, part_truth, cap, n_records))
            part = parts_dir / f"part-{len(parts):05d}.parquet"
            _write_part(union[union["union_rank"] <= cap], part)
            parts.append(part)
        del results
        gc.collect()

    n_rows = _finalize(parts, rec, out)
    shutil.rmtree(parts_dir)
    n_s1 = int((rec["source"] == "S1").sum()) if keep is None else int(keep.sum() - (rec["source"] != "S1").sum())
    logger.info("Wrote %s: %d pairs, %.1f per S1 (cap %d)", out, n_rows, n_rows / max(n_s1, 1), cap)
    if truth is not None:
        _log_report(report, art / f"blocking_recall_{split}.tsv", split)
        co = pd.concat(capped_out, ignore_index=True) if capped_out else pd.DataFrame({"s1": [], "cand": []})
        pq.write_table(pa.table({"s1_id": ids.take(pa.array(co["s1"].to_numpy(np.int64))),  # string even when empty
                                 "cand_id": ids.take(pa.array(co["cand"].to_numpy(np.int64)))}),
                       art / f"blocking_capped_out_{split}.parquet")
        logger.info("%d true pairs in the uncapped union were cut by the cap", len(co))
