"""Stage `feat`: pairwise, group-context and blocking features.

Reads:  <artifacts>/records_{split}.parquet, candidates_{split}.parquet
        (+ <data>/train/train_ground_truth.tsv for the train label)
Writes: <artifacts>/features_{split}/part-*.parquet: s1_id, cand_id (string),
        float32 feature columns (names fixed by the code and config.FEATURE_SET, same for every
        part), + int8 ``label`` on train. v2 = every v1 column (same values) plus ``v2_features``.
        One row per candidate pair, in candidates file order.

No country identity features: only ``country_match`` (agreement flag).

Memory: the needed records columns stay in Arrow (~240 bytes per record,
~3 GB for 12.5M records). Candidates are streamed in chunks of about
config.FEATURE_CHUNK_PAIRS pairs, cut at S1 boundaries so every S1 group is
whole within one chunk (group-context features need it). Parts go to
``features_{split}.tmp/`` and the folder is renamed when all parts exist.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray
from rapidfuzz import fuzz, utils
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src import config, contracts, decide, io_utils
from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "features.py (lane feat)"

RECORD_COLUMNS: tuple[str, ...] = (
    "entity_id", "name_norm", "name_core", "legal_suffix", "name_acronym", "name_translit", "name_key",
    "addr_norm", "addr_translit", "postal_tokens", "num_tokens", "landmark_flag", "name_empty", "addr_empty",
)
# Blocking columns copied as float32 features (country_match is the allowed agreement flag).
BLOCK_FEATURES: tuple[str, ...] = tuple(
    c for c in contracts.CANDIDATES_COLUMNS if c not in ("s1_id", "cand_id", "cand_source")
)
IDF_BATCH_RECORDS = 500_000  # records hashed per batch while counting document frequencies
HASH_FEATURES = 1 << 20

Scorer = Callable[..., float]
FUZZ_SCORERS: dict[str, tuple[Scorer, float]] = {  # name -> (scorer, divisor to reach 0..1)
    "ratio": (fuzz.ratio, 100.0),
    "token_set": (fuzz.token_set_ratio, 100.0),
    "token_sort": (fuzz.token_sort_ratio, 100.0),
    "partial": (fuzz.partial_ratio, 100.0),
    "jw": (JaroWinkler.normalized_similarity, 1.0),
    "lev": (Levenshtein.normalized_similarity, 1.0),
}
# Similarity columns ranked within each S1 for the group-context features.
GROUP_SOURCES: tuple[str, ...] = ("name_char3_cos", "name_core_token_set", "addr_tok_cos")

F32 = NDArray[np.float32]


# --------------------------------------------------------------------------- text vectors


@dataclass(frozen=True)
class HashedIdf:
    """A stateless hashing vectoriser plus IDF weights counted over one split's records.

    Attributes:
        vec: sklearn HashingVectorizer (no vocabulary, so no OOV at test time).
        idf: float32 smoothed IDF per hash bucket, ``log((1+n)/(1+df)) + 1``.
    """

    vec: HashingVectorizer
    idf: F32

    def binary(self, texts: list[str]) -> sp.csr_matrix:
        """0/1 term-presence matrix, one row per text."""
        x: sp.csr_matrix = self.vec.transform(texts)
        x.data[:] = 1.0
        return x

    def tfidf(self, texts: list[str]) -> sp.csr_matrix:
        """L2-normalised TF-IDF rows (all-zero row for an empty text)."""
        x: sp.csr_matrix = self.vec.transform(texts)
        x = x.multiply(self.idf).tocsr()
        out: sp.csr_matrix = l2_normalize(x, copy=False)
        return out


def _hashing(analyzer: str, ngram: int) -> HashingVectorizer:
    """HashingVectorizer with raw counts (no sign flip, no norm), float32."""
    return HashingVectorizer(
        n_features=HASH_FEATURES, analyzer=analyzer, ngram_range=(ngram, ngram), alternate_sign=False,
        norm=None, lowercase=False, token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )


def fit_idf(texts: pa.ChunkedArray | pa.Array, analyzer: str, ngram: int) -> HashedIdf:
    """Count document frequencies over every text of the split, IDF_BATCH_RECORDS at a time.

    Args:
        texts: One string per record (a records column).
        analyzer: ``"word"`` or ``"char_wb"``.
        ngram: n-gram length.

    Returns:
        ``HashedIdf`` for that column. Memory: one batch's sparse matrix plus a
        HASH_FEATURES-long df array.
    """
    vec = _hashing(analyzer, ngram)
    df = np.zeros(HASH_FEATURES, dtype=np.int64)
    for start in range(0, len(texts), IDF_BATCH_RECORDS):
        x = vec.transform(texts.slice(start, IDF_BATCH_RECORDS).to_pylist()).tocsr()
        x.sum_duplicates()
        df += np.bincount(x.indices, minlength=HASH_FEATURES)
    n = len(texts)
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    return HashedIdf(vec=vec, idf=idf)


def _row_dot(a: sp.csr_matrix, b: sp.csr_matrix) -> F32:
    """Row-wise dot product of two same-shape sparse matrices."""
    return np.asarray(a.multiply(b).sum(axis=1), dtype=np.float32).ravel()


def cosine(h: HashedIdf, texts: list[str], left: NDArray[np.intp], right: NDArray[np.intp]) -> F32:
    """TF-IDF cosine between ``texts[left[i]]`` and ``texts[right[i]]`` (0 if either is empty)."""
    x = h.tfidf(texts)
    return _row_dot(x[left], x[right])


def idf_overlap(h: HashedIdf, texts: list[str], left: NDArray[np.intp],
                right: NDArray[np.intp]) -> tuple[F32, F32]:
    """IDF-weighted Jaccard and plain token Jaccard per pair (NaN if both texts are empty).

    Returns:
        ``(weighted_jaccard, jaccard)``.
    """
    b = h.binary(texts)
    bl, br = b[left], b[right]
    inter = bl.multiply(br).tocsr()
    w_inter = inter @ h.idf
    w_union = bl @ h.idf + br @ h.idf - w_inter
    n_inter = np.asarray(inter.sum(axis=1)).ravel()
    n_union = np.diff(bl.indptr) + np.diff(br.indptr) - n_inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(w_union > 0, w_inter / w_union, np.nan).astype(np.float32),
                np.where(n_union > 0, n_inter / n_union, np.nan).astype(np.float32))


# --------------------------------------------------------------------------- pair features


def fuzz_scores(left: list[str], right: list[str], scorer: Scorer, divisor: float,
                processor: Callable[[str], str] | None = None) -> F32:
    """Pairwise rapidfuzz score in 0..1 (rapidfuzz.process.cpdist); NaN where either side is empty."""
    out = cpdist(left, right, scorer=scorer, processor=processor, dtype=np.float32,
                 workers=config.RAPIDFUZZ_WORKERS).astype(np.float32) / np.float32(divisor)
    empty = np.fromiter((not a or not b for a, b in zip(left, right, strict=True)), bool, len(left))
    out[empty] = np.nan
    return out


def set_counts(left: list[list[str]], right: list[list[str]]) -> tuple[NDArray[np.int32], ...]:
    """Per pair: distinct sizes of both token lists and of their intersection.

    Returns:
        ``(n_left, n_right, n_shared)`` int32 arrays. Python loop, ~1 µs per pair.
    """
    n = len(left)
    nl, nr, ns = np.empty(n, np.int32), np.empty(n, np.int32), np.empty(n, np.int32)
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        sa, sb = set(a), set(b)
        nl[i], nr[i], ns[i] = len(sa), len(sb), len(sa & sb)
    return nl, nr, ns


def _digits(text: str) -> list[str]:
    """Whitespace tokens of ``text`` that contain a digit."""
    return [t for t in text.split() if any(ch.isdigit() for ch in t)]


def _f(x: NDArray[np.generic]) -> F32:
    """Cast to float32."""
    return np.asarray(x, dtype=np.float32)


def pair_features(rec: pa.Table, left: NDArray[np.intp], right: NDArray[np.intp],
                  name_char: HashedIdf, name_word: HashedIdf, addr_word: HashedIdf) -> dict[str, F32]:
    """Name and address similarity features for pairs of records.

    Args:
        rec: Unique records of the chunk, columns ``RECORD_COLUMNS``.
        left: Row in ``rec`` of each pair's S1 record.
        right: Row in ``rec`` of each pair's candidate record.
        name_char, name_word, addr_word: IDF models of the split.

    Returns:
        Feature name -> float32 array, one value per pair (NaN = undefined).
    """
    col = {c: rec.column(c).to_pylist() for c in RECORD_COLUMNS if c != "entity_id"}

    # Any: Arrow to_pylist values are untyped (str, bool or list[str] per column).
    def lr(c: str) -> tuple[list[Any], list[Any]]:
        """Values of column ``c`` for the left and right side of every pair."""
        v = col[c]
        return [v[i] for i in left], [v[i] for i in right]

    out: dict[str, F32] = {}
    for c in ("name_core", "name_norm"):
        a, b = lr(c)
        for name, (scorer, div) in FUZZ_SCORERS.items():
            out[f"{c}_{name}"] = fuzz_scores(a, b, scorer, div)
    out["name_key_ratio"] = fuzz_scores(*lr("name_key"), fuzz.ratio, 100.0)
    out["name_translit_ratio"] = fuzz_scores(*lr("name_translit"), fuzz.ratio, 100.0, utils.default_process)
    out["name_translit_token_set"] = fuzz_scores(*lr("name_translit"), fuzz.token_set_ratio, 100.0,
                                                 utils.default_process)
    out["name_char3_cos"] = cosine(name_char, col["name_norm"], left, right)
    out["name_tok_idf_jacc"], out["name_tok_jacc"] = idf_overlap(name_word, col["name_norm"], left, right)

    # Acronym: one side's acronym spells the other's name_core (and is not its own name).
    compact = np.array([c.replace(" ", "") for c in col["name_core"]], dtype=object)
    acronym = np.array(col["name_acronym"], dtype=object)
    usable = np.array([len(a) >= 2 for a in col["name_acronym"]], dtype=bool) & (acronym != compact)
    out["acr_s1_is_cand"] = _f(usable[left] & (acronym[left] == compact[right]))
    out["acr_cand_is_s1"] = _f(usable[right] & (acronym[right] == compact[left]))

    sfx = np.array(col["legal_suffix"], dtype=object)
    has = sfx != ""
    out["sfx_equal"] = _f(has[left] & has[right] & (sfx[left] == sfx[right]))
    out["sfx_conflict"] = _f(has[left] & has[right] & (sfx[left] != sfx[right]))
    out["sfx_missing"] = _f(~(has[left] & has[right]))

    length = np.array([len(s) for s in col["name_norm"]], dtype=np.float32)
    lo, hi = np.minimum(length[left], length[right]), np.maximum(length[left], length[right])
    out["name_len_ratio"] = np.where(hi > 0, lo / np.maximum(hi, 1), np.nan).astype(np.float32)
    first = np.array([s.split(" ", 1)[0] for s in col["name_core"]], dtype=object)
    out["first_tok_match"] = _f((first[left] == first[right]) & (first[left] != ""))

    digits = [_digits(s) for s in col["name_norm"]]
    nl, nr, ns = set_counts([digits[i] for i in left], [digits[i] for i in right])
    both = (nl > 0) & (nr > 0)
    out["name_digit_jacc"] = np.where(both, ns / np.maximum(nl + nr - ns, 1), np.nan).astype(np.float32)
    out["name_digit_one_side"] = _f((nl > 0) != (nr > 0))

    # Address.
    nl, nr, ns = set_counts(*lr("postal_tokens"))
    out["postal_match"] = _f(ns > 0)
    out["postal_conflict"] = _f((nl > 0) & (nr > 0) & (ns == 0))
    out["postal_missing"] = _f((nl == 0) | (nr == 0))
    nl, nr, ns = set_counts(*lr("num_tokens"))
    out["num_overlap"] = np.where((nl > 0) & (nr > 0), ns / np.maximum(np.minimum(nl, nr), 1),
                                  np.nan).astype(np.float32)
    out["num_conflict"] = _f((nl > 0) & (nr > 0) & (ns == 0))
    out["addr_tok_cos"] = cosine(addr_word, col["addr_norm"], left, right)
    out["addr_tok_idf_jacc"], out["addr_tok_jacc"] = idf_overlap(addr_word, col["addr_norm"], left, right)
    out["addr_norm_token_set"] = fuzz_scores(*lr("addr_norm"), fuzz.token_set_ratio, 100.0)
    out["addr_translit_token_set"] = fuzz_scores(*lr("addr_translit"), fuzz.token_set_ratio, 100.0,
                                                 utils.default_process)
    for c in ("landmark_flag", "name_empty", "addr_empty"):
        flag = np.array(col[c], dtype=bool)
        out[f"s1_{c}"], out[f"cand_{c}"] = _f(flag[left]), _f(flag[right])
    return out


def group_features(s1: NDArray[np.int32], feats: dict[str, F32]) -> dict[str, F32]:
    """Context of each pair within its S1's candidate list.

    Args:
        s1: S1 code per pair; every S1's pairs are complete within this chunk.
        feats: Must contain every column in ``GROUP_SOURCES``.

    Returns:
        ``grp_n_cands`` and, per source column, ``_rank`` (1 = best, ties get
        the lowest rank), ``_gap`` (S1 best minus this) and ``_z`` (z-score in
        the S1; 0 when the S1's values are all equal). NaN inputs count as 0.
    """
    g = pd.Series(s1)
    out: dict[str, F32] = {"grp_n_cands": _f(g.map(g.value_counts()).to_numpy())}
    for c in GROUP_SOURCES:
        x = pd.Series(np.nan_to_num(feats[c], nan=0.0), dtype=np.float64)
        grp = x.groupby(s1)
        std = grp.transform("std", ddof=0).to_numpy()
        mean = grp.transform("mean").to_numpy()
        out[f"grp_{c}_rank"] = _f(grp.rank(ascending=False, method="min").to_numpy())
        out[f"grp_{c}_gap"] = _f(grp.transform("max").to_numpy() - x.to_numpy())
        out[f"grp_{c}_z"] = _f(np.where(std > 1e-9, (x.to_numpy() - mean) / np.where(std > 1e-9, std, 1.0), 0.0))
    return out


# --------------------------------------------------------------------------- v2 features (config.FEATURE_SET)

FEATURE_SETS: tuple[str, ...] = ("v1", "v2")


def _addr_parts(addr: str) -> tuple[frozenset[str], frozenset[str], str]:
    """Unit tokens, pure numbers and house number of one ``addr_norm``.

    A unit is a token mixing letters and digits (``a407``) or a single letter
    followed by a short number, which normalisation splits (``a-402`` ->
    ``a 402`` -> ``a402``). The house number is the first pure number under 5
    digits (5-6 digit numbers are postal-like).

    Returns:
        ``(units, numbers, house)``; ``house`` is "" when there is none.
    """
    toks = addr.split()
    units: set[str] = set()
    nums: set[str] = set()
    i = 0
    while i < len(toks):
        t = toks[i]
        if len(t) == 1 and t.isalpha() and i + 1 < len(toks) and toks[i + 1].isdigit() and len(toks[i + 1]) < 5:
            units.add(t + toks[i + 1])  # the number is part of the unit, not a bare number
            i += 1
        elif t.isdigit():
            nums.add(t)
        elif t.isalnum() and any(c.isdigit() for c in t):
            units.add(t)
        i += 1
    house = min((t for t in nums if len(t) < 5), key=toks.index, default="")
    return frozenset(units), frozenset(nums), house


def _name_no_suffix(translit: str, norm: str, core: str) -> str:
    """``name_translit`` (rapidfuzz default_process) minus the legal-suffix tokens.

    Suffix tokens = tokens of ``name_norm`` not in ``name_core`` (e.g. ``pvt ltd``);
    a non-Latin translit shares none, so it is kept whole.
    """
    sfx = set(norm.split()) - set(core.split())
    return " ".join(t for t in utils.default_process(translit).split() if t not in sfx)


def v2_features(rec: pa.Table, left: NDArray[np.intp], right: NDArray[np.intp], s1: NDArray[np.int32],
                feats: dict[str, F32]) -> dict[str, F32]:
    """Extra v2 features: unit/number conflicts, exact-address, suffix-robust names, same-address context.

    Args:
        rec: Unique records of the chunk, columns ``RECORD_COLUMNS``.
        left: Row in ``rec`` of each pair's S1 record.
        right: Row in ``rec`` of each pair's candidate record.
        s1: S1 code per pair; every S1's pairs are complete within this chunk.
        feats: v1 features of the chunk; must contain ``name_char3_cos``.

    Returns:
        Feature name -> float32 array, one value per pair (NaN = undefined).
        Python loops, a few µs per pair; memory linear in the chunk.
    """
    addr = rec.column("addr_norm").to_pylist()
    parts = [_addr_parts(a) for a in addr]
    postal = [frozenset(p) for p in rec.column("postal_tokens").to_pylist()]
    core_toks = [frozenset(c.split()) for c in rec.column("name_core").to_pylist()]
    addr_toks = [frozenset(a.split()) for a in addr]
    nosfx = [_name_no_suffix(t, n, c) for t, n, c in zip(rec.column("name_translit").to_pylist(),
                                                         rec.column("name_norm").to_pylist(),
                                                         rec.column("name_core").to_pylist(), strict=True)]
    names = ("unit_equal", "unit_conflict", "unit_missing_one", "num_one_side_n", "num_disjoint",
             "addr_exact", "addr_tokset_equal", "house_postal_equal", "name_core_tokset_equal",
             "name_core_contained")
    v = np.zeros((len(names), len(left)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right, strict=True)):  # ponytail: Python loop, ~3 µs/pair
        (ua, na, ha), (ub, nb, hb) = parts[a], parts[b]
        ca, cb = core_toks[a], core_toks[b]
        v[0, i] = bool(ua & ub)
        v[1, i] = bool(ua) and bool(ub) and not ua & ub
        v[2, i] = bool(ua) != bool(ub)
        v[3, i] = len(na ^ nb)
        v[4, i] = bool(na) and bool(nb) and not na & nb
        v[5, i] = bool(addr[a]) and addr[a] == addr[b]
        v[6, i] = bool(addr_toks[a]) and addr_toks[a] == addr_toks[b]
        v[7, i] = bool(ha) and ha == hb and bool(postal[a] & postal[b])
        v[8, i] = bool(ca) and ca == cb
        v[9, i] = bool(ca) and bool(cb) and (ca <= cb or cb <= ca)
    out = {n: v[k] for k, n in enumerate(names)}
    out["name_nosfx_token_set"] = fuzz_scores([nosfx[i] for i in left], [nosfx[i] for i in right],
                                              fuzz.token_set_ratio, 100.0)

    # Same-address context: this S1's candidates sharing this candidate's addr_norm.
    cand_addr = pd.Series([addr[i] for i in right])
    has = (cand_addr != "").to_numpy()
    df = pd.DataFrame({"s1": s1, "addr": cand_addr, "sim": np.nan_to_num(feats["name_char3_cos"], nan=0.0)})[has]
    grp = df.groupby(["s1", "addr"], sort=False)["sim"]
    n_same = np.zeros(len(left), dtype=np.float32)
    rank = np.full(len(left), np.nan, dtype=np.float32)
    n_same[has] = grp.transform("size").to_numpy()
    rank[has] = grp.rank(ascending=False, method="min").to_numpy()
    out["grp_same_addr_n"], out["grp_same_addr_name_rank"] = n_same, rank
    return out


# --------------------------------------------------------------------------- chunking


def s1_chunks(path: Path, columns: list[str], chunk_pairs: int) -> Iterator[pa.Table]:
    """Stream candidates in ~``chunk_pairs`` pairs, cutting only between S1 groups.

    Assumes each S1's pairs are contiguous in the file (blocking writes them so);
    ``run_stage`` verifies it.

    Args:
        path: candidates parquet.
        columns: Columns to read.
        chunk_pairs: Target pairs per chunk (a chunk grows to hold a whole S1).

    Yields:
        Arrow tables; the trailing S1 group of a batch is carried to the next.
    """
    carry: pa.Table | None = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_pairs, columns=columns):
        t = pa.Table.from_batches([batch])
        t = t if carry is None else pa.concat_tables([carry, t])
        s1 = t.column("s1_id")
        other = np.flatnonzero(~pc.equal(s1, s1[-1]).to_numpy(zero_copy_only=False))
        cut = int(other[-1]) + 1 if len(other) else 0
        if cut:
            yield t.slice(0, cut)
        carry = t.slice(cut)
    if carry is not None and carry.num_rows:
        yield carry


# --------------------------------------------------------------------------- stage


def _truth_keys(data_dir: Path, keys: NDArray[np.int64], order: NDArray[np.int64], n_rec: int) -> NDArray[np.int64]:
    """Sorted int64 keys ``s1_row * n_rec + cand_row`` of every true train pair."""
    gt = io_utils.read_ground_truth_pairs(data_dir / "train" / "train_ground_truth.tsv")
    gt = gt[gt["cand_id"] != ""]
    s1 = decide.encode(pa.array(gt["s1_id"]), keys, order, "ground-truth s1_id").astype(np.int64)
    cand = decide.encode(pa.array(gt["cand_id"]), keys, order, "ground-truth cand_id").astype(np.int64)
    return np.sort(s1 * n_rec + cand)


def run_stage(paths: Paths, split: str) -> None:
    """Build features_{split}/part-*.parquet, one part per candidate chunk.

    Args:
        paths: Resolved run directories.
        split: ``"train"`` (adds ``label``) or ``"test"``.

    Raises:
        ValueError: If an input column is missing, an ID is unknown, or a
            S1's candidates are not contiguous in candidates_{split}.parquet,
            or config.FEATURE_SET is not in FEATURE_SETS.
    """
    if config.FEATURE_SET not in FEATURE_SETS:
        raise ValueError(f"config.FEATURE_SET={config.FEATURE_SET!r}; expected one of {FEATURE_SETS}")
    logger.info("Feature set %s", config.FEATURE_SET)
    art = paths.artifacts_dir
    rec_path, cand_path = art / f"records_{split}.parquet", art / f"candidates_{split}.parquet"
    io_utils.require_columns(pq.ParquetFile(rec_path).schema_arrow.names, RECORD_COLUMNS, str(rec_path))
    cand_cols = ["s1_id", "cand_id", "cand_source", *BLOCK_FEATURES]
    io_utils.require_columns(pq.ParquetFile(cand_path).schema_arrow.names, cand_cols, str(cand_path))

    t0 = time.perf_counter()
    rec = pq.read_table(rec_path, columns=list(RECORD_COLUMNS))
    n_rec = rec.num_rows
    keys = decide.id_keys(rec.column("entity_id"))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    logger.info("Records %s: %d rows, %.2f GB in Arrow", split, n_rec, rec.nbytes / 1024**3)
    name_char = fit_idf(rec.column("name_norm"), "char_wb", 3)
    name_word = fit_idf(rec.column("name_norm"), "word", 1)
    addr_word = fit_idf(rec.column("addr_norm"), "word", 1)
    logger.info("IDF fitted on %d records in %.1fs", n_rec, time.perf_counter() - t0)
    truth = _truth_keys(paths.data_dir, keys, order, n_rec) if split == "train" else None

    names: list[str] = []
    final, tmp = art / f"features_{split}", art / f"features_{split}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    seen = np.zeros(n_rec, dtype=bool)
    n_pairs = n_pos = 0
    for part, chunk in enumerate(s1_chunks(cand_path, cand_cols, config.FEATURE_CHUNK_PAIRS)):
        s1 = decide.encode(chunk.column("s1_id"), keys, order, "candidates s1_id")
        cand = decide.encode(chunk.column("cand_id"), keys, order, "candidates cand_id")
        run_s1 = s1[np.r_[True, s1[1:] != s1[:-1]]]
        if seen[run_s1].any() or len(np.unique(run_s1)) != len(run_s1):
            raise ValueError(f"{cand_path}: candidates of one s1_id are not contiguous")
        seen[run_s1] = True

        uniq, inv = np.unique(np.concatenate([s1, cand]), return_inverse=True)
        left, right = inv[: len(s1)].astype(np.intp), inv[len(s1):].astype(np.intp)
        sub = rec.take(pa.array(uniq))
        feats = pair_features(sub, left, right, name_char, name_word, addr_word)
        feats |= group_features(s1, feats)
        if config.FEATURE_SET == "v2":
            feats |= v2_features(sub, left, right, s1, feats)
        for c in BLOCK_FEATURES:
            feats[c] = chunk.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        feats["cand_is_s3"] = _f(pc.equal(chunk.column("cand_source"), "S3").to_numpy(zero_copy_only=False))

        names = names or list(feats)
        table = {"s1_id": chunk.column("s1_id"), "cand_id": chunk.column("cand_id")}
        table |= {c: pa.array(feats[c], pa.float32()) for c in names}
        if truth is not None:
            pair = s1.astype(np.int64) * n_rec + cand
            pos = np.minimum(np.searchsorted(truth, pair), max(len(truth) - 1, 0))
            label = (truth[pos] == pair) if len(truth) else np.zeros(len(pair), bool)
            table["label"] = pa.array(label.astype(np.int8))
            n_pos += int(label.sum())
        pq.write_table(pa.table(table), tmp / f"part-{part:05d}.parquet")
        n_pairs += chunk.num_rows
        logger.info("Part %d: %d pairs (total %d, %.0f pairs/s)", part, chunk.num_rows, n_pairs,
                    n_pairs / (time.perf_counter() - t0))
    if truth is not None:
        logger.info("Labels: %d positive of %d pairs; %d of %d true pairs covered", n_pos, n_pairs, n_pos, len(truth))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    logger.info("Wrote %s: %d pairs x %d features in %.1fs", final, n_pairs, len(names), time.perf_counter() - t0)
