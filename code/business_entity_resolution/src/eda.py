"""Exploratory data analysis: plain-text report answering E1-E13 + IMPLICATIONS.

Usage (from code/business_entity_resolution/):
    python -m src.eda --data-dir PATH --artifacts-dir PATH

Writes ``<artifacts-dir>/eda/eda_report.txt`` and logs every report line at
INFO. If ``<data-dir>/test/`` is missing (e.g. the dev sample), test-split
sections are skipped with a WARNING.

Memory (full data target < 6 GB): one split's records are held at a time
(~1.5 GB for train); per-record work runs in chunks of SCAN_CHUNK_ROWS; ground
truth is a long Arrow-string table; name TF-IDF is fitted by streaming hashed
char 3-gram counts (identical cosines to TfidfVectorizer up to rare hash
collisions); E9 ranks score one country's S2/S3 pool against RANK_BATCH
sampled S1 at a time as sparse products. Pairwise similarities are only ever
computed for sampled pairs.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src import config, io_utils, normalize
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

SCAN_CHUNK_ROWS = 1_000_000
HASH_FEATURES = 2**22
N_QUANTILE_PAIRS = 20_000
N_RANK_S1 = 5_000
RANK_BATCH = 16
RECALL_KS: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
TOP_TOKENS = 100
TOP_CHARS = 20
N_PAIR_EXAMPLES = 30
N_ADDR_EXAMPLES = 20

_EDGE_L = r"(?:^|[^0-9A-Za-z])"
_EDGE_R = r"(?:[^0-9A-Za-z]|$)"
RE_SPACED6 = _EDGE_L + r"[0-9]{3} [0-9]{3}" + _EDGE_R  # (b) "600 001"
RE_ZIP4 = _EDGE_L + r"[0-9]{5}-[0-9]{4}" + _EDGE_R  # (c) "12345-6789"
RE_SPACED6_TOKENS = r"(?<![0-9A-Za-z])([0-9]{3}) ([0-9]{3})(?![0-9A-Za-z])"
RE_NON_ASCII = r"[^\x00-\x7f]"

SECTIONS: dict[str, str] = {
    "E1": "Rows per source per country; test/train ratio; test-only countries",
    "E2": "Singletons and matches per S1",
    "E3": "One-owner check",
    "E4": "Train S2/S3 records in no ground-truth list (distractors)",
    "E5": "Country agreement of true pairs",
    "E6": "Empty name / address rates",
    "E7": "Postal-like tokens",
    "E8": "Exact name match of true pairs (lowercase, letters+digits only)",
    "E9": "Char 3-gram TF-IDF name cosine",
    "E10": "Top name and address tokens per country",
    "E11": "Random true pairs side by side",
    "E12": "Non-ASCII characters",
    "E13": "Ground-truth integrity",
    "IMPLICATIONS": "Implications (from the numbers above)",
}


# --------------------------------------------------------------------------- helpers


def bucket_counts(counts: pd.Series) -> dict[str, float]:
    """Percentage of values equal to 0, 1, 2, 3, 4 and >= 5.

    Args:
        counts: Non-negative integer counts (one per S1).

    Returns:
        ``{"0": pct, ..., "4": pct, "5+": pct}``.
    """
    n = max(len(counts), 1)
    out = {str(k): 100.0 * float((counts == k).sum()) / n for k in range(5)}
    out["5+"] = 100.0 * float((counts >= 5).sum()) / n
    return out


def postal_info(addr_raw: pd.Series, addr_norm: pd.Series | None = None) -> pd.DataFrame:
    """Postal-like patterns in addresses.

    Args:
        addr_raw: Raw address strings.
        addr_norm: ``normalize.normalize_text(addr_raw)`` if already computed.

    Returns:
        Same index; bool ``has_a`` (standalone 5-6 digit token, prep v0 rule),
        ``has_b`` ("600 001" style), ``has_c`` (ZIP+4), and str ``tokens``:
        sorted space-joined union of (a) tokens and (b) tokens with the space
        removed.
    """
    an = normalize.normalize_text(addr_raw) if addr_norm is None else addr_norm
    a_tok = normalize.postal_tokens(an).tolist()
    b_tok = cast(list[list[tuple[str, str]]], addr_raw.str.findall(RE_SPACED6_TOKENS).tolist())
    tokens = [" ".join(sorted(set(a) | {p + q for p, q in b})) for a, b in zip(a_tok, b_tok, strict=True)]
    return pd.DataFrame(
        {
            "has_a": np.fromiter((len(a) > 0 for a in a_tok), dtype=bool, count=len(a_tok)),
            "has_b": addr_raw.str.contains(RE_SPACED6, regex=True).to_numpy(dtype=bool),
            "has_c": addr_raw.str.contains(RE_ZIP4, regex=True).to_numpy(dtype=bool),
            "tokens": pd.array(tokens, dtype="str"),
        },
        index=addr_raw.index,
    )


def name_key(names: pd.Series) -> pd.Series:
    """Lowercase and keep only letters, combining marks and digits (any script)."""
    return names.str.lower().str.replace(r"[^\p{L}\p{M}\p{N}]", "", regex=True)


class NameVectorizer:
    """Char 3-gram TF-IDF (smooth idf, l2 norm) fitted by streaming hashed counts.

    Equivalent to ``TfidfVectorizer(analyzer="char", ngram_range=(3, 3),
    lowercase=False)`` except for rare hash collisions in HASH_FEATURES buckets.
    ``fit`` may be called repeatedly on chunks; memory is one chunk's sparse
    counts plus an int64 document-frequency vector.
    """

    def __init__(self, n_features: int = HASH_FEATURES) -> None:
        """Create an unfitted vectorizer with ``n_features`` hash buckets."""
        self._hv = HashingVectorizer(
            analyzer="char", ngram_range=(3, 3), lowercase=False, n_features=n_features,
            alternate_sign=False, norm=None, dtype=np.float32,
        )
        self._df = np.zeros(n_features, dtype=np.int64)
        self._n_docs = 0
        self._idf: NDArray[np.float32] | None = None

    def _counts(self, names: pd.Series) -> sp.csr_matrix:
        """Hashed char 3-gram counts, one row per name."""
        x: sp.csr_matrix = self._hv.transform(names.tolist()).tocsr()
        x.sum_duplicates()
        return x

    def fit(self, names: pd.Series) -> None:
        """Add a chunk of documents to the document frequencies."""
        x = self._counts(names)
        self._df += np.bincount(x.indices, minlength=self._df.size)
        self._n_docs += x.shape[0]
        self._idf = None

    def transform(self, names: pd.Series) -> sp.csr_matrix:
        """L2-normalised TF-IDF rows (float32 CSR), one per name."""
        if self._idf is None:
            self._idf = (np.log((1.0 + self._n_docs) / (1.0 + self._df)) + 1.0).astype(np.float32)
        x = self._counts(names)
        x.data *= self._idf[x.indices]
        out: sp.csr_matrix = l2_normalize(x, copy=False)
        return out


def true_match_ranks(q: sp.csr_matrix, xt: sp.csr_matrix, true_cols: Sequence[NDArray[np.int64]]) -> NDArray[np.float64]:
    """Rank of each true match among all pool records by cosine.

    Rank = 1 + number of pool records with a strictly higher score. A true
    match with cosine 0 (no shared 3-gram) or column -1 (not in the pool, e.g.
    another country) gets rank inf.

    Args:
        q: L2-normalised query rows (one per S1), shape (b, V).
        xt: Transposed L2-normalised pool matrix, shape (V, n_pool), CSR.
        true_cols: For each query row, pool column indices of its true matches.

    Returns:
        Ranks, concatenated in query order then match order.
    """
    r = (q @ xt).tocsr()
    r.sort_indices()
    out: list[float] = []
    for i, cols in enumerate(true_cols):
        idx = r.indices[r.indptr[i] : r.indptr[i + 1]]
        val = r.data[r.indptr[i] : r.indptr[i + 1]]
        scores = np.zeros(len(cols), dtype=np.float32)
        if len(idx):
            pos = np.minimum(np.searchsorted(idx, cols), len(idx) - 1)
            hit = (np.asarray(cols) >= 0) & (idx[pos] == cols)
            scores[hit] = val[pos[hit]]
        ahead = (val[:, None] > scores[None, :]).sum(axis=0) if len(cols) else np.array([])
        out.extend(float(1 + a) if s > 0 else np.inf for a, s in zip(ahead, scores, strict=True))
    return np.asarray(out, dtype=np.float64)


def recall_at_k(ranks: NDArray[np.float64], ks: Sequence[int]) -> dict[int, float]:
    """Share of ranks <= K for each K."""
    return {k: float((ranks <= k).mean()) if len(ranks) else 0.0 for k in ks}


def _pct(x: float) -> str:
    """Format a fraction as a percentage."""
    return f"{100.0 * x:.2f}%"


def _lines(df: pd.DataFrame | pd.Series) -> list[str]:
    """Render a table as text lines."""
    return df.to_string().splitlines()


def _top(counter: Counter[str], n: int) -> list[str]:
    """Top-n tokens as lines of five ``token (count)`` entries."""
    items = [f"{t} ({c:,})" for t, c in counter.most_common(n)]
    return ["    " + " | ".join(items[i : i + 5]) for i in range(0, len(items), 5)]


# --------------------------------------------------------------------------- report


@dataclass
class Report:
    """Report lines collected per section, rendered in SECTIONS order."""

    sections: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, sec: str, *lines: str) -> None:
        """Append lines to a section."""
        self.sections[sec].extend(lines)

    def text(self) -> str:
        """Render the whole report."""
        out: list[str] = []
        for sec, title in SECTIONS.items():
            out += [f"=== {sec}  {title}", *self.sections.get(sec, ["(not computed)"]), ""]
        return "\n".join(out) + "\n"


@dataclass
class ScanResult:
    """Per-country token and character counters from one split."""

    name_tokens: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    addr_tokens: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    non_ascii_chars: Counter[str] = field(default_factory=Counter)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load S1, S2 and S3 of a split into one frame.

    Args:
        data_dir: Folder containing ``{split}/{split}_source{1,2,3}.tsv``.
        split: ``"train"`` or ``"test"``.

    Returns:
        One row per record: str ``entity_id, name, addr``; categorical
        ``source`` (S1/S2/S3) and ``country``.
    """
    frames = []
    for s in (1, 2, 3):
        df = io_utils.read_source(data_dir / split / f"{split}_source{s}.tsv")
        frames.append(pd.DataFrame({
            "entity_id": df["entity_id"], "source": f"S{s}", "country": df["country"],
            "name": df["business_name"], "addr": df["business_address"],
        }))
    rec = pd.concat(frames, ignore_index=True)
    rec["source"] = rec["source"].astype("category")
    rec["country"] = rec["country"].astype("category")
    return rec


def scan_records(rec: pd.DataFrame) -> ScanResult:
    """Add normalised-name, postal and non-ASCII columns; count tokens and chars.

    Processes SCAN_CHUNK_ROWS rows at a time. Adds columns ``name_norm`` (str),
    ``has_a``, ``has_b``, ``has_c`` (bool), ``post_tokens`` (str) and
    ``non_ascii`` (bool) to ``rec`` in place.

    Args:
        rec: Frame from ``load_split``.

    Returns:
        Token counters per country (normalised names / addresses) and a
        counter of non-ASCII characters in raw names + addresses.
    """
    res = ScanResult()
    parts: list[pd.DataFrame] = []
    for start in range(0, len(rec), SCAN_CHUNK_ROWS):
        ch = rec.iloc[start : start + SCAN_CHUNK_ROWS]
        nn = normalize.normalize_text(ch["name"])
        an = normalize.normalize_text(ch["addr"])
        info = postal_info(ch["addr"], an)
        non_ascii = ch["name"].str.contains(RE_NON_ASCII, regex=True) | ch["addr"].str.contains(RE_NON_ASCII, regex=True)
        country = ch["country"].to_numpy()
        for c in pd.unique(country):
            m = country == c
            res.name_tokens[str(c)].update(" ".join(nn[m].tolist()).split())
            res.addr_tokens[str(c)].update(" ".join(an[m].tolist()).split())
        raw = (ch["name"] + ch["addr"])[non_ascii.to_numpy(dtype=bool)]
        res.non_ascii_chars.update("".join(raw.str.replace(r"[\x00-\x7f]", "", regex=True).tolist()))
        parts.append(pd.DataFrame({
            "name_norm": nn, "has_a": info["has_a"], "has_b": info["has_b"], "has_c": info["has_c"],
            "post_tokens": info["tokens"], "non_ascii": non_ascii.to_numpy(dtype=bool),
        }, index=ch.index))
    scanned = pd.concat(parts)
    for col in scanned.columns:
        rec[col] = scanned[col]
    return res


def _rates(rec: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Percentage of True per source x country for bool columns."""
    return (100 * rec.groupby(["source", "country"], observed=True)[list(cols)].mean()).round(2)


def _empty_rates(rec: pd.DataFrame) -> pd.DataFrame:
    """Empty (whitespace-only) name/address rate per source x country, in %."""
    flags = pd.DataFrame({
        "source": rec["source"], "country": rec["country"],
        "name_empty": rec["name"].str.strip().eq(""), "addr_empty": rec["addr"].str.strip().eq(""),
    })
    return _rates(flags, ["name_empty", "addr_empty"])


def _describe(rec: pd.DataFrame, pos: int) -> str:
    """One record as ``id [country] name | address``."""
    r = rec.iloc[pos]
    return f"{r['entity_id']} [{r['country']}] {r['name']} | {r['addr']}"


# --------------------------------------------------------------------------- train


@dataclass
class TrainFacts:
    """Numbers from the train split needed by later sections."""

    counts: pd.DataFrame
    one_owner_shared: float = 0.0
    country_same: float = 1.0
    recall: dict[int, float] = field(default_factory=dict)
    postal_extra: dict[str, float] = field(default_factory=dict)
    postal_pair_share: float = 0.0


def analyse_train(data_dir: Path, rep: Report, n_quantile_pairs: int, n_rank_s1: int,
                  rng: np.random.Generator) -> tuple[TrainFacts, ScanResult]:
    """Compute every train-split section (E1-E13 parts that need train).

    Args:
        data_dir: Folder with ``train/``.
        rep: Report to append to.
        n_quantile_pairs: Sampled true / random pairs for E9 quantiles.
        n_rank_s1: Sampled S1 (with matches) for E9 recall at K.
        rng: Random generator (fixed seed).

    Returns:
        Facts for IMPLICATIONS and the train token/char counters.
    """
    with track_stage("eda-train-load"):
        rec = load_split(data_dir, "train")
        pairs = io_utils.read_ground_truth_pairs(data_dir / "train" / "train_ground_truth.tsv")
    with track_stage("eda-train-scan"):
        scan = scan_records(rec)

    counts = rec.groupby(["source", "country"], observed=True).size().unstack(fill_value=0)
    facts = TrainFacts(counts=counts)
    rep.add("E1", "Train rows per source x country:", *_lines(counts))

    idx = pd.Index(rec["entity_id"])
    cc = rec["country"].cat.codes.to_numpy().astype(np.int32)
    cats = rec["country"].cat.categories
    is_s1 = (rec["source"] == "S1").to_numpy()

    # E13 integrity + pair positions
    tp = pairs[pairs["cand_id"] != ""].reset_index(drop=True)
    s1_file = set(rec.loc[is_s1, "entity_id"].tolist())
    s1_gt = set(pairs["s1_id"].unique().tolist())
    p1 = idx.get_indexer(tp["s1_id"])
    p2 = idx.get_indexer(tp["cand_id"])
    in_list_s1 = int(tp["cand_id"].str.startswith("S1-").sum())
    dup_in_list = int(tp.duplicated().sum())
    rep.add("E13",
            f"Ground-truth rows: {len(s1_gt):,} S1 (duplicate S1 rows: 0; the reader rejects duplicates)",
            f"S1 in source file but not in ground truth: {len(s1_file - s1_gt):,}",
            f"S1 in ground truth but not in source file: {len(s1_gt - s1_file):,}",
            f"True pairs: {len(tp):,}; matched IDs missing from train S2/S3 files: {int((p2 < 0).sum()):,}",
            f"S1 IDs inside matched lists: {in_list_s1:,}; duplicate IDs within a list: {dup_in_list:,}")
    ok = (p1 >= 0) & (p2 >= 0) & ~tp["cand_id"].str.startswith("S1-").to_numpy(dtype=bool)
    p1v, p2v = p1[ok], p2[ok]
    tpv = tp[ok].reset_index(drop=True)

    # E2
    src = tpv["cand_id"].str[:2]
    per = tpv.groupby([tpv["s1_id"], src]).size().unstack(fill_value=0)
    per = per.reindex(index=rec.loc[is_s1, "entity_id"], columns=["S2", "S3"], fill_value=0)
    total = per["S2"] + per["S3"]
    buckets = pd.DataFrame({k: bucket_counts(v) for k, v in (("S2", per["S2"]), ("S3", per["S3"]), ("S2+S3", total))}).T.round(2)
    rep.add("E2", f"Singleton S1 (no match): {_pct(float((total == 0).mean()))} of {len(total):,}",
            f"Mean matches per S1: S2 {per['S2'].mean():.2f}, S3 {per['S3'].mean():.2f}, total {total.mean():.2f}",
            "% of S1 by number of matches:", *_lines(buckets),
            f"S1 with 2+ matches from S2: {_pct(float((per['S2'] >= 2).mean()))}; from S3: "
            f"{_pct(float((per['S3'] >= 2).mean()))}; from either one source: "
            f"{_pct(float(((per['S2'] >= 2) | (per['S3'] >= 2)).mean()))}")

    # E3
    vc = tpv["cand_id"].value_counts()
    multi = vc[vc > 1]
    facts.one_owner_shared = len(multi) / max(len(vc), 1)
    rep.add("E3", f"Matched S2/S3 IDs: {len(vc):,}; in more than one S1 list: {len(multi):,} "
                  f"({_pct(facts.one_owner_shared)})")
    for cid in multi.index[:10]:
        owners = tpv.loc[tpv["cand_id"] == cid, "s1_id"].tolist()
        rep.add("E3", f"  {_describe(rec, int(idx.get_loc(cid)))}")
        rep.add("E3", *[f"    owner {_describe(rec, int(idx.get_loc(o)))}" for o in owners])

    # E4
    matched = rec["entity_id"].isin(pd.Index(vc.index))
    is23 = ~is_s1
    dist = pd.DataFrame({"source": rec["source"][is23], "country": rec["country"][is23],
                         "unmatched": ~matched[is23]})
    rep.add("E4", "% of train S2/S3 records in no ground-truth list:",
            *_lines(_rates(dist, ["unmatched"])),
            f"Overall: {_pct(float(dist['unmatched'].mean()))}")

    # E5
    c1, c2 = cc[p1v], cc[p2v]
    same = c1 == c2
    facts.country_same = float(same.mean()) if len(same) else 1.0
    rep.add("E5", f"True pairs with identical country: {_pct(facts.country_same)} "
                  f"({int((~same).sum()):,} mismatches of {len(same):,})")
    if (~same).any():
        mm = pd.DataFrame({"s1": np.asarray(cats[c1[~same]]), "cand": np.asarray(cats[c2[~same]]),
                           "i": np.flatnonzero(~same)})
        for (a, b), g in mm.groupby(["s1", "cand"]):
            rep.add("E5", f"  S1 {a} -> match {b}: {len(g):,}")
            for i in g["i"].head(5):
                rep.add("E5", f"    {_describe(rec, int(p1v[i]))}", f"    = {_describe(rec, int(p2v[i]))}")

    # E6 (train part)
    rep.add("E6", "Train, % empty (whitespace-only) per source x country:", *_lines(_empty_rates(rec)))

    # E7 (train part)
    rep.add("E7", "Train, % of records: has_a = standalone 5-6 digit token (prep v0 rule), "
                  "has_b = '600 001' style, has_c = ZIP+4:", *_lines(_rates(rec, ["has_a", "has_b", "has_c"])))
    extra = (rec["has_b"] | rec["has_c"]) & ~rec["has_a"]
    for c in cats:
        m = (rec["country"] == c).to_numpy()
        facts.postal_extra[str(c)] = float(extra[m].mean())
        rep.add("E7", f"Train {c}: records with (b) or (c) but not (a): {_pct(facts.postal_extra[str(c)])}")
    for c in ("India", "US"):
        pool = np.flatnonzero((rec["country"] == c).to_numpy() & ~rec["has_a"].to_numpy() & rec["addr"].ne("").to_numpy())
        pick = rng.choice(pool, size=min(N_ADDR_EXAMPLES, len(pool)), replace=False)
        rep.add("E7", f"(d) {len(pick)} random non-empty {c} addresses with NO (a) token:",
                *[f"    {rec['addr'].iloc[int(i)]}" for i in pick])
    t1 = rec["post_tokens"].iloc[p1v].to_numpy()
    t2 = rec["post_tokens"].iloc[p2v].to_numpy()
    both = (t1 != "") & (t2 != "")
    shared = np.fromiter((bool(set(a.split()) & set(b.split())) for a, b in zip(t1[both], t2[both], strict=True)),
                         dtype=bool, count=int(both.sum()))
    facts.postal_pair_share = float(shared.sum()) / max(len(t1), 1)
    rep.add("E7", f"True pairs where both sides have an (a)/(b) token: {int(both.sum()):,} ({_pct(float(both.mean()) if len(both) else 0.0)} of pairs); "
                  f"of these sharing a token: {_pct(float(shared.mean()) if len(shared) else 0.0)}; "
                  f"=> share of ALL true pairs linked by a shared postal token: {_pct(facts.postal_pair_share)}")
    c1_both = c1[both]
    for code, c in enumerate(cats):
        k = c1_both == code
        n_c = int((c1 == code).sum())
        if k.any():
            rep.add("E7", f"  S1 {c}: both-sides-token pairs {int(k.sum()):,} "
                          f"({_pct(float(k.sum()) / max(n_c, 1))} of {c} pairs), sharing {_pct(float(shared[k].mean()))}")

    # E8
    key = name_key(rec["name"])
    k1 = key.iloc[p1v].reset_index(drop=True)
    k2 = key.iloc[p2v].reset_index(drop=True)
    exact = (k1 == k2).to_numpy(dtype=bool)
    rep.add("E8", f"All true pairs: {_pct(float(exact.mean()) if len(exact) else 0.0)} exact")
    for c in cats:
        m = c1 == cats.get_loc(c)
        if m.any():
            rep.add("E8", f"  S1 {c}: {_pct(float(exact[m].mean()))} of {int(m.sum()):,} pairs")
    del key, k1, k2

    # E11
    for c in cats:
        cand = np.flatnonzero(c1 == cats.get_loc(c))
        pick = rng.choice(cand, size=min(N_PAIR_EXAMPLES, len(cand)), replace=False)
        rep.add("E11", f"--- {c}: {len(pick)} random true pairs (S1 name | matched name || S1 address | matched address)")
        for i in pick:
            r1, r2 = rec.iloc[int(p1v[i])], rec.iloc[int(p2v[i])]
            rep.add("E11", f"  {r1['name']} | {r2['name']} ({r2['entity_id'][:2]}) || {r1['addr']} | {r2['addr']}")

    # E12 (train part)
    rep.add("E12", "Train, % of records with non-ASCII characters (name or address):",
            *_lines((100 * rec.groupby("country", observed=True)["non_ascii"].mean()).round(2)))

    # E9 — drop raw text first to keep peak memory down
    rec.drop(columns=["name", "addr", "post_tokens"], inplace=True)
    gc.collect()
    with track_stage("eda-E9"):
        facts.recall = _e9(rec, p1v, p2v, cc, cats, is_s1, rep, n_quantile_pairs, n_rank_s1, rng)
    return facts, scan


def _e9(rec: pd.DataFrame, p1v: NDArray[np.intp], p2v: NDArray[np.intp], cc: NDArray[np.int32],
        cats: pd.Index, is_s1: NDArray[np.bool_], rep: Report, n_quantile_pairs: int, n_rank_s1: int,
        rng: np.random.Generator) -> dict[int, float]:
    """E9: TF-IDF cosine quantiles and same-country recall at K for true matches.

    Memory: one country's S2/S3 pool as a transposed float32 CSR matrix plus
    one RANK_BATCH x pool sparse score block at a time.

    Returns:
        Overall recall at each K in RECALL_KS.
    """
    names = rec["name_norm"]
    t0 = time.perf_counter()
    vec = NameVectorizer()
    for start in range(0, len(names), SCAN_CHUNK_ROWS):
        vec.fit(names.iloc[start : start + SCAN_CHUNK_ROWS])
    rep.add("E9", f"IDF fitted on {len(names):,} train names (S1+S2+S3, normalised with prep v0).")
    logger.info("E9: IDF fit %.1fs", time.perf_counter() - t0)

    def cosines(a: NDArray[np.intp], b: NDArray[np.intp]) -> NDArray[np.float32]:
        """Row-wise cosine between records a[i] and b[i]."""
        va, vb = vec.transform(names.iloc[a]), vec.transform(names.iloc[b])
        return np.asarray(va.multiply(vb).sum(axis=1), dtype=np.float32).ravel()

    n = min(n_quantile_pairs, len(p1v))
    pick = rng.choice(len(p1v), size=n, replace=False)
    true_cos = cosines(p1v[pick], p2v[pick])
    s1_pos = np.flatnonzero(is_s1)
    rs1 = rng.choice(s1_pos, size=min(n_quantile_pairs, len(s1_pos)), replace=False)
    rcand = np.empty_like(rs1)
    for code in np.unique(cc[rs1]):
        pool = np.flatnonzero(~is_s1 & (cc == code))
        m = cc[rs1] == code
        rcand[m] = rng.choice(pool, size=int(m.sum()))
    rand_cos = cosines(rs1, rcand)
    q = pd.DataFrame({f"q{int(p * 100)}": [np.quantile(true_cos, p), np.quantile(rand_cos, p)] for p in QUANTILES},
                     index=[f"true pairs (n={len(true_cos):,})", f"random same-country (n={len(rand_cos):,})"]).round(3)
    rep.add("E9", "Cosine quantiles:", *_lines(q))
    logger.info("E9: quantiles done at %.1fs", time.perf_counter() - t0)

    order = np.argsort(p1v, kind="stable")
    s1_sorted, p2_sorted = p1v[order], p2v[order]
    with_match = np.unique(p1v)
    sample = np.sort(rng.choice(with_match, size=min(n_rank_s1, len(with_match)), replace=False))
    all_ranks: list[NDArray[np.float64]] = []
    for code in np.unique(cc[sample]):
        pool = np.flatnonzero(~is_s1 & (cc == code))
        col_of = np.full(len(rec), -1, dtype=np.int64)
        col_of[pool] = np.arange(len(pool))
        blocks = [vec.transform(names.iloc[pool[i : i + SCAN_CHUNK_ROWS]]) for i in range(0, len(pool), SCAN_CHUNK_ROWS)]
        xt = sp.vstack(blocks, format="csr").T.tocsr()
        del blocks
        logger.info("E9: %s pool %d vectorised at %.1fs", cats[code], len(pool), time.perf_counter() - t0)
        s1c = sample[cc[sample] == code]
        ranks_c: list[NDArray[np.float64]] = []
        for i in range(0, len(s1c), RANK_BATCH):
            batch = s1c[i : i + RANK_BATCH]
            qv = vec.transform(names.iloc[batch])
            lo = np.searchsorted(s1_sorted, batch, side="left")
            hi = np.searchsorted(s1_sorted, batch, side="right")
            ranks_c.append(true_match_ranks(qv, xt, [col_of[p2_sorted[a:b]] for a, b in zip(lo, hi, strict=True)]))
        r = np.concatenate(ranks_c) if ranks_c else np.array([])
        all_ranks.append(r)
        rec_c = recall_at_k(r, RECALL_KS)
        logger.info("E9: %s ranked %d S1 at %.1fs", cats[code], len(s1c), time.perf_counter() - t0)
        rep.add("E9", f"  {cats[code]}: {len(s1c):,} S1, {len(r):,} true pairs, pool {len(pool):,} S2/S3 | "
                      + "  ".join(f"R@{k}={v:.3f}" for k, v in rec_c.items()))
        del xt, col_of
        gc.collect()
    ranks = np.concatenate(all_ranks) if all_ranks else np.array([])
    recall = recall_at_k(ranks, RECALL_KS)
    rep.add("E9", f"Recall at K of true matches among ALL same-country S2/S3 by name cosine "
                  f"({len(sample):,} sampled S1 with matches, {len(ranks):,} true pairs; cross-country "
                  f"or zero-cosine matches count as misses):",
            "  " + "  ".join(f"R@{k}={v:.3f}" for k, v in recall.items()))
    return recall


# --------------------------------------------------------------------------- test + implications


def analyse_test(data_dir: Path, rep: Report, train_counts: pd.DataFrame) -> ScanResult | None:
    """Test-split parts of E1, E6, E7, E10, E12. Returns None if test is missing."""
    if not (data_dir / "test" / "test_source1.tsv").exists():
        logger.warning("No test split under %s; test sections skipped", data_dir)
        for sec in ("E1", "E6", "E7", "E10", "E12"):
            rep.add(sec, "(test split not found under --data-dir; test part skipped)")
        return None
    with track_stage("eda-test-load"):
        rec = load_split(data_dir, "test")
    with track_stage("eda-test-scan"):
        scan = scan_records(rec)
    counts = rec.groupby(["source", "country"], observed=True).size().unstack(fill_value=0)
    ratio = (counts.sum(axis=1) / train_counts.sum(axis=1)).round(3)
    new = [c for c in counts.columns if c not in train_counts.columns]
    rep.add("E1", "Test rows per source x country:", *_lines(counts),
            "Test/train total size ratio per source:", *_lines(ratio),
            f"Countries in test but not in train: {new or 'none'}")
    for c in new:
        rep.add("E1", f"  {c} rows per source: " + ", ".join(f"{s} {int(counts.loc[s, c]):,}" for s in counts.index))
    rep.add("E6", "Test, % empty (whitespace-only) per source x country:", *_lines(_empty_rates(rec)))
    rep.add("E7", "Test, % of records with has_a / has_b / has_c:", *_lines(_rates(rec, ["has_a", "has_b", "has_c"])))
    rep.add("E12", "Test, % of records with non-ASCII characters (name or address):",
            *_lines((100 * rec.groupby("country", observed=True)["non_ascii"].mean()).round(2)))
    return scan


def _implications(rep: Report, f: TrainFacts) -> None:
    """IMPLICATIONS section, derived only from the computed numbers."""
    owner = ("supported" if f.one_owner_shared == 0 else
             "mostly supported" if f.one_owner_shared < 0.005 else "NOT supported")
    rep.add("IMPLICATIONS",
            f"- E3: {_pct(f.one_owner_shared)} of matched S2/S3 IDs appear in more than one S1 list "
            f"-> one-owner rule {owner} (rule: 0% supported, <0.5% mostly).",
            f"- E5: {_pct(f.country_same)} of true pairs share a country label -> cross-country fallback pass "
            f"{'NEEDED' if f.country_same < 0.995 else 'not needed'} (threshold 99.5%).")
    if f.recall:
        ks = sorted(f.recall)
        hit = next((k for k in ks if f.recall[k] >= 0.95), None)
        if hit is None:
            top = f.recall[ks[-1]]
            k95 = next(k for k in ks if f.recall[k] >= 0.95 * top)
            txt = (f"no K <= {ks[-1]} reaches 95% (R@{ks[-1]}={top:.3f}), so name-only Pass A cannot be the "
                   f"only pass; K={k95} is the smallest K capturing >= 95% of what K={ks[-1]} captures "
                   f"(R@{k95}={f.recall[k95]:.3f}) -> start Pass A at K={k95}, recover the rest with Passes B/C")
        else:
            txt = f"R@{hit}={f.recall[hit]:.3f} is the smallest K reaching 95% -> start Pass A at K={hit}"
        rep.add("IMPLICATIONS", f"- E9: {txt} (config.BLOCK_TOP_K is {config.BLOCK_TOP_K}).")
    extra = ", ".join(f"{c} {_pct(v)}" for c, v in f.postal_extra.items())
    widen = any(v >= 0.01 for v in f.postal_extra.values())
    rep.add("IMPLICATIONS",
            f"- E7: records gaining a postal token from (b)/(c) patterns: {extra} -> "
            f"{'WIDEN' if widen else 'do not widen'} the prep postal rule (threshold 1 pp; check E7(d) samples by eye).",
            f"- E7: {_pct(f.postal_pair_share)} of all true pairs share a postal token -> Pass C "
            f"{'worth building' if f.postal_pair_share >= 0.05 else 'low value'} (threshold 5% of true pairs).")


def run_eda(data_dir: Path, artifacts_dir: Path, n_quantile_pairs: int = N_QUANTILE_PAIRS,
            n_rank_s1: int = N_RANK_S1, seed: int = config.SEED) -> str:
    """Run the full EDA and write ``<artifacts_dir>/eda/eda_report.txt``.

    Args:
        data_dir: Folder with ``train/`` (required) and ``test/`` (optional).
        artifacts_dir: Output root.
        n_quantile_pairs: Sampled pairs for E9 quantiles.
        n_rank_s1: Sampled S1 for E9 recall at K.
        seed: Random seed for every sample.

    Returns:
        The report text.

    Raises:
        ValueError: If an input file fails the ``io_utils`` checks.
    """
    rng = np.random.default_rng(seed)
    rep = Report()
    facts, train_scan = analyse_train(data_dir, rep, n_quantile_pairs, n_rank_s1, rng)
    gc.collect()
    test_scan = analyse_test(data_dir, rep, facts.counts)

    for split, scan in (("train", train_scan), ("test", test_scan)):
        if scan is None:
            continue
        for c in sorted(scan.name_tokens):
            rep.add("E10", f"--- {split} {c}: top {TOP_TOKENS} name tokens (normalised)", *_top(scan.name_tokens[c], TOP_TOKENS))
            rep.add("E10", f"--- {split} {c}: top {TOP_TOKENS} address tokens (normalised)", *_top(scan.addr_tokens[c], TOP_TOKENS))
    chars = train_scan.non_ascii_chars + (test_scan.non_ascii_chars if test_scan else Counter())
    rep.add("E12", f"Top {TOP_CHARS} non-ASCII characters (train{' + test' if test_scan else ''}, names + addresses):",
            *[f"    {ch!r} U+{ord(ch):04X} {unicodedata.name(ch, '?')}: {n:,}" for ch, n in chars.most_common(TOP_CHARS)])
    _implications(rep, facts)

    text = rep.text()
    out = artifacts_dir / "eda" / "eda_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    for line in text.splitlines():
        logger.info("%s", line)
    logger.info("EDA report written to %s", out)
    return text


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="folder with train/ and test/")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="report goes to <artifacts>/eda/")
    parser.add_argument("--n-rank-s1", type=int, default=N_RANK_S1, help="sampled S1 for E9 recall at K")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    paths = config.get_paths(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    setup_logging(args.log_level, paths.log_dir)
    with fulldata_lock(paths.data_dir, "src.eda " + " ".join(sys.argv[1:])), track_stage("eda"):
        run_eda(paths.data_dir, paths.artifacts_dir, n_rank_s1=args.n_rank_s1)


if __name__ == "__main__":
    main()
