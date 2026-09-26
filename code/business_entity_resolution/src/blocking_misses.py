"""Blocking miss analysis (Phase 3): true train pairs that are not candidates.

Usage (from code/business_entity_resolution/, after ``--stage block --split train``):
    python -m src.blocking_misses --data-dir PATH --artifacts-dir PATH --n-examples 50

Reads ``<artifacts>/records_train.parquet``, ``<artifacts>/candidates_train.parquet``,
``<artifacts>/blocking_capped_out_train.parquet`` and ``<data>/train/train_ground_truth.tsv``. Writes to ``<artifacts>/blocking_misses/``:

- ``missed_pairs.tsv``: one row per missed true pair (``MISS_COLUMNS``):
  IDs, country, candidate source, raw names/addresses, ``sim_<X>`` = char/word
  TF-IDF cosine of every blocking pass text (``sim_A`` is the name similarity,
  on the pass A text) and ``sim_addr`` (char 3-gram cosine on ``addr_norm``),
  empty-address flags, name script of both sides, the S1's candidate count,
  one boolean column per cause in ``CAUSES`` and the first matching ``cause``.
- ``report.txt``: miss counts and shares per country and per source, cause
  breakdown, similarity quantiles and ``--n-examples`` random misses side by side.

Causes (not exclusive; ``cause`` is the first in this order, else "other"):
- ``pushed_out_by_cap``: the pair is in blocking's uncapped union but beyond
  ``config.MAX_CANDIDATES_PER_S1`` (``blocking_capped_out_train.parquet``,
  written by the train block stage; all False with a WARNING if absent).
- ``empty_address``: ``addr_norm`` of the S1 or the candidate is empty.
- ``cross_script``: one name (``name_raw``) is pure Latin, the other pure Indic.
- ``low_name_and_addr``: ``sim_A`` and ``sim_addr`` both below ``LOW_SIM``.

Cosines use a ``HashingVectorizer`` (``HASH_FEATURES`` buckets) with the
smooth IDF of ``TfidfVectorizer`` computed per country over the country's
non-empty texts, so they equal the blocking passes' cosines up to hash
collisions without holding a country's TF-IDF matrix.

Memory: entity IDs as one Arrow string array, country/source codes, a few
int32/float32 arrays per record and int64 keys of the true pairs. Candidates
are streamed in ``CAND_BATCH_ROWS`` batches and records in ``RECORD_BATCH_ROWS``
batches (texts kept only for records in a missed pair), so the peak is set by
the record count, well under 4 GB on full train.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src import blocking, config, io_utils
from src.config import Paths
from src.fulldata_lock import fulldata_lock
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

HASH_FEATURES: int = 1 << 20
RECORD_BATCH_ROWS: int = 100_000
CAND_BATCH_ROWS: int = 2_000_000
LOW_SIM: float = 0.3
ADDR_SPEC = blocking.PassSpec("addr", ("addr_norm",), "char", 3, top_k=0, max_df=1.0, min_query_terms=0)
SIM_SPECS: tuple[blocking.PassSpec, ...] = (*blocking.PASSES, ADDR_SPEC)
CAUSES: tuple[str, ...] = ("pushed_out_by_cap", "empty_address", "cross_script", "low_name_and_addr")
TEXT_COLUMNS: tuple[str, ...] = tuple(sorted({"name_raw", "addr_raw", *(c for s in SIM_SPECS for c in s.columns)}))
RECORD_COLUMNS: tuple[str, ...] = ("entity_id", "source", "country", *TEXT_COLUMNS)
MISS_COLUMNS: tuple[str, ...] = (
    "s1_id", "cand_id", "country", "cand_source", "s1_name", "cand_name", "s1_addr", "cand_addr",
    *(f"sim_{s.name}" for s in SIM_SPECS), "s1_addr_empty", "cand_addr_empty", "s1_script", "cand_script",
    "s1_n_cands", *CAUSES, "cause",
)
_INDIC = r"[ऀ-෿]"
_LATIN = r"[A-Za-zÀ-ɏḀ-ỿ]"


@dataclass
class _DocFreq:
    """Hashed document frequencies per (spec name, country code)."""

    df: dict[tuple[str, int], NDArray[np.int64]] = field(default_factory=dict)
    n_docs: dict[tuple[str, int], int] = field(default_factory=dict)

    def idf(self, spec: str, country: int) -> NDArray[np.float32]:
        """Smooth IDF as TfidfVectorizer: ln((1 + n) / (1 + df)) + 1."""
        df = self.df.get((spec, country), np.zeros(HASH_FEATURES, np.int64))
        n = self.n_docs.get((spec, country), 0)
        return (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)


def script_of(text: pd.Series) -> pd.Series:
    """Script of each text: "Latin", "Indic", "mixed" (both) or "other" (neither).

    Args:
        text: str Series.

    Returns:
        str Series aligned with ``text``.
    """
    indic = text.str.contains(_INDIC, regex=True).to_numpy(dtype=bool)
    latin = text.str.contains(_LATIN, regex=True).to_numpy(dtype=bool)
    return pd.Series(np.select([indic & latin, indic, latin], ["mixed", "Indic", "Latin"], "other"), index=text.index)


def _hasher(spec: blocking.PassSpec) -> HashingVectorizer:
    """Raw-count hashing vectorizer with the tokenisation of ``blocking._vectorizer``."""
    kw = {"token_pattern": r"\S+"} if spec.analyzer == "word" else {}
    return HashingVectorizer(analyzer=spec.analyzer, ngram_range=(spec.ngram, spec.ngram), lowercase=False,
                             n_features=HASH_FEATURES, alternate_sign=False, norm=None, dtype=np.float32, **kw)


def _index_of(ids: pa.Array, values: pa.Array | pa.ChunkedArray) -> NDArray[np.int64]:
    """Position of each value in ``ids`` (-1 if absent)."""
    return pc.index_in(values, value_set=ids).fill_null(-1).to_numpy().astype(np.int64)  # type: ignore[no-any-return]  # pyarrow is untyped


def _truth_pairs(path: Path, ids: pa.Array) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Unique true (s1, cand) record positions, sorted by (s1, cand); singletons skipped."""
    gt = io_utils.read_ground_truth_pairs(path)
    gt = gt[gt["cand_id"] != ""]
    s1 = _index_of(ids, pa.array(gt["s1_id"], pa.string()))
    cand = _index_of(ids, pa.array(gt["cand_id"], pa.string()))
    del gt
    bad = (s1 < 0) | (cand < 0)
    if bad.any():
        logger.warning("%d ground-truth pairs reference IDs not in records, skipped", int(bad.sum()))
    key = np.unique(s1[~bad] * len(ids) + cand[~bad])
    return key // len(ids), key % len(ids)


def _scan_candidates(path: Path, ids: pa.Array, truth_key: NDArray[np.int64]) -> tuple[
    NDArray[np.bool_], NDArray[np.int32]
]:
    """Stream the candidates once: which true pairs are found and candidates per S1.

    Memory: one batch of ``CAND_BATCH_ROWS`` pairs plus one int32 per record.

    Args:
        path: candidates_train.parquet (``s1_id, cand_id`` str).
        ids: Entity ID of every record position.
        truth_key: Sorted unique ``s1 * len(ids) + cand`` of the pairs to look up.

    Returns:
        ``(found, n_cands)``: bool per ``truth_key``; candidates per record
        position (non-zero for S1 only).
    """
    n = len(ids)
    found = np.zeros(len(truth_key), dtype=bool)
    n_cands = np.zeros(n, dtype=np.int32)
    unknown = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=CAND_BATCH_ROWS, columns=["s1_id", "cand_id"]):
        s1, cand = _index_of(ids, batch.column("s1_id")), _index_of(ids, batch.column("cand_id"))
        ok = (s1 >= 0) & (cand >= 0)
        unknown += int((~ok).sum())
        s1, cand = s1[ok], cand[ok]
        if len(truth_key):
            key = s1 * n + cand
            pos = np.minimum(np.searchsorted(truth_key, key), len(truth_key) - 1)
            found[pos[truth_key[pos] == key]] = True
        n_cands += np.bincount(s1, minlength=n).astype(np.int32)
    if unknown:
        logger.warning("%d candidate pairs reference IDs not in records, skipped", unknown)
    return found, n_cands


def _scan_records(path: Path, need: NDArray[np.bool_], country: NDArray[np.int32]) -> tuple[_DocFreq, pd.DataFrame]:
    """Stream the records once: hashed document frequencies and the texts of ``need`` records.

    Memory: one batch of ``RECORD_BATCH_ROWS`` records and its hashed matrices,
    ``HASH_FEATURES`` int64 per (spec, country), and the kept texts.

    Args:
        path: records_train.parquet with every ``TEXT_COLUMNS`` column (str).
        need: True for record positions whose texts are kept.
        country: Country code per record position.

    Returns:
        ``(doc_freq, texts)``: texts has ``TEXT_COLUMNS``, indexed by record position.
    """
    freq = _DocFreq()
    hashers = {s.name: _hasher(s) for s in SIM_SPECS}
    keep: list[pd.DataFrame] = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=RECORD_BATCH_ROWS, columns=list(TEXT_COLUMNS)):
        b = batch.to_pandas()
        cc = country[offset: offset + len(b)]
        for s in SIM_SPECS:
            text = blocking.pass_text(b, s)
            ok = (text != "").to_numpy()
            x = hashers[s.name].transform(text[ok]).tocsr()
            for c in np.unique(cc[ok]):
                rows = x[cc[ok] == c]
                k = (s.name, int(c))
                freq.df[k] = freq.df.get(k, 0) + np.bincount(rows.indices, minlength=HASH_FEATURES)
                freq.n_docs[k] = freq.n_docs.get(k, 0) + rows.shape[0]
        m = need[offset: offset + len(b)]
        if m.any():
            keep.append(b[m].set_axis(np.flatnonzero(m) + offset))
        offset += len(b)
    texts = pd.concat(keep) if keep else pd.DataFrame(columns=list(TEXT_COLUMNS), dtype=str)
    return freq, texts


def pair_cosines(a: pd.Series, b: pd.Series, country: NDArray[np.int32], spec: blocking.PassSpec,
                 freq: _DocFreq) -> NDArray[np.float32]:
    """Hashed TF-IDF cosine of aligned text pairs, with the IDF of each pair's country.

    Args:
        a: Left texts (str; "" gives cosine 0).
        b: Right texts, same length as ``a``.
        country: Country code per pair (the S1's).
        spec: Analyzer / n-gram of the pass.
        freq: Document frequencies from ``_scan_records``.

    Returns:
        float32 cosine per pair.
    """
    h = _hasher(spec)
    a_arr, b_arr = a.to_numpy(dtype=object), b.to_numpy(dtype=object)
    out = np.zeros(len(a_arr), dtype=np.float32)
    for c in np.unique(country):
        m = country == c
        idf = sp.diags(freq.idf(spec.name, int(c)))
        xa = l2_normalize(h.transform(a_arr[m]) @ idf)
        xb = l2_normalize(h.transform(b_arr[m]) @ idf)
        out[m] = np.asarray(xa.multiply(xb).sum(axis=1), dtype=np.float32).ravel()
    return out


def _capped_out(path: Path, ids: pa.Array, miss_key: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Bool per missed pair: listed in blocking's capped-out file (all False if absent)."""
    if not path.exists():
        logger.warning("%s not found (rerun --stage block --split train): pushed_out_by_cap is all False", path)
        return np.zeros(len(miss_key), dtype=bool)
    t = pq.read_table(path, columns=["s1_id", "cand_id"])
    key = _index_of(ids, t["s1_id"]) * len(ids) + _index_of(ids, t["cand_id"])
    return np.isin(miss_key, key)


def analyze(records_path: Path, candidates_path: Path, capped_out_path: Path,
            gt_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the true pairs missing from the candidates and describe each miss.

    Args:
        records_path: records_train.parquet with ``RECORD_COLUMNS`` (str).
        candidates_path: candidates_train.parquet with ``s1_id, cand_id`` (str).
        capped_out_path: blocking_capped_out_train.parquet (``s1_id, cand_id``).
        gt_path: train ground-truth TSV.

    Returns:
        ``(truth, misses)``: truth has one row per true pair (categorical
        ``country, cand_source``; bool ``missed``); misses has one row per
        missed pair with ``MISS_COLUMNS``.

    Raises:
        ValueError: If a required column is missing.
    """
    io_utils.require_columns(pq.read_schema(records_path).names, RECORD_COLUMNS, "blocking_misses records")
    io_utils.require_columns(pq.read_schema(candidates_path).names, ("s1_id", "cand_id"),
                             "blocking_misses candidates")
    t = pq.read_table(records_path, columns=["entity_id", "source", "country"])
    ids = t["entity_id"].combine_chunks()
    cty, src = t["country"].combine_chunks().dictionary_encode(), t["source"].combine_chunks().dictionary_encode()
    del t
    country = cty.indices.to_numpy().astype(np.int32)
    source = src.indices.to_numpy().astype(np.int32)
    countries, sources = cty.dictionary.to_pylist(), src.dictionary.to_pylist()

    s1, cand = _truth_pairs(gt_path, ids)
    found, n_cands = _scan_candidates(candidates_path, ids, s1 * len(ids) + cand)
    truth = pd.DataFrame({"country": pd.Categorical.from_codes(country[s1], countries),
                          "cand_source": pd.Categorical.from_codes(source[cand], sources), "missed": ~found})
    ms, mc = s1[~found], cand[~found]
    logger.info("%d true pairs, %d missed (%.2f%%)", len(found), len(ms), 100.0 * len(ms) / max(len(found), 1))

    need = np.zeros(len(ids), dtype=bool)
    need[ms] = need[mc] = True
    freq, tx = _scan_records(records_path, need, country)
    left, right = tx.reindex(ms), tx.reindex(mc)
    script = script_of(tx["name_raw"])
    miss = pd.DataFrame({
        "s1_id": ids.take(pa.array(ms)).to_numpy(zero_copy_only=False),
        "cand_id": ids.take(pa.array(mc)).to_numpy(zero_copy_only=False),
        "country": np.asarray(countries, dtype=object)[country[ms]],
        "cand_source": np.asarray(sources, dtype=object)[source[mc]],
        "s1_name": left["name_raw"].to_numpy(), "cand_name": right["name_raw"].to_numpy(),
        "s1_addr": left["addr_raw"].to_numpy(), "cand_addr": right["addr_raw"].to_numpy(),
    })
    for s in SIM_SPECS:
        miss[f"sim_{s.name}"] = pair_cosines(blocking.pass_text(left, s), blocking.pass_text(right, s),
                                             country[ms], s, freq)
    miss["s1_addr_empty"] = (left["addr_norm"] == "").to_numpy()
    miss["cand_addr_empty"] = (right["addr_norm"] == "").to_numpy()
    miss["s1_script"], miss["cand_script"] = script.reindex(ms).to_numpy(), script.reindex(mc).to_numpy()
    miss["s1_n_cands"] = n_cands[ms]
    miss["pushed_out_by_cap"] = _capped_out(capped_out_path, ids, ms * len(ids) + mc)
    miss["empty_address"] = miss["s1_addr_empty"] | miss["cand_addr_empty"]
    pair_scripts = miss["s1_script"] + "/" + miss["cand_script"]
    miss["cross_script"] = pair_scripts.isin(["Latin/Indic", "Indic/Latin"])
    miss["low_name_and_addr"] = (miss["sim_A"] < LOW_SIM) & (miss["sim_addr"] < LOW_SIM)
    miss["cause"] = np.select([miss[c].to_numpy() for c in CAUSES], list(CAUSES), "other")
    return truth, miss[list(MISS_COLUMNS)]


def _share_table(truth: pd.DataFrame, by: str) -> pd.DataFrame:
    """Missed pairs and share of true pairs per ``by`` value plus an ALL row."""
    g = truth.groupby(by, observed=True).agg(n_true=("missed", "size"), n_missed=("missed", "sum"))
    g.loc["ALL"] = [len(truth), int(truth["missed"].sum())]
    g["share_missed"] = (g["n_missed"] / g["n_true"].clip(lower=1)).round(4)
    return g


def _report(truth: pd.DataFrame, miss: pd.DataFrame, n_examples: int, seed: int) -> str:
    """Plain-text report: miss shares, causes, similarity quantiles, random examples."""
    sims = [f"sim_{s.name}" for s in SIM_SPECS]
    flags = miss.groupby("country")[list(CAUSES)].sum()
    flags.loc["ALL"] = miss[list(CAUSES)].sum()
    parts = [
        "Blocking misses: true pairs not in candidates_train.parquet",
        f"cap {config.MAX_CANDIDATES_PER_S1}; LOW_SIM {LOW_SIM}; sim_A = name (pass A text), "
        "sim_addr = addr_norm char 3-gram",
        "", "== Missed pairs per country", _share_table(truth, "country").to_string(),
        "", "== Missed pairs per source", _share_table(truth, "cand_source").to_string(),
        "", "== Causes (flags, not exclusive)", flags.to_string(),
        "", "== Causes (primary, first flag in order)",
        pd.crosstab(miss["country"], miss["cause"], margins=True, margins_name="ALL").to_string(),
        "", "== Similarity of missed pairs", miss[sims].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string(),
        "", "== Name scripts of missed pairs (S1 / candidate)",
        (miss["s1_script"] + " / " + miss["cand_script"]).value_counts().to_string(),
        "", f"== Examples ({min(n_examples, len(miss))} random misses, seed {seed})",
    ]
    for i, r in enumerate(miss.sample(n=min(n_examples, len(miss)), random_state=seed).itertuples(), 1):
        sim = " ".join(f"{c}={getattr(r, c):.3f}" for c in sims)
        parts += [
            f"[{i}] {r.s1_id} -> {r.cand_id}  {r.country} {r.cand_source}  cause={r.cause}  "
            f"S1 cands={r.s1_n_cands}  scripts={r.s1_script}/{r.cand_script}  {sim}",
            f"    S1   name: {r.s1_name}", f"         addr: {r.s1_addr}",
            f"    cand name: {r.cand_name}", f"         addr: {r.cand_addr}",
        ]
    return "\n".join(parts) + "\n"


def run(paths: Paths, n_examples: int) -> Path:
    """Analyse the train blocking misses and write the report files.

    Args:
        paths: Run directories (records/candidates in ``artifacts_dir``,
            ground truth in ``data_dir/train``).
        n_examples: Random missed pairs printed side by side.

    Returns:
        The output folder ``<artifacts_dir>/blocking_misses``.

    Raises:
        ValueError: If a required column is missing.
    """
    art = paths.artifacts_dir
    truth, miss = analyze(art / "records_train.parquet", art / "candidates_train.parquet",
                          art / "blocking_capped_out_train.parquet", paths.data_dir / "train" / "train_ground_truth.tsv")
    out = art / "blocking_misses"
    out.mkdir(parents=True, exist_ok=True)
    miss.to_csv(out / "missed_pairs.tsv", sep="\t", index=False, float_format="%.4f", lineterminator="\n",
                encoding="utf-8")
    report = _report(truth, miss, n_examples, config.SEED)
    (out / "report.txt").write_text(report, encoding="utf-8", newline="\n")
    for line in report.split("\n== Examples")[0].splitlines():
        logger.info("%s", line)
    logger.info("Wrote %s and %s", out / "missed_pairs.tsv", out / "report.txt")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train/train_ground_truth.tsv")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="folder with records/candidates_train")
    parser.add_argument("--n-examples", type=int, default=50, help="random misses printed side by side")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    paths = config.get_paths(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    setup_logging(args.log_level, paths.log_dir)
    with fulldata_lock(paths.data_dir, "src.blocking_misses " + " ".join(sys.argv[1:])), \
            track_stage("blocking_misses"):
        run(paths, args.n_examples)


if __name__ == "__main__":
    main()
