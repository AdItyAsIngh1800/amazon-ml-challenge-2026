"""Tests for blocking_misses.py on synthetic data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src import blocking, blocking_misses, config, normalize
from src.config import Paths

ROWS = [
    ("S1-1", "US", "acme widgets inc", "12 main st springfield"),
    ("S1-2", "US", "blue river cafe", "400 oak ave portland"),
    ("S1-3", "US", "zyx lonely name", "9 nowhere rd"),
    ("S2-1", "US", "acme widgets", "12 main street springfield"),
    ("S3-1", "US", "acme widgets", ""),
    ("S2-2", "US", "blue river coffee", "400 oak avenue portland"),
    ("S3-2", "US", "red mountain bakery", "77 pine rd denver"),
    ("S2-4", "US", "qqq ppp", "1 elsewhere"),
    ("S1-4", "India", "sharma traders", "shop 5 mg road pune 411001"),
    ("S2-3", "India", "sharma traders pvt ltd", "mg road pune 411001"),
    ("S3-3", "India", "शर्मा ट्रेडर्स", "पुणे"),
]
GT = "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-2\nS1-3\tS2-4\nS1-4\tS2-3,S3-3\n"


def _paths(tmp: Path) -> Paths:
    """Records + ground truth + candidates (blocking with cap 1) for two countries."""
    src = pd.DataFrame(ROWS, columns=["entity_id", "country", "business_name", "business_address"])
    recs = pd.concat(
        [normalize.build_records(g[["entity_id", "business_name", "business_address", "country"]], s)
         for s, g in src.groupby(src["entity_id"].str[:2])],
        ignore_index=True,
    )
    paths = Paths(tmp / "data", tmp / "art", tmp / "out")
    paths.artifacts_dir.mkdir(parents=True)
    recs.to_parquet(paths.artifacts_dir / "records_train.parquet", index=False)
    (paths.data_dir / "train").mkdir(parents=True)
    (paths.data_dir / "train" / "train_ground_truth.tsv").write_text(GT, encoding="utf-8", newline="\n")
    blocking.run_stage(paths, "train")
    return paths


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Paths, pd.DataFrame]:
    """Blocking (cap 1) then the miss analysis in tiny batches; returns paths and missed_pairs.tsv."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    monkeypatch.setattr(config, "MAX_CANDIDATES_PER_S1", 1)
    monkeypatch.setattr(blocking_misses, "RECORD_BATCH_ROWS", 3)
    monkeypatch.setattr(blocking_misses, "CAND_BATCH_ROWS", 2)
    paths = _paths(tmp_path)
    out = blocking_misses.run(paths, n_examples=2)
    misses = pd.read_csv(out / "missed_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    return paths, misses


def test_misses_are_true_pairs_not_in_candidates(run: tuple[Paths, pd.DataFrame]) -> None:
    """Missed pairs equal ground truth minus candidates, computed independently."""
    paths, misses = run
    cands = pd.read_parquet(paths.artifacts_dir / "candidates_train.parquet")
    truth = {(s, c) for s, cs in [ln.split("\t") for ln in GT.splitlines()[1:]] for c in cs.split(",") if c}
    want = truth - set(zip(cands["s1_id"], cands["cand_id"], strict=True))
    assert want and set(zip(misses["s1_id"], misses["cand_id"], strict=True)) == want
    n = cands.groupby("s1_id").size()
    assert (misses["s1_n_cands"].astype(int) == n.reindex(misses["s1_id"]).fillna(0).to_numpy()).all()


def _row(misses: pd.DataFrame, s1: str, cand: str) -> pd.Series:
    """The missed_pairs.tsv row of one pair."""
    return misses[(misses["s1_id"] == s1) & (misses["cand_id"] == cand)].iloc[0]


def test_causes(run: tuple[Paths, pd.DataFrame]) -> None:
    """Cap push-out, empty address, cross-script and low-similarity flags."""
    _, misses = run
    # Cap 1 keeps S2-1 (best in both passes); S3-1 is in pass A's top-K, cut by the cap.
    row = _row(misses, "S1-1", "S3-1")
    assert row["cand_addr_empty"] == "True" and row["empty_address"] == "True"
    assert row["pushed_out_by_cap"] == "True" and row["cause"] == "pushed_out_by_cap"
    row = _row(misses, "S1-4", "S3-3")
    assert (row["s1_script"], row["cand_script"], row["cross_script"]) == ("Latin", "Indic", "True")
    row = _row(misses, "S1-3", "S2-4")  # nothing in common and not in any pass top-K
    assert row["low_name_and_addr"] == "True" and row["pushed_out_by_cap"] == "False"
    assert set(misses["cause"]) <= {*blocking_misses.CAUSES, "other"}


def test_capped_out_file_is_exact(run: tuple[Paths, pd.DataFrame]) -> None:
    """blocking_capped_out lists true pairs cut by the cap; misses flag exactly those."""
    paths, misses = run
    co = pd.read_parquet(paths.artifacts_dir / "blocking_capped_out_train.parquet")
    want = set(zip(co["s1_id"], co["cand_id"], strict=True))
    got = set(zip(misses["s1_id"], misses["cand_id"], strict=True))
    assert ("S1-1", "S3-1") in want and want <= got
    flagged = misses[misses["pushed_out_by_cap"] == "True"]
    assert set(zip(flagged["s1_id"], flagged["cand_id"], strict=True)) == want


def test_name_sim_matches_tfidf(run: tuple[Paths, pd.DataFrame]) -> None:
    """Hashed cosine equals the pass A TF-IDF cosine fitted on the country's records."""
    paths, misses = run
    rec = pd.read_parquet(paths.artifacts_dir / "records_train.parquet")
    us = rec[rec["country"] == "US"].reset_index(drop=True)
    text = blocking.pass_text(us, blocking.PASSES[0])
    x = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False).fit_transform(text)
    pos = pd.Series(range(len(us)), index=us["entity_id"])
    row = misses[(misses["s1_id"] == "S1-3")].iloc[0]
    want = float((x[pos[row["s1_id"]]] @ x[pos[row["cand_id"]]].T).toarray()[0, 0])
    assert float(row["sim_A"]) == pytest.approx(want, abs=1e-5)


def test_report(run: tuple[Paths, pd.DataFrame]) -> None:
    """Report has the summary sections and exactly n_examples examples."""
    paths, _ = run
    text = (paths.artifacts_dir / "blocking_misses" / "report.txt").read_text(encoding="utf-8")
    for section in ("per country", "per source", "Causes", "Examples"):
        assert section in text
    assert text.count("\n[") == 2


def test_script_of() -> None:
    """Latin / Indic / mixed / other by letter ranges."""
    got = blocking_misses.script_of(pd.Series(["Café", "शर्मा", "sharma शर्मा", "123 !"]))
    assert got.tolist() == ["Latin", "Indic", "mixed", "other"]


def test_missing_candidate_column_raises(tmp_path: Path) -> None:
    """A candidates file without cand_id is rejected, naming it."""
    paths = Paths(tmp_path / "d", tmp_path / "a", tmp_path / "o")
    paths.artifacts_dir.mkdir(parents=True)
    rec = pd.DataFrame({c: ["x"] for c in blocking_misses.RECORD_COLUMNS})
    rec.to_parquet(paths.artifacts_dir / "records_train.parquet", index=False)
    pd.DataFrame({"s1_id": ["x"]}).to_parquet(
        paths.artifacts_dir / "candidates_train.parquet", index=False)
    with pytest.raises(ValueError, match="cand_id"):
        blocking_misses.run(paths, n_examples=1)
