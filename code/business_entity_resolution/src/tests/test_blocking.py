"""Tests for blocking.py on synthetic data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import blocking, config, contracts, normalize
from src.config import Paths

SPEC = blocking.PassSpec("A", "name_norm", "char", 3, top_k=3, max_df=1.0, min_query_terms=1)


def _rand_csr(n: int, v: int, seed: int) -> sp.csr_matrix:
    """Random L2-normalised non-negative float32 CSR rows."""
    m = sp.random(n, v, density=0.2, format="csr", dtype=np.float32, random_state=seed)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / norms) @ m, dtype=np.float32)


def test_top_k_matches_brute_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pruning, top-K equals dense cosine top-K, in tiny chunks too."""
    monkeypatch.setattr(blocking, "BLOCK_MAX_PRODUCT_NNZ", 5)
    monkeypatch.setattr(config, "BLOCK_CHUNK_S1_ROWS", 4)
    q, c = _rand_csr(12, 30, 0), _rand_csr(40, 30, 1)
    qi, ci, sc, rk = blocking.top_k_pairs(q, c, SPEC)
    dense = (q @ c.T).toarray()
    for i in range(12):
        got = ci[qi == i]
        want = [j for j in np.lexsort((np.arange(40), -dense[i]))[:3] if dense[i, j] > 0]
        assert list(got) == want
        assert list(rk[qi == i]) == list(range(1, len(want) + 1))
        np.testing.assert_allclose(sc[qi == i], dense[i, want], rtol=1e-5)
    assert sc.dtype == np.float32 and qi.dtype == np.int32


def test_pruned_scores_are_exact_cosines() -> None:
    """With pruning the returned scores are still the full-vector cosines."""
    q, c = _rand_csr(10, 30, 2), _rand_csr(50, 30, 3)
    qi, ci, sc, _ = blocking.top_k_pairs(q, c, replace(SPEC, max_df=0.05, min_query_terms=2))
    dense = (q @ c.T).toarray()
    np.testing.assert_allclose(sc, dense[qi, ci], rtol=1e-5)
    assert len(set(qi)) == 10  # every query keeps its rarest terms -> gets candidates


def test_prune_query_keeps_rarest_terms() -> None:
    """Common columns are dropped except each row's min_terms rarest."""
    q = sp.csr_matrix(np.array([[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.float32))
    df = np.array([100, 50, 1, 7], dtype=np.int64)
    p = blocking.prune_query(q, df, max_df_count=10, min_terms=2).toarray()
    assert p.tolist() == [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    p1 = blocking.prune_query(q, df, max_df_count=10, min_terms=0).toarray()
    assert p1.tolist() == [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]


def test_union_passes() -> None:
    """Union keeps per-pass score/rank, counts passes and ranks by best score."""
    a = blocking.PassResult(np.array([0, 0], np.int32), np.array([5, 6], np.int32),
                            np.array([0.9, 0.2], np.float32), np.array([1, 2], np.int32))
    b = blocking.PassResult(np.array([0, 1], np.int32), np.array([6, 5], np.int32),
                            np.array([0.95, 0.4], np.float32), np.array([1, 1], np.int32))
    u = blocking.union_passes({"A": a, "B": b}, n_records=10)
    assert u[["s1", "cand"]].values.tolist() == [[0, 6], [0, 5], [1, 5]]
    assert u["n_passes"].tolist() == [2, 1, 1]
    np.testing.assert_allclose(u["best_block_score"], [0.95, 0.9, 0.4])
    assert u["union_rank"].tolist() == [1, 2, 1]
    assert np.isnan(u["pass_B_score"].iloc[1]) and u["pass_A_rank"].iloc[0] == 2


def test_union_rank_uses_rank_fusion() -> None:
    """A pass's rank-1 pair outranks a higher raw score ranked 5th elsewhere."""
    a = blocking.PassResult(np.array([0], np.int32), np.array([1], np.int32),
                            np.array([0.3], np.float32), np.array([1], np.int32))
    b = blocking.PassResult(np.array([0], np.int32), np.array([2], np.int32),
                            np.array([0.9], np.float32), np.array([5], np.int32))
    u = blocking.union_passes({"A": a, "B": b}, n_records=10)
    assert u["cand"].tolist() == [1, 2] and u["union_rank"].tolist() == [1, 2]


def test_reverse_features() -> None:
    """rev_n_s1 / rev_rank / rev_gap per candidate over all S1."""
    s1 = np.array([0, 1, 2, 0], np.int32)
    cand = np.array([7, 7, 7, 8], np.int32)
    score = np.array([0.5, 0.9, 0.5, 0.3], np.float32)
    n, r, g = blocking.reverse_features(s1, cand, score)
    assert n.tolist() == [3, 3, 3, 1]
    assert r.tolist() == [2, 1, 3, 1]
    np.testing.assert_allclose(g, [0.4, 0.0, 0.4, 0.0], rtol=1e-6)


def _write_synthetic(tmp: Path) -> Paths:
    """Records parquet + ground truth for two countries; returns run paths."""
    rows = [
        ("S1-1", "US", "acme widgets inc", "12 main st springfield"),
        ("S1-2", "US", "blue river cafe", "400 oak ave portland"),
        ("S1-3", "US", "zyx lonely name", "9 nowhere rd"),
        ("S2-1", "US", "acme widgets", "12 main street springfield"),
        ("S3-1", "US", "acme widget incorporated", ""),
        ("S2-2", "US", "blue river coffee", "400 oak avenue portland"),
        ("S3-2", "US", "red mountain bakery", "77 pine rd denver"),
        ("S1-4", "India", "sharma traders", "shop 5 mg road pune 411001"),
        ("S2-3", "India", "sharma traders pvt ltd", "mg road pune 411001"),
        ("S3-3", "India", "acme widgets inc", "12 main st springfield"),
    ]
    src = pd.DataFrame(rows, columns=["entity_id", "country", "business_name", "business_address"])
    recs = pd.concat(
        [normalize.build_records(g[["entity_id", "business_name", "business_address", "country"]], s)
         for s, g in src.groupby(src["entity_id"].str[:2])],
        ignore_index=True,
    )
    paths = Paths(tmp / "data", tmp / "art", tmp / "out")
    paths.artifacts_dir.mkdir(parents=True)
    recs.to_parquet(paths.artifacts_dir / "records_train.parquet", index=False)
    (paths.data_dir / "train").mkdir(parents=True)
    (paths.data_dir / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-2\nS1-3\t\nS1-4\tS2-3\n",
        encoding="utf-8", newline="\n",
    )
    return paths


def test_run_stage_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract columns, same-country candidates, cap, recall report, log row."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    monkeypatch.setattr(config, "MAX_CANDIDATES_PER_S1", 2)
    monkeypatch.setattr(config, "BLOCK_CHUNK_S1_ROWS", 2)
    paths = _write_synthetic(tmp_path)
    blocking.run_stage(paths, "train")

    out = pd.read_parquet(paths.artifacts_dir / "candidates_train.parquet")
    assert set(contracts.CANDIDATES_COLUMNS) <= set(out.columns)
    assert not out.duplicated(["s1_id", "cand_id"]).any()
    assert out.groupby("s1_id").size().max() <= 2
    assert out["country_match"].all()
    assert out["cand_id"].str.match(r"S[23]-").all() and out["s1_id"].str.startswith("S1-").all()
    assert (out["cand_source"] == out["cand_id"].str[:2]).all()
    assert out["pass_C_score"].isna().all() and out["pass_F_rank"].isna().all()
    assert out["best_block_score"].dtype == np.float32
    # S3-3 (India) has the same name as S1-1 (US) but must never be its candidate.
    assert not ((out["s1_id"] == "S1-1") & (out["cand_id"] == "S3-3")).any()
    assert {("S1-1", "S2-1"), ("S1-2", "S2-2"), ("S1-4", "S2-3")} <= set(zip(out["s1_id"], out["cand_id"]))
    top = out[out["rev_rank"] == 1].groupby("cand_id")["rev_gap"].max()
    assert (top == 0).all()

    rep = pd.read_csv(paths.artifacts_dir / "blocking_recall_train.tsv", sep="\t")
    assert {"pass_A", "pass_B", "union", "union@20", "union@80"} <= set(rep["variant"])
    assert set(rep["country"]) == {"ALL", "US", "India"}
    union = rep[(rep["country"] == "ALL") & (rep["variant"] == "union")].iloc[0]
    assert union["n_true_pairs"] == 4 and union["n_s1"] == 4
    assert (tmp_path / "shared" / "experiments.tsv").exists()
    assert not (paths.artifacts_dir / "candidates_train.parts.tmp").exists()


def test_run_stage_missing_column_raises(tmp_path: Path) -> None:
    """A records file without a pass column is rejected, naming the column."""
    paths = Paths(tmp_path / "d", tmp_path / "a", tmp_path / "o")
    paths.artifacts_dir.mkdir(parents=True)
    pd.DataFrame({"entity_id": ["S1-1"], "source": ["S1"], "country": ["US"], "name_norm": ["x"]}).to_parquet(
        paths.artifacts_dir / "records_test.parquet", index=False)
    with pytest.raises(ValueError, match="addr_norm"):
        blocking.run_stage(paths, "test")
