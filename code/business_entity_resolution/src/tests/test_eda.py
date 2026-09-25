"""Tests for eda.py metric helpers and an end-to-end run on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src import eda, io_utils


def test_bucket_counts() -> None:
    """0..4 exact, 5+ pooled, as percentages."""
    got = eda.bucket_counts(pd.Series([0, 1, 1, 2, 5, 9, 4, 3]))
    assert got == {"0": 12.5, "1": 25.0, "2": 12.5, "3": 12.5, "4": 12.5, "5+": 25.0}


def test_postal_info_patterns() -> None:
    """(a) standalone 5-6 digits after v0 normalisation, (b) '600 001', (c) ZIP+4."""
    addr = pd.Series([
        "MG Road, Chennai 600 001",
        "105 Elm St, Morganton, NC 28655-1234",
        "Bangalore-560001",
        "Plot 1234567, no pin",
        "",
    ], dtype="str")
    info = eda.postal_info(addr)
    assert info["has_a"].tolist() == [False, True, True, False, False]
    assert info["has_b"].tolist() == [True, False, False, False, False]
    assert info["has_c"].tolist() == [False, True, False, False, False]
    assert info["tokens"].tolist() == ["600001", "28655", "560001", "", ""]


def test_name_key() -> None:
    """Lowercase, keep only letters, marks and digits (any script)."""
    got = eda.name_key(pd.Series(["B+ Retail, Inc.", "Café-24", "राम मार्केटिंग"], dtype="str"))
    assert got.tolist() == ["bretailinc", "café24", "राममार्केटिंग"]


def test_name_vectorizer_matches_sklearn_tfidf() -> None:
    """Streaming hashed TF-IDF gives the same cosines as TfidfVectorizer."""
    docs = ["acme corp", "acme corporation", "zenith labs", "zenith lab inc", "aaaa aaaa acme", "raj traders"]
    vec = eda.NameVectorizer()
    vec.fit(pd.Series(docs[:3], dtype="str"))
    vec.fit(pd.Series(docs[3:], dtype="str"))  # fit accumulates over chunks
    ours = vec.transform(pd.Series(docs, dtype="str"))
    ref = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False).fit_transform(docs)
    np.testing.assert_allclose((ours @ ours.T).toarray(), (ref @ ref.T).toarray(), atol=1e-5)


def test_true_match_ranks() -> None:
    """Rank = 1 + #scores strictly greater; zero cosine or missing column -> inf."""
    q = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    x = np.array([[0.9, 0.0], [0.5, 0.5], [0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
    xt = sp.csr_matrix(x.T)
    ranks = eda.true_match_ranks(q, xt, [np.array([1, 3]), np.array([2, -1])])
    # row 0 scores [0.9, 0.5, 0.1, 0]: col1 -> rank 2, col3 (0) -> inf
    # row 1 scores [0, 0.5, 0, 0.2]: col2 (0) -> inf, -1 (cross-country) -> inf
    assert ranks.tolist() == [2.0, np.inf, np.inf, np.inf]


def test_recall_at_k() -> None:
    """Share of ranks <= K."""
    got = eda.recall_at_k(np.array([1, 3, 7, np.inf]), (1, 5, 10))
    assert got == {1: 0.25, 5: 0.5, 10: 0.75}


def _write_split(root: Path, split: str, n: int, countries: tuple[str, ...]) -> None:
    """Synthetic split: S1 i matches S2 i and S3 i (same name) for odd i."""
    cols = list(io_utils.SOURCE_COLUMNS)
    rows: dict[int, list[tuple[str, str, str, str]]] = {1: [], 2: [], 3: []}
    gt: dict[str, set[str]] = {}
    for i in range(n):
        c = countries[i % len(countries)]
        name, addr = f"Shop {i} Traders Pvt Ltd", f"{i} Main Road Near Bus Stand 5600{i % 10:02d}"
        rows[1].append((f"S1-{i}", name, addr, c))
        rows[2].append((f"S2-{i}", name.upper(), addr, c))
        rows[3].append((f"S3-{i}", f"Shop {i} Traders", "", c))
        gt[f"S1-{i}"] = {f"S2-{i}", f"S3-{i}"} if i % 2 else set()
    for s, r in rows.items():
        io_utils.write_source_tsv(pd.DataFrame(r, columns=cols), root / split / f"{split}_source{s}.tsv")
    if split == "train":
        io_utils.write_id_list_tsv(gt, list(gt), root / "train" / "train_ground_truth.tsv",
                                   io_utils.GROUND_TRUTH_COLUMNS)


@pytest.mark.parametrize("with_test", [True, False])
def test_run_eda_end_to_end(tmp_path: Path, with_test: bool) -> None:
    """Report contains every section; missing test split is skipped, not fatal."""
    _write_split(tmp_path / "data", "train", 40, ("US", "India"))
    if with_test:
        _write_split(tmp_path / "data", "test", 30, ("US", "India", "France"))
    report = eda.run_eda(tmp_path / "data", tmp_path / "art", n_quantile_pairs=10, n_rank_s1=8)
    text = (tmp_path / "art" / "eda" / "eda_report.txt").read_text(encoding="utf-8")
    assert report == text
    for sec in [f"E{i}" for i in range(1, 14)] + ["IMPLICATIONS"]:
        assert f"=== {sec}" in text
    assert "one-owner rule" in text
    assert ("France" in text) == with_test
