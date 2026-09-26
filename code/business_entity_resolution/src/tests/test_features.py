"""Tests for features.py on synthetic records and candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import blocking, config, features, normalize
from src.config import Paths
from src.tests.test_blocking import _write_synthetic


def _records(rows: list[tuple[str, str, str]]) -> pa.Table:
    """records columns for (entity_id, name, address) rows via normalize.build_records."""
    src = pd.DataFrame([(e, n, a, "US") for e, n, a in rows],
                       columns=["entity_id", "business_name", "business_address", "country"])
    rec = pd.concat([normalize.build_records(g, s) for s, g in src.groupby(src["entity_id"].str[:2])],
                    ignore_index=True).set_index("entity_id").loc[[r[0] for r in rows]].reset_index()
    return pa.Table.from_pandas(rec[list(features.RECORD_COLUMNS)], preserve_index=False)


def test_pair_features_flags() -> None:
    """Acronym, legal suffix, postal and address flags on hand-made pairs."""
    rec = _records([
        ("S1-1", "International Business Machines Inc", "1 Main St 411001"),
        ("S2-1", "IBM LLC", "1 Main Street 411001"),
        ("S2-2", "International Business Machines Inc", "9 Elm St 560001"),
        ("S2-3", "International Business Machines", ""),
    ])
    idf = features.fit_idf(rec.column("name_norm"), "word", 1)
    left, right = np.zeros(3, np.intp), np.array([1, 2, 3], np.intp)
    f = features.pair_features(rec, left, right, idf, idf, features.fit_idf(rec.column("addr_norm"), "word", 1))
    assert f["acr_s1_is_cand"].tolist() == [1, 0, 0]
    assert f["sfx_equal"].tolist() == [0, 1, 0]
    assert f["sfx_conflict"].tolist() == [1, 0, 0]
    assert f["sfx_missing"].tolist() == [0, 0, 1]
    assert f["postal_match"].tolist() == [1, 0, 0]
    assert f["postal_conflict"].tolist() == [0, 1, 0]
    assert f["postal_missing"].tolist() == [0, 0, 1]
    assert f["name_core_ratio"][1] == 1.0
    assert np.isnan(f["addr_norm_token_set"][2])  # empty address -> undefined, not 0 or 1
    assert f["cand_addr_empty"].tolist() == [0, 0, 1]
    assert all(v.dtype == np.float32 and len(v) == 3 for v in f.values())


def test_idf_similarities() -> None:
    """Identical texts score 1, disjoint texts 0; a shared rare token outweighs a common one."""
    texts = pa.array(["acme corp", "acme corp", "zeta labs", "beta corp", "gamma corp"])
    h = features.fit_idf(texts, "word", 1)
    t = texts.to_pylist()
    cos = features.cosine(h, t, np.array([0, 0], np.intp), np.array([1, 2], np.intp))
    np.testing.assert_allclose(cos, [1.0, 0.0], atol=1e-6)
    w, j = features.idf_overlap(h, t, np.array([0, 3], np.intp), np.array([1, 4], np.intp))
    assert j.tolist() == [1.0, pytest.approx(1 / 3)]
    assert w[1] < j[1]  # "corp" is common, so its weighted share is below the plain Jaccard


def test_group_features() -> None:
    """Rank 1 = best, gap to best, count and z-score per S1."""
    s1 = np.array([7, 7, 7, 9], np.int32)
    feats = {c: np.array([0.2, 0.9, 0.5, 0.4], np.float32) for c in features.GROUP_SOURCES}
    g = features.group_features(s1, feats)
    assert g["grp_n_cands"].tolist() == [3, 3, 3, 1]
    assert g["grp_name_char3_cos_rank"].tolist() == [3, 1, 2, 1]
    np.testing.assert_allclose(g["grp_addr_tok_cos_gap"], [0.7, 0.0, 0.4, 0.0], atol=1e-6)
    assert g["grp_addr_tok_cos_z"][3] == 0.0 and g["grp_addr_tok_cos_z"][1] > 0


def test_s1_chunks_never_split_a_group(tmp_path: Path) -> None:
    """Chunks cut only between S1 groups, and together cover every row in order."""
    s1 = ["S1-1"] * 3 + ["S1-2"] * 1 + ["S1-3"] * 4 + ["S1-4"] * 2
    path = tmp_path / "c.parquet"
    pq.write_table(pa.table({"s1_id": s1, "x": list(range(len(s1)))}), path)
    chunks = list(features.s1_chunks(path, ["s1_id", "x"], 2))
    seen = [c.column("s1_id").to_pylist() for c in chunks]
    assert sum(seen, []) == s1
    for a, b in zip(seen, seen[1:], strict=False):
        assert a[-1] != b[0]


def test_run_stage_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """features_train parts: one row per candidate in order, float32, labels from ground truth."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    monkeypatch.setattr(config, "FEATURE_CHUNK_PAIRS", 2)
    paths = _write_synthetic(tmp_path)
    blocking.run_stage(paths, "train")
    features.run_stage(paths, "train")

    cand = pd.read_parquet(paths.artifacts_dir / "candidates_train.parquet")
    out = pd.read_parquet(paths.artifacts_dir / "features_train")
    assert len(list((paths.artifacts_dir / "features_train").glob("part-*.parquet"))) > 1
    assert not (paths.artifacts_dir / "features_train.tmp").exists()
    assert out[["s1_id", "cand_id"]].equals(cand[["s1_id", "cand_id"]])
    feat_cols = [c for c in out.columns if c not in ("s1_id", "cand_id", "label")]
    assert (out[feat_cols].dtypes == np.float32).all()
    assert not any("country" in c and c != "country_match" for c in feat_cols)
    truth = {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-4", "S2-3")}
    expected = [int(p in truth) for p in zip(out["s1_id"], out["cand_id"], strict=True)]
    assert out["label"].tolist() == expected and sum(expected) >= 3
    assert (out.groupby("s1_id")["grp_n_cands"].agg(["min", "size"]).pipe(lambda d: d["min"] == d["size"])).all()


def test_run_stage_rejects_split_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An S1 whose candidates are not contiguous is rejected (group features would be wrong)."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    paths = _write_synthetic(tmp_path)
    blocking.run_stage(paths, "train")
    path = paths.artifacts_dir / "candidates_train.parquet"
    cand = pd.read_parquet(path)
    first = cand["s1_id"] == cand["s1_id"].value_counts().idxmax()
    assert first.sum() > 1
    cand = pd.concat([cand[first].iloc[:1], cand[~first], cand[first].iloc[1:]], ignore_index=True)
    cand.to_parquet(path, index=False)
    monkeypatch.setattr(config, "FEATURE_CHUNK_PAIRS", 1)
    with pytest.raises(ValueError, match="contiguous"):
        features.run_stage(paths, "train")


def test_run_stage_missing_column_raises(tmp_path: Path) -> None:
    """A records file without name_key is rejected, naming the column."""
    paths = Paths(tmp_path / "d", tmp_path / "a", tmp_path / "o")
    paths.artifacts_dir.mkdir(parents=True)
    rec = _records([("S1-1", "Acme", "1 Main St")]).drop_columns(["name_key"])
    pq.write_table(rec, paths.artifacts_dir / "records_test.parquet")
    with pytest.raises(ValueError, match="name_key"):
        features.run_stage(paths, "test")
