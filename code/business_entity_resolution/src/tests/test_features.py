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

# Column order of features_{split} parts with FEATURE_SET v1 (the submitted M2 pipeline).
V1_COLUMNS: list[str] = [
    's1_id', 'cand_id', 'name_core_ratio', 'name_core_token_set', 'name_core_token_sort', 'name_core_partial',
    'name_core_jw', 'name_core_lev', 'name_norm_ratio', 'name_norm_token_set', 'name_norm_token_sort', 'name_norm_partial',
    'name_norm_jw', 'name_norm_lev', 'name_key_ratio', 'name_translit_ratio', 'name_translit_token_set', 'name_char3_cos',
    'name_tok_idf_jacc', 'name_tok_jacc', 'acr_s1_is_cand', 'acr_cand_is_s1', 'sfx_equal', 'sfx_conflict',
    'sfx_missing', 'name_len_ratio', 'first_tok_match', 'name_digit_jacc', 'name_digit_one_side', 'postal_match',
    'postal_conflict', 'postal_missing', 'num_overlap', 'num_conflict', 'addr_tok_cos', 'addr_tok_idf_jacc',
    'addr_tok_jacc', 'addr_norm_token_set', 'addr_translit_token_set', 's1_landmark_flag', 'cand_landmark_flag', 's1_name_empty',
    'cand_name_empty', 's1_addr_empty', 'cand_addr_empty', 'grp_n_cands', 'grp_name_char3_cos_rank', 'grp_name_char3_cos_gap',
    'grp_name_char3_cos_z', 'grp_name_core_token_set_rank', 'grp_name_core_token_set_gap', 'grp_name_core_token_set_z', 'grp_addr_tok_cos_rank', 'grp_addr_tok_cos_gap',
    'grp_addr_tok_cos_z', 'country_match', 'pass_A_score', 'pass_A_rank', 'pass_B_score', 'pass_B_rank',
    'pass_C_score', 'pass_C_rank', 'pass_F_score', 'pass_F_rank', 'n_passes', 'best_block_score',
    'rrf_score', 'rev_n_s1', 'rev_rank', 'rev_gap', 'cand_is_s3', 'label',
]


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


def _v2(rows: list[tuple[str, str, str]], s1: list[int] | None = None) -> dict[str, np.ndarray]:
    """v2_features of pairs (row 0, row i) for i >= 1; S1 code 0 for every pair unless given."""
    rec = _records(rows)
    n = len(rows) - 1
    left, right = np.zeros(n, np.intp), np.arange(1, n + 1, dtype=np.intp)
    idf = features.fit_idf(rec.column("name_norm"), "char_wb", 3)
    base = {"name_char3_cos": features.cosine(idf, rec.column("name_norm").to_pylist(), left, right)}
    codes = np.array(s1 if s1 is not None else [0] * n, np.int32)
    out = features.v2_features(rec, left, right, codes, base)
    assert all(v.dtype == np.float32 and len(v) == n for v in out.values())
    return out


def test_v2_unit_and_number_features() -> None:
    """A402 vs A407 in one building is a unit conflict; same unit equal; unit on one side only."""
    f = _v2([
        ("S1-1", "Acme", "Flat A-402, Tower 3, MG Road 411001"),
        ("S2-1", "Acme", "A407 Tower 3 MG Road 411001"),
        ("S2-2", "Acme", "A402 Tower 3 MG Road 411001"),
        ("S2-3", "Acme", "Tower 3 MG Road 411001"),
        ("S2-4", "Acme", "Tower 9 MG Road 560001"),
    ])
    assert f["unit_equal"].tolist() == [0, 1, 0, 0]
    assert f["unit_conflict"].tolist() == [1, 0, 0, 0]
    assert f["unit_missing_one"].tolist() == [0, 0, 1, 1]
    # S1 numbers {3, 411001} (402 belongs to unit a402); S2-4 {9, 560001}.
    assert f["num_one_side_n"].tolist() == [0, 0, 0, 4]
    assert features._addr_parts("flat a 402 tower 3 rd 411001") == (frozenset({"a402"}), frozenset({"3", "411001"}), "3")
    assert f["num_disjoint"].tolist() == [0, 0, 0, 1]


def test_v2_exact_address_features() -> None:
    """Exact addr_norm, same token set in another order, house number + postal equal."""
    f = _v2([
        ("S1-1", "Acme", "12 Main St 411001"),
        ("S2-1", "Zeta", "12 Main St 411001"),
        ("S2-2", "Zeta", "Main St 12 411001"),
        ("S2-3", "Zeta", "12 Oak Avenue 411001"),
        ("S2-4", "Zeta", "14 Main St 411001"),
        ("S2-5", "Zeta", ""),
    ])
    assert f["addr_exact"].tolist() == [1, 0, 0, 0, 0]
    assert f["addr_tokset_equal"].tolist() == [1, 1, 0, 0, 0]
    assert f["house_postal_equal"].tolist() == [1, 1, 1, 0, 0]


def test_v2_name_features() -> None:
    """Legal suffix and word order do not matter; containment; suffix-free token_set."""
    f = _v2([
        ("S1-1", "Acme Industries Pvt. Ltd.", "x"),
        ("S2-1", "Industries Acme Inc", "x"),
        ("S2-2", "Acme Industries Pune", "x"),
        ("S2-3", "Zeta Labs", "x"),
    ])
    assert f["name_core_tokset_equal"].tolist() == [1, 0, 0]
    assert f["name_core_contained"].tolist() == [1, 1, 0]
    assert f["name_nosfx_token_set"][0] == 1.0 and f["name_nosfx_token_set"][2] < 0.5
    assert features._name_no_suffix("Acme Industries Pvt. Ltd.", "acme industries pvt ltd",
                                    "acme industries") == "acme industries"


def test_v2_same_address_context() -> None:
    """Count of this S1's candidates at the same addr_norm and name rank among them; other S1 separate."""
    f = _v2([
        ("S1-1", "Acme Stores", "1 Main St"),
        ("S2-1", "Acme Stores", "5 Oak Rd"),
        ("S2-2", "Zeta", "5 Oak Rd"),
        ("S2-3", "Acme Store", "9 Elm St"),
        ("S2-4", "Acme Stores", "5 Oak Rd"),
        ("S2-5", "Acme", ""),
    ], s1=[0, 0, 0, 1, 0])
    assert f["grp_same_addr_n"].tolist() == [2, 2, 1, 1, 0]
    assert f["grp_same_addr_name_rank"][:4].tolist() == [1, 2, 1, 1]
    assert np.isnan(f["grp_same_addr_name_rank"][4])


def test_feature_set_v1_unchanged_v2_superset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 keeps the exact v1 column list; v2 = the same v1 columns with identical values + the v2 columns."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    paths = _write_synthetic(tmp_path)
    blocking.run_stage(paths, "train")
    features.run_stage(paths, "train")
    v1 = pd.read_parquet(paths.artifacts_dir / "features_train")
    assert list(v1.columns) == V1_COLUMNS
    monkeypatch.setattr(config, "FEATURE_SET", "v2")
    features.run_stage(paths, "train")
    v2 = pd.read_parquet(paths.artifacts_dir / "features_train")
    extra = [c for c in v2.columns if c not in V1_COLUMNS]
    assert len(extra) == 13 and not any("country" in c for c in extra)
    pd.testing.assert_frame_equal(v2[V1_COLUMNS], v1)
    assert (v2[extra].dtypes == np.float32).all()
    monkeypatch.setattr(config, "FEATURE_SET", "v3")
    with pytest.raises(ValueError, match="FEATURE_SET"):
        features.run_stage(paths, "train")
