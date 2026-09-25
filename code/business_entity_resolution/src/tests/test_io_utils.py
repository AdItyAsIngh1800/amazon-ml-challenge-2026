"""Tests for src.io_utils using small temporary files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import io_utils

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path: Path, text: str) -> Path:
    """Write ``text`` as UTF-8 bytes with no newline translation."""
    path.write_bytes(text.encode("utf-8"))
    return path


def test_read_source_keeps_quotes_na_and_empty(tmp_path: Path) -> None:
    """Quotes stay literal, "NA" stays a string, empty fields are ''."""
    f = _write(
        tmp_path / "s.tsv",
        HEADER
        + 'S1-1\t"""ehpad Club SAS"\t12 "Main" St, Lille\tFrance\n'
        + "S1-2\tNA\t\tIndia\n"
        + "S1-3\tCafé Ünï\tRue d'Été\tFrance\n",
    )
    df = io_utils.read_source(f)
    assert df.shape == (3, 4)
    assert list(df.columns) == list(io_utils.SOURCE_COLUMNS)
    assert df.loc[0, "business_name"] == '"""ehpad Club SAS"'
    assert df.loc[0, "business_address"] == '12 "Main" St, Lille'
    assert df.loc[1, "business_name"] == "NA"
    assert df.loc[1, "business_address"] == ""
    assert df.loc[2, "business_name"] == "Café Ünï"
    assert not df.isna().any().any()
    assert all(pd.api.types.is_string_dtype(df[c]) for c in df.columns)


def test_read_source_without_trailing_newline(tmp_path: Path) -> None:
    """A last line without '\\n' still counts as a row."""
    f = _write(tmp_path / "s.tsv", HEADER + "S2-1\tA\tB\tUS\nS2-2\tC\tD\tUS")
    assert len(io_utils.read_source(f)) == 2


def test_read_source_row_count_mismatch_raises(tmp_path: Path) -> None:
    """A blank line is silently skipped by pandas, so the count check fires."""
    f = _write(tmp_path / "s.tsv", HEADER + "S2-1\tA\tB\tUS\n\nS2-2\tC\tD\tUS\n")
    with pytest.raises(ValueError, match="row count"):
        io_utils.read_source(f)


def test_read_source_missing_column_raises(tmp_path: Path) -> None:
    """A header without 'country' names the missing column."""
    f = _write(tmp_path / "s.tsv", "entity_id\tbusiness_name\tbusiness_address\nS1-1\tA\tB\n")
    with pytest.raises(ValueError, match="country"):
        io_utils.read_source(f)


def test_read_ground_truth(tmp_path: Path) -> None:
    """Comma lists become sets; an empty list becomes an empty set."""
    f = _write(
        tmp_path / "gt.tsv",
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-9\nS1-2\t\nS1-3\tS3-4\n",
    )
    assert io_utils.read_ground_truth(f) == {
        "S1-1": {"S2-1", "S3-9"},
        "S1-2": set(),
        "S1-3": {"S3-4"},
    }


def test_write_id_list_tsv_format(tmp_path: Path) -> None:
    """Rows follow s1_ids order, dedupe IDs, empty when none, exact bytes."""
    out = tmp_path / "sub" / "m.tsv"
    io_utils.write_id_list_tsv(
        {"S1-2": ["S2-5", "S3-1", "S2-5"], "S1-1": set(), "S1-9": ["S2-0"]},
        ["S1-2", "S1-1", "S1-3"],
        out,
        io_utils.MATCHING_HEADER,
    )
    assert out.read_bytes() == (
        b"source1_entity_id\tmatched_entity_ids\n"
        b"S1-2\tS2-5,S3-1\n"
        b"S1-1\t\n"
        b"S1-3\t\n"
    )


def test_write_id_list_tsv_duplicate_s1_raises(tmp_path: Path) -> None:
    """Duplicate S1 rows would be rejected by the portal."""
    with pytest.raises(ValueError, match="duplicate"):
        io_utils.write_id_list_tsv({}, ["S1-1", "S1-1"], tmp_path / "m.tsv", io_utils.MATCHING_HEADER)


def test_write_then_read_ground_truth_roundtrip(tmp_path: Path) -> None:
    """Writer output parses back into the same mapping."""
    mapping = {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()}
    out = tmp_path / "gt.tsv"
    io_utils.write_id_list_tsv(mapping, ["S1-1", "S1-2"], out, io_utils.MATCHING_HEADER)
    assert io_utils.read_ground_truth(out) == mapping


def test_parquet_roundtrip_with_columns(tmp_path: Path) -> None:
    """Save a column subset, then load a smaller subset."""
    df = pd.DataFrame({"a": ["x", "NA"], "b": [1.5, 2.5], "c": [[1], []]})
    out = tmp_path / "t.parquet"
    io_utils.save_parquet(df, out, columns=["a", "c"])
    assert list(io_utils.load_parquet(out).columns) == ["a", "c"]
    back = io_utils.load_parquet(out, columns=["a"])
    assert back["a"].tolist() == ["x", "NA"]


def test_parquet_missing_columns_raise(tmp_path: Path) -> None:
    """Missing columns are named in a ValueError on save and on load."""
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "t.parquet"
    with pytest.raises(ValueError, match="zzz"):
        io_utils.save_parquet(df, out, columns=["a", "zzz"])
    io_utils.save_parquet(df, out)
    with pytest.raises(ValueError, match="zzz"):
        io_utils.load_parquet(out, columns=["zzz"])


def test_write_source_tsv_roundtrip(tmp_path: Path) -> None:
    """Quotes, "NA" and empty fields survive write -> read_source unchanged."""
    df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ['"""ehpad Club SAS"', "NA"],
        "business_address": ['12 "Main" St', ""],
        "country": ["France", "US"],
    })
    out = tmp_path / "d" / "s.tsv"
    io_utils.write_source_tsv(df, out)
    assert out.read_bytes().startswith(HEADER.encode("utf-8"))
    pd.testing.assert_frame_equal(io_utils.read_source(out), df, check_dtype=False)


def test_write_source_tsv_rejects_tab_in_field(tmp_path: Path) -> None:
    """A tab inside a value would corrupt the file."""
    df = pd.DataFrame({"entity_id": ["S2-1"], "business_name": ["a\tb"],
                       "business_address": [""], "country": ["US"]})
    with pytest.raises(ValueError, match="business_name"):
        io_utils.write_source_tsv(df, tmp_path / "s.tsv")


def test_read_ground_truth_pairs(tmp_path: Path) -> None:
    """One row per true pair; a singleton S1 keeps one row with cand_id ''."""
    f = _write(
        tmp_path / "gt.tsv",
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-9\nS1-2\t\nS1-3\tS3-4\n",
    )
    df = io_utils.read_ground_truth_pairs(f)
    assert list(df.columns) == ["s1_id", "cand_id"]
    assert df.values.tolist() == [["S1-1", "S2-1"], ["S1-1", "S3-9"], ["S1-2", ""], ["S1-3", "S3-4"]]
    assert all(pd.api.types.is_string_dtype(df[c]) for c in df.columns)
