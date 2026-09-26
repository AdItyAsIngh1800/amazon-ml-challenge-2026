"""Tests for normalize.py prep v0."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import contracts, io_utils, normalize
from src.config import Paths


def _norm(*values: str) -> list[str]:
    """normalize_text on plain strings."""
    return normalize.normalize_text(pd.Series(list(values), dtype="str")).tolist()


def test_normalize_text_rules() -> None:
    """Lowercase, Latin accent strip, & -> and, punctuation -> space, collapse."""
    assert _norm("Café & Bar, Inc.", '"""ehpad Club SAS"', "  B+  Retail\tInc ", "--", "Ünï_Co") == [
        "cafe and bar inc", "ehpad club sas", "b retail inc", "", "uni co",
    ]


def test_normalize_text_keeps_devanagari_marks() -> None:
    """Virama/nukta and vowel signs survive (only U+0300-U+036F is stripped)."""
    hindi = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert _norm(hindi, "सड़क") == [hindi, unicodedata.normalize("NFKD", "सड़क")]


def test_address_tokens_and_landmark() -> None:
    """Postal = standalone 5-6 digits; nums = every standalone number."""
    addr = normalize.normalize_text(pd.Series([
        "Near SBI ATM, Bangalore-560001",
        "105 ELM ST, 12345-6789",
        "KH NO. -570/13, PH 1234567",
        "Nearby Mall, Opp. Station",
        "Behindthe shop",
    ], dtype="str"))
    assert normalize.postal_tokens(addr).tolist() == [["560001"], ["12345"], [], [], []]
    assert normalize.num_tokens(addr).tolist() == [["560001"], ["105", "12345", "6789"], ["570", "13", "1234567"], [], []]
    assert normalize.landmark_flag(addr).tolist() == [True, False, False, True, False]


def test_acronym() -> None:
    """First letter of each name_norm token."""
    assert normalize.acronym(pd.Series(["b retail inc", "", "राम मार्केटिंग"], dtype="str")).tolist() == ["bri", "", "रम"]


def _write_split(root: Path, split: str) -> None:
    """Tiny 3-source split with an empty address and a punctuation-only name."""
    cols = list(io_utils.SOURCE_COLUMNS)
    rows = {
        1: [("S1-1", "B+ Retail Inc", "12 Main St, Phoenix, AZ 85001", "US")],
        2: [("S2-1", "--", "", "India"), ("S2-2", "Café Lune SARL", "Opp. Gare, 75001 Paris", "France")],
        3: [("S3-1", "NA", "near Temple Road 560001", "India"),
            ("S3-2", "श्री राम ट्रेडर्स", "मुंबई ४००००१", "India")],
    }
    for s, r in rows.items():
        io_utils.write_source_tsv(pd.DataFrame(r, columns=cols), root / split / f"{split}_source{s}.tsv")


def test_run_stage_writes_contract(tmp_path: Path) -> None:
    """records_{split}.parquet has every contract column with the right types."""
    paths = Paths(tmp_path / "data", tmp_path / "art", tmp_path / "out")
    _write_split(paths.data_dir, "test")
    normalize.run_stage(paths, "test")

    out = paths.artifacts_dir / "records_test.parquet"
    schema = pq.read_schema(out)
    assert tuple(schema.names) == contracts.RECORDS_COLUMNS
    df = io_utils.load_parquet(out)
    assert df["entity_id"].tolist() == ["S1-1", "S2-1", "S2-2", "S3-1", "S3-2"]
    assert df["source"].tolist() == ["S1", "S2", "S2", "S3", "S3"]
    r = df.set_index("entity_id")
    assert r.loc["S2-1", "name_empty"] and r.loc["S2-1", "addr_empty"]
    assert r.loc["S3-1", "name_norm"] == "na" and bool(r.loc["S3-1", "landmark_flag"])
    assert df.loc[df["entity_id"] == "S2-2", "postal_tokens"].map(list).tolist() == [["75001"]]
    assert r.loc["S2-2", "name_core"] == "cafe lune" and r.loc["S2-2", "legal_suffix"] == "sarl"
    assert r.loc["S1-1", "name_acronym"] == "br"  # from name_core "b retail"
    # transliterated before normalisation; Indic digits feed postal tokens
    assert r.loc["S3-2", "name_translit"] == "shree raam tredars"
    assert r.loc["S3-2", "name_norm"] == "shree raam tredars"
    assert r.loc["S3-2", "name_key"] == "sr rm trdrs"
    assert r.loc["S3-2", "addr_translit"] == "mumbaee 400001"
    assert df.loc[df["entity_id"] == "S3-2", "postal_tokens"].map(list).tolist() == [["400001"]]
    assert r.loc["S1-1", "name_translit"] == "B+ Retail Inc" and r.loc["S1-1", "name_key"] == "b rtl"
    assert r.loc["S1-1", "legal_suffix"] == "inc" and r.loc["S3-2", "name_core"] == "sri raam tredars"
    assert schema.equals(normalize.RECORDS_SCHEMA)
    assert schema.field("postal_tokens").type == pa.list_(pa.string())
    assert schema.field("landmark_flag").type == pa.bool_()
    assert not (paths.artifacts_dir / "records_test.parquet.tmp").exists()


def test_bad_id_prefix_raises(tmp_path: Path) -> None:
    """An ID whose prefix does not match its source file is rejected."""
    paths = Paths(tmp_path / "data", tmp_path / "art", tmp_path / "out")
    _write_split(paths.data_dir, "train")
    bad = pd.DataFrame([("S9-1", "x", "y", "US")], columns=list(io_utils.SOURCE_COLUMNS))
    io_utils.write_source_tsv(bad, paths.data_dir / "train" / "train_source3.tsv")
    with pytest.raises(ValueError, match="S9-1"):
        normalize.run_stage(paths, "train")


@pytest.mark.parametrize(("name", "core", "suffix"), [
    ("silver trading pvt ltd", "silver trading", "private limited"),
    ("sky estate praa li", "sky estate", "private limited"),  # Gujarati Pvt. Ltd.
    ("abc praaivet limitet", "abc", "private limited"),  # transliterated forms
    ("l l p creative engineering", "creative engineering", "llp"),  # dotted + leading
    ("limited yamutech sciences center", "yamutech sciences center", "limited"),
    ("bansal traders llp center", "bansal traders center", "llp"),  # noise after legal
    ("smt pamtl holdings", "pamtl holdings", ""),  # honorific
    ("m s shree ram traders", "sri ram traders", ""),  # M/s + shree -> sri
    ("heartcenter com", "heartcenter", ""),
    ("xyz public limited", "xyz", "public limited"),
    ("abc limited limited", "abc", "limited"),
    ("abc and co", "abc", "co"),
    ("club de foot sas", "club de foot", "sas"),
    ("cafe et cie", "cafe", "cie"),
    ("delhi public school", "delhi public school", ""),  # legal word mid-name kept
    ("private limited", "private limited", ""),  # nothing left -> keep all
])
def test_split_legal(name: str, core: str, suffix: str) -> None:
    """Legal suffix from the trailing (then leading) legal run; core is the rest."""
    c, s = normalize.split_legal(pd.Series([name], dtype="str"))
    assert (c.iloc[0], s.iloc[0]) == (core, suffix)


def test_canonical_address() -> None:
    """Long forms -> short; multi-word state names first."""
    got = normalize.canonical_address(pd.Series([
        "12 main street near temple road uttar pradesh",
        "5 rue de la paix", "west virginia 25301", "2803 la loma drive rancho cordova california",
    ], dtype="str")).tolist()
    assert got == ["12 main st nr temple rd up", "5 r de la paix", "wv 25301", "2803 la loma dr rancho cordova ca"]


def test_postal_pairs_joined_only_on_request() -> None:
    """E7 pattern b ("600 001") is joined in prep; the v0 rule is unchanged."""
    addr = pd.Series(["chennai 600 001", "560001 560001", "no pin"], dtype="str")
    assert normalize.postal_tokens(addr).tolist() == [[], ["560001", "560001"], []]
    assert normalize.postal_tokens(addr, join_pairs=True).tolist() == [["600001"], ["560001", "560001"], []]
