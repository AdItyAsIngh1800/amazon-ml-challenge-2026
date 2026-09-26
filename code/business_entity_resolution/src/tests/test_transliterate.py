"""Tests for transliterate.py: script mapping, Devanagari -> Latin rules, phonetic key."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from rapidfuzz import fuzz

from src import config, io_utils, normalize, transliterate
from src.transliterate import transliterate_text as tr

# "limited" as written in the nine scripts (E10 top India name tokens).
LIMITED = {
    "Devanagari": ("लिमिटेड", "limited"),
    "Bengali": ("লিমিটেড", "limited"),
    "Gurmukhi": ("ਲਿਮਟਿਡ", "limatid"),
    "Gujarati": ("લિમિટેડ", "limited"),
    "Oriya": ("ଲିମିଟେଡ୍", "limited"),
    "Tamil": ("லிமிடெட்", "limitet"),  # Tamil ட is both t and d
    "Telugu": ("లిమిటెడ్", "limited"),
    "Kannada": ("ಲಿಮಿಟೆಡ್", "limited"),
    "Malayalam": ("ലിമിറ്റഡ്", "limittad"),  # റ്റ = "tt"
}


@pytest.mark.parametrize("script", sorted(LIMITED))
def test_each_script_maps_to_latin(script: str) -> None:
    """Every Brahmic block reaches Latin through the Devanagari offset."""
    src, expected = LIMITED[script]
    assert tr(src) == expected


def test_virama_and_vowel_signs() -> None:
    """Virama removes the inherent a; vowel signs replace it; conjuncts join."""
    assert tr("क्षमा") == "kshamaa"
    assert tr("संतोष") == "santosh"
    assert tr("ज्ञान") == "gyaan"


def test_final_schwa_deletion() -> None:
    """Word-final inherent a is dropped, except after a conjunct ending in r/y/v."""
    assert tr("राम") == "raam"
    assert tr("कमल") == "kamal"
    assert tr("परफेक्ट") == "paraphekt"
    assert tr("मिश्र") == "mishra"
    assert tr("आदित्य") == "aaditya"
    assert tr("राम कमल") == "raam kamal"


def test_signs_and_nukta() -> None:
    """Anusvara n (m before labials / word-final), visarga h, nukta forms."""
    assert tr("मुंबई") == "mumbaee"
    assert tr("കേരളം") == "keralam"
    assert tr("कः") == "kah"
    assert (tr("ज़मीन"), tr("फ़िल्म"), tr("सड़क"), tr("क़ुरैशी")) == ("zameen", "film", "sarak", "quraishee")


def test_non_aligned_code_points() -> None:
    """Chillu, khanda ta, tippi/addak, Tamil ச and joiners are mapped explicitly."""
    assert tr("അവൻ") == "avan"  # chillu n U+0D7B
    assert tr("উৎসব") == "utsab"  # khanda ta U+09CE
    assert tr("ਪੰਜਾਬ") == "panjaab"  # tippi U+0A70
    assert tr("ਇੱਕ") == "ik"  # addak U+0A71
    assert tr("சன்") == "san"
    assert tr("ఎక్స్‌పోర్ట్స్") == "eksports"  # ZWNJ removed


@pytest.mark.parametrize("digits", [
    "०१२३४५६७८९", "০১২৩৪৫৬৭৮৯", "੦੧੨੩੪੫੬੭੮੯", "૦૧૨૩૪૫૬૭૮૯", "୦୧୨୩୪୫୬୭୮୯",
    "௦௧௨௩௪௫௬௭௮௯", "౦౧౨౩౪౫౬౭౮౯", "೦೧೨೩೪೫೬೭೮೯", "൦൧൨൩൪൫൬൭൮൯",
])
def test_indic_digits_become_ascii(digits: str) -> None:
    """Digits of all nine scripts -> ASCII."""
    assert tr(digits) == "0123456789"


def test_latin_passes_through() -> None:
    """Latin, accents and punctuation are untouched; mixed text keeps the Latin part."""
    assert tr("Café & Co. 123") == "Café & Co. 123"
    assert tr("Shree राम Traders") == "Shree raam Traders"
    s = pd.Series(["Plain Name", "राम", "NA"], index=[5, 7, 9], dtype="str")
    out = transliterate.transliterate(s)
    assert out.tolist() == ["Plain Name", "raam", "NA"] and out.index.tolist() == [5, 7, 9]


def test_phonetic_key_rules() -> None:
    """Long vowels, aspirates, w->v, doubles, loanword letters, consonant skeleton."""
    got = transliterate.phonetic_key(pd.Series(
        ["bhaarat", "Shree", "khanna", "wood", "chhaya", "exports", "classic", "food", "business", "silver trading"],
        dtype="str",
    )).tolist()
    assert got == ["brt", "sr", "kn", "vd", "k", "eksprts", "klsk", "pd", "bsns", "slvr trdng"]


# Hand-written cross-script spellings of common India name words (not dataset records).
SYNTHETIC_PAIRS = [
    ("Silver Trading Private Limited", "सिल्वर ट्रेडिंग प्राइवेट लिमिटेड"),
    ("Om Estate Limited", "ಓಂ ಎಸ್ಟೇಟ್ ಲಿಮಿಟೆಡ್"),
    ("Shree Exports", "శ్రీ ఎక్స్‌పోర్ట్స్"),
    ("Golden Investments", "গোল্ডেন ইনভেস্টমেন্টস"),
    ("Balaji Products", "બાલાજી પ્રોડક્ટ્સ"),
]


def _key(names: list[str]) -> list[str]:
    """name_key exactly as prep builds it."""
    s = pd.Series(names, dtype="str")
    return transliterate.phonetic_key(normalize.normalize_text(transliterate.transliterate(s))).tolist()


@pytest.mark.parametrize(("latin", "indic"), SYNTHETIC_PAIRS)
def test_synthetic_cross_script_keys_match(latin: str, indic: str) -> None:
    """Same name in Latin and an Indic script gives a near-equal key."""
    a, b = _key([latin, indic])
    assert fuzz.ratio(a, b) >= 85, (a, b)


# 20 real cross-script true pairs (S1 Latin, match Indic), 2-3 per script,
# drawn from the dev-sample ground truth. Only IDs are committed; names are
# read from the local dev sample, so the test skips where it is absent.
REAL_PAIRS: list[tuple[str, str]] = [
    ("S1-546775342", "S2-661194820"), ("S1-751899837", "S3-337067706"), ("S1-16448025", "S2-439514773"),  # Beng
    ("S1-60794362", "S3-499651457"), ("S1-978171143", "S2-372616075"),  # Deva
    ("S1-920299098", "S3-625050108"), ("S1-882765114", "S2-826098345"), ("S1-340910283", "S2-215992346"),  # Gujr
    ("S1-729056008", "S3-893478853"), ("S1-226054157", "S2-847446897"),  # Guru
    ("S1-601545252", "S2-936059883"), ("S1-201917637", "S3-386713452"),  # Knda
    ("S1-457728197", "S2-339378964"), ("S1-736126753", "S2-42416492"),  # Mlym
    ("S1-289815599", "S2-153874087"), ("S1-493268506", "S3-971705441"),  # Orya
    ("S1-907988432", "S2-855559518"), ("S1-750776584", "S2-481367076"),  # Taml
    ("S1-548362014", "S2-139421746"), ("S1-77098179", "S2-469054576"),  # Telu
]


def _dev_sample_dir() -> Path:
    """Dev-sample folder: $BER_DEV_SAMPLE_DIR, else <repo>/artifacts/dev_sample."""
    return Path(os.environ.get("BER_DEV_SAMPLE_DIR", config.REPO_ROOT / "artifacts" / "dev_sample"))


def test_real_cross_script_pairs_have_near_equal_keys() -> None:
    """Real true pairs in different scripts: key ratio >= 85 (raw names score ~10)."""
    root = _dev_sample_dir() / "train"
    if not (root / "train_source1.tsv").exists():
        pytest.skip(f"dev sample not found at {root} (set BER_DEV_SAMPLE_DIR)")
    ids = {i for pair in REAL_PAIRS for i in pair}
    names: dict[str, str] = {}
    for s in (1, 2, 3):
        df = io_utils.read_source(root / f"train_source{s}.tsv")
        df = df[df["entity_id"].isin(ids)]
        names.update(zip(df["entity_id"], df["business_name"], strict=True))
    assert ids <= names.keys(), sorted(ids - names.keys())
    keys = dict(zip(names, _key(list(names.values())), strict=True))
    ratios = [fuzz.ratio(keys[a], keys[b]) for a, b in REAL_PAIRS]
    assert min(ratios) >= 85, ratios
