"""Indic scripts -> Latin transliteration and a phonetic name key.

Why: many India true matches are the same name written in different scripts
(``Om Estate Limited`` vs ``ಓಂ ಎಸ್ಟೇಟ್ ಲಿಮಿಟೆಡ್``); char n-gram TF-IDF cannot link
them unless both sides are in one script. Our own code, no external libraries.

Step 1 — any Brahmic script -> Devanagari. Bengali, Gurmukhi, Gujarati,
Oriya, Tamil, Telugu, Kannada and Malayalam (all nine scripts appear in the
data, E10/E12) use parallel Unicode blocks 0x80 apart starting at Devanagari
U+0900, so a letter maps to ``U+0900 + (cp - U+0900) % 0x80``. Offsets
0x70-0x7F (fractions, currency, script-specific signs) map to a space unless
listed below. Code points whose offset lands on a wrong or missing Devanagari
character are mapped explicitly (``_OVERRIDES``):

- Bengali: khanda ta U+09CE -> त्; AU length mark U+09D7 -> ौ;
  Assamese ra/wa U+09F0/U+09F1 -> र/व.
- Gurmukhi: tippi U+0A70 -> anusvara; addak U+0A71 (doubles the next
  consonant) -> dropped; iri/ura vowel carriers U+0A72/U+0A73 -> अ;
  ek onkar U+0A74 -> ॐ; udaat U+0A51 -> dropped; yakash U+0A75 -> ्य.
- Oriya: wa U+0B71 -> व; AI/AU length marks U+0B56/U+0B57 -> ै/ौ;
  overline U+0B55 -> dropped.
- Tamil: ழ U+0BB4, ற U+0BB1, ன U+0BA9 have no common Devanagari letter;
  they land on the rare ऴ/ऱ/ऩ, which the Latin step reads as l/r/n.
  AU length mark U+0BD7 -> ौ. ச U+0B9A -> स ("s", as in loanwords and most
  names; "ch" would become "k" in the key). Tamil has no voiced/aspirated consonants, so
  க is always "k" (the phonetic key does not merge voicing).
- Telugu: tsa/dza/rrra U+0C58-U+0C5A -> च/ज/र; nakaara pollu U+0C5D -> न्;
  combining anusvara U+0C04 -> anusvara; length marks U+0C55 -> dropped,
  U+0C56 -> ै.
- Kannada: nakaara pollu U+0CDD -> न्; length marks U+0CD5 -> dropped,
  U+0CD6 -> ै.
- Malayalam: chillu U+0D7A-U+0D7F (nn, n, rr, l, ll, k) and U+0D54-U+0D56
  (m, y, lll) -> consonant + virama; dot reph U+0D4E -> र्; AU length mark
  U+0D57 -> ौ; TTTA U+0D3A -> ट; vertical-bar/circular virama
  U+0D3B/U+0D3C -> virama; vedic anusvara U+0D04 -> anusvara; archaic II
  U+0D5F -> ई; fractions U+0D58-U+0D5E and para sign U+0D4F -> space;
  clusters റ്റ -> ट्ट ("tt") and ന്റ -> न्ट ("nt"), where RRA is not "r".
- Zero-width joiner / non-joiner (U+200C/U+200D) are removed.

Text is NFC-composed first so split vowel signs (e.g. Tamil ொ = ெ + ா,
Malayalam ൌ = െ + ൗ) map as one sign.

Step 2 — Devanagari -> Latin (``_devanagari_to_latin``): each consonant
carries an inherent "a"; a virama removes it, a vowel sign replaces it; the
inherent "a" is dropped at the end of a word unless the consonant closes a
conjunct ending in र/य/व (मिश्र -> mishra, आदित्य -> aaditya, but राम -> raam,
परफेक्ट -> paraphekt). Anusvara and
chandrabindu -> "n" ("m" before p/b/bh/m: मुंबई -> mumbai; word-final anusvara -> "m":
ഓം -> om, കേരളം -> keralam); visarga ->
"h"; nukta forms क़ q, ख़ kh, ग़ g, ज़ z, ड़ r, ढ़ rh, फ़ f, य़ y; ज्ञ -> gy.
Digits of all nine scripts -> ASCII. Latin and every other character pass
through unchanged.

Step 3 — ``phonetic_key``: lowercase, ee/ii -> i, oo/uu -> u, aa -> a,
aspirates kh gh ch jh th dh ph bh sh -> k g c j t d p b s, w -> v, runs of a
repeated letter -> one; then x -> ks, q -> k, z -> j, f -> p, c -> k, and every
vowel (a e i o u y) except a word's first letter is dropped. Applied to every
record (no country logic).

The vowel drop is what links cross-script pairs (loanwords like "silvar
treding" for "silver trading"); used alone the key loses same-script
precision, so blocking should vectorise ``name_norm | name_norm | name_key``
(prep builds name_key from name_core, i.e. without the legal suffix)
(name_norm counted twice; dev-sample R@20 vs prep v0: India 0.661 -> 0.683,
US 0.829 -> 0.835; see the PR table).

Memory/speed: vectorised over a pandas Series; the per-character Python work
runs only on strings that contain an Indic character.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

_DEVA = 0x0900
_INDIC_FIRST, _INDIC_END = 0x0900, 0x0D80  # Devanagari .. Malayalam
_HAS_INDIC = "[ऀ-ൿ]"

_OVERRIDES: dict[int, str] = {
    # Bengali
    0x09CE: "त्", 0x09D7: "ौ", 0x09F0: "र", 0x09F1: "व",
    # Gurmukhi
    0x0A51: "", 0x0A70: "ं", 0x0A71: "", 0x0A72: "अ", 0x0A73: "अ", 0x0A74: "ॐ", 0x0A75: "्य",
    # Oriya
    0x0B55: "", 0x0B56: "ै", 0x0B57: "ौ", 0x0B71: "व",
    # Tamil
    0x0B9A: "स", 0x0BD7: "ौ",
    # Telugu
    0x0C04: "ं", 0x0C55: "", 0x0C56: "ै", 0x0C58: "च", 0x0C59: "ज", 0x0C5A: "र", 0x0C5D: "न्",
    # Kannada
    0x0CD5: "", 0x0CD6: "ै", 0x0CDD: "न्",
    # Malayalam
    0x0D04: "ं", 0x0D3A: "ट", 0x0D3B: "्", 0x0D3C: "्", 0x0D4E: "र्", 0x0D4F: " ",
    0x0D54: "म्", 0x0D55: "य्", 0x0D56: "ऴ्", 0x0D57: "ौ",
    **{cp: " " for cp in range(0x0D58, 0x0D5F)}, 0x0D5F: "ई",
    0x0D7A: "ण्", 0x0D7B: "न्", 0x0D7C: "र्", 0x0D7D: "ल्", 0x0D7E: "ळ्", 0x0D7F: "क्",
    # joiners
    0x200C: "", 0x200D: "",
}

# Malayalam clusters whose sound differs from their parts: റ്റ "tt", ന്റ "nt".
_CLUSTERS: tuple[tuple[str, str], ...] = (("\u0d31\u0d4d\u0d31", "ट्ट"), ("\u0d28\u0d4d\u0d31", "न्ट"))

_CONS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # precomposed nukta forms (U+0958-U+095F)
    **dict(zip(map(chr, range(0x0958, 0x0960)), ("q", "kh", "g", "z", "r", "rh", "f", "y"))),
}
_NUKTA: dict[str, str] = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}

_SIGNS: dict[str, str] = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "ॄ": "ri",
    "ॅ": "e", "ॆ": "e", "े": "e", "ै": "ai", "ॉ": "o", "ॊ": "o", "ो": "o", "ौ": "au",
    "ॢ": "l", "ॣ": "l", "ऺ": "e", "ऻ": "e", "ॎ": "e", "ॏ": "au",
    "ॕ": "e", "ॖ": "u", "ॗ": "u",
}
_VIRAMA, _NUKTA_SIGN = "\u094d", "\u093c"
_KEEP_FINAL_A = frozenset("रयव\u095f")  # मिश्र mishra, आदित्य aditya

_OTHER: dict[str, str] = {
    "ऄ": "a", "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ॠ": "ri", "ऌ": "l", "ॡ": "l", "ऍ": "e", "ऎ": "e", "ए": "e", "ऐ": "ai",
    "ऑ": "o", "ऒ": "o", "ओ": "o", "औ": "au", "ॲ": "a",
    "ॳ": "e", "ॴ": "o", "ॵ": "au", "ॶ": "u", "ॷ": "u",
    "ं": "n", "ँ": "n", "ऀ": "n", "ः": "h", "ॐ": "om",
    "।": " ", "॥": " ", "॰": " ",
    **{chr(0x0966 + d): str(d) for d in range(10)},
}

_CONS_CLASS = "".join(_CONS)
_SIGN_CLASS = "".join(_SIGNS)
# consonant, optional nukta, optional vowel sign or virama
_SYLLABLE = re.compile(f"([{_CONS_CLASS}])({_NUKTA_SIGN}?)([{_SIGN_CLASS}{_VIRAMA}]?)")
_LABIAL_ANUSVARA = re.compile("[ंँ](?=[पबभम])|ं(?![ऀ-ॣ])")  # not फ: usually "f" in loanwords
_CARRIER = re.compile(f"अ(?=[{_SIGN_CLASS}])")  # Gurmukhi iri/ura + vowel sign


@functools.cache
def _to_devanagari_table() -> dict[int, str]:
    """``str.translate`` table: other Indic blocks -> Devanagari (see module docstring)."""
    table: dict[int, str] = {}
    for cp in range(_INDIC_FIRST + 0x80, _INDIC_END):
        if unicodedata.name(chr(cp), ""):
            off = (cp - _DEVA) % 0x80
            table[cp] = chr(_DEVA + off) if off < 0x70 else " "
    table.update(_OVERRIDES)
    return table


@functools.cache
def _latin_table() -> dict[int, str]:
    """``str.translate`` table for Devanagari left after syllables: vowels, signs, digits.

    Any other Devanagari code point (orphan signs, rare Sindhi letters) is dropped.
    """
    table: dict[int, str] = {cp: "" for cp in range(_DEVA, _DEVA + 0x80)}
    table.update({ord(k): v for k, v in (_SIGNS | _OTHER).items()})
    return table


def _is_word_char(ch: str) -> bool:
    """True for a Devanagari letter or sign that continues a word (not danda/digit)."""
    return _DEVA <= ord(ch) <= 0x0963


def _syllable(m: re.Match[str]) -> str:
    """Latin for one consonant (+ nukta) (+ vowel sign / virama)."""
    cons, nukta, sign = m.group(1), m.group(2), m.group(3)
    out = _NUKTA[cons] if nukta and cons in _NUKTA else _CONS[cons]
    if sign == _VIRAMA:
        return out
    if sign:
        return out + _SIGNS[sign]
    s, end = m.string, m.end()
    if end < len(s) and _is_word_char(s[end]):
        return out + "a"
    # word-final: drop the inherent "a" unless a conjunct ends in r / y / v
    start = m.start()
    keep = cons in _KEEP_FINAL_A and start >= 2 and s[start - 1] == _VIRAMA
    return out + ("a" if keep else "")


def _devanagari_to_latin(text: str) -> str:
    """Romanise Devanagari in ``text``; other characters pass through."""
    text = text.replace("ज्ञ", "ग्य")
    text = _LABIAL_ANUSVARA.sub("म्", text)
    text = _CARRIER.sub("", text)
    text = _SYLLABLE.sub(_syllable, text)
    return text.translate(_latin_table())


def transliterate_text(text: str) -> str:
    """Romanise every Indic-script character of one string.

    Args:
        text: Any string (mixed scripts allowed).

    Returns:
        ``text`` with Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
        Telugu, Kannada and Malayalam converted to lowercase Latin and Indic
        digits to ASCII; everything else unchanged.
    """
    text = unicodedata.normalize("NFC", text)
    for src, dst in _CLUSTERS:
        text = text.replace(src, dst)
    text = text.translate(_to_devanagari_table())
    return _devanagari_to_latin(text)


def transliterate(s: pd.Series) -> pd.Series:
    """Vectorised ``transliterate_text``; only strings with an Indic character are touched.

    Args:
        s: String Series; no missing values.

    Returns:
        String Series, same index.
    """
    mask = s.str.contains(_HAS_INDIC, regex=True).to_numpy(dtype=bool)
    if not mask.any():
        return s.astype("str")
    vals = s.to_numpy(dtype=object)  # copy; Arrow-backed Series reject masked list assignment
    vals[mask] = [transliterate_text(t) for t in vals[mask]]
    return pd.Series(vals, index=s.index, dtype="str")


_KEY_RULES: tuple[tuple[str, str], ...] = (
    (r"ee|ii", "i"),
    (r"oo|uu", "u"),
    (r"aa", "a"),
    (r"([kgcjtdpbs])h+", r"\1"),
    (r"w", "v"),
    # doubled letters; one rule per letter (pyarrow's RE2 has no backreferences)
    *((f"{c}{{2,}}", c) for c in "abcdefghijklmnopqrstuvwxyz"),
    # letters that Indic spellings of English loanwords swap (E11/dev-sample pairs):
    # exports/eksports, food/phood, business/bijanes, classic/klasik
    (r"x", "ks"), (r"q", "k"), (r"z", "j"), (r"f", "p"), (r"c", "k"),
    # consonant skeleton: silver/silvar, trading/treding, energy/enarji
    (r"\B[aeiouy]", ""),
)


def phonetic_key(s: pd.Series) -> pd.Series:
    """Phonetic key of already-normalised text (see module docstring, step 3).

    Args:
        s: String Series, e.g. ``name_norm``; no missing values.

    Returns:
        String Series, same index.
    """
    out = s.str.lower()
    for pat, rep in _KEY_RULES:
        out = out.str.replace(pat, rep, regex=True)
    return out
