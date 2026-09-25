"""Stage `prep`: build records_{split}.parquet from the three source TSVs.

*** PREP v0 (lead's stop-gap) ***
Minimal, country-agnostic normalisation so blocking and features can start
before Member 2's full version lands. Member 2 replaces the internals
(legal-suffix extraction, abbreviation dictionaries, better acronyms) WITHOUT
changing the output contract (contracts.RECORDS_COLUMNS and RECORDS_SCHEMA).

v0 rules:
- name_norm / addr_norm: NFKD, lowercase, strip Latin combining accents
  (U+0300-U+036F only, so Devanagari virama/nukta/vowel signs survive),
  "&" -> " and ", punctuation / symbols / whitespace / control chars -> space,
  spaces collapsed.
- name_core = name_norm, legal_suffix = "" (placeholders).
- name_acronym = first character of each name_norm token.
- postal_tokens = every standalone 5-6 digit ASCII number in addr_norm.
- num_tokens = every standalone ASCII number in addr_norm.
- landmark_flag = addr_norm has the word near / opp / opposite / behind / beside.
- name_empty / addr_empty = normalised field is empty.
- source = ID prefix (S1 / S2 / S3), checked against the file it came from.

Memory: one source file at a time is held in memory; records are processed
and written in chunks of PREP_CHUNK_ROWS rows.
"""

from __future__ import annotations

import functools
import gc
import logging
import sys
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import io_utils
from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "normalize.py (Member 2)"
PREP_CHUNK_ROWS = 500_000
LANDMARK_WORDS: tuple[str, ...] = ("near", "opp", "opposite", "behind", "beside")

_STR = pa.string()
_LIST = pa.list_(pa.string())
RECORDS_SCHEMA = pa.schema([
    ("entity_id", _STR), ("source", _STR), ("country", _STR),
    ("name_raw", _STR), ("addr_raw", _STR),
    ("name_norm", _STR), ("name_core", _STR), ("legal_suffix", _STR),
    ("name_acronym", _STR), ("addr_norm", _STR),
    ("postal_tokens", _LIST), ("num_tokens", _LIST),
    ("landmark_flag", pa.bool_()), ("name_empty", pa.bool_()), ("addr_empty", pa.bool_()),
])

_LANDMARK_RE = r"(?:^| )(?:" + "|".join(LANDMARK_WORDS) + r")(?: |$)"


@functools.cache
def _separator_table() -> dict[int, str]:
    """Map every punctuation, symbol, separator and control code point to a space.

    Built once (scans all Unicode code points, ~0.5 s).
    """
    return {
        cp: " "
        for cp in range(sys.maxunicode + 1)
        if unicodedata.category(chr(cp))[0] in "PSZ" or unicodedata.category(chr(cp)) == "Cc"
    }


def normalize_text(s: pd.Series) -> pd.Series:
    """Apply the v0 text normalisation (see module docstring).

    Args:
        s: String Series (names or addresses); no missing values.

    Returns:
        Normalised string Series, same index.
    """
    out = s.str.normalize("NFKD").str.lower()
    out = out.str.replace(r"[̀-ͯ]", "", regex=True)
    out = out.str.replace("&", " and ", regex=False)
    out = out.str.translate(_separator_table())
    return out.str.replace(r" +", " ", regex=True).str.strip()


def postal_tokens(addr_norm: pd.Series) -> pd.Series:
    """Standalone 5-6 digit numbers in a normalised address (lists of str)."""
    return addr_norm.str.findall(r"(?<!\S)[0-9]{5,6}(?!\S)")


def num_tokens(addr_norm: pd.Series) -> pd.Series:
    """Every standalone number in a normalised address (lists of str)."""
    return addr_norm.str.findall(r"(?<!\S)[0-9]+(?!\S)")


def landmark_flag(addr_norm: pd.Series) -> pd.Series:
    """True where a normalised address contains a landmark word."""
    return addr_norm.str.contains(_LANDMARK_RE, regex=True).astype(bool)


def acronym(name_norm: pd.Series) -> pd.Series:
    """First character of each whitespace token of a normalised name."""
    return name_norm.map(lambda n: "".join(t[0] for t in n.split())).astype("str")


def build_records(src: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise source records into the records contract.

    Args:
        src: One row per record with string columns ``entity_id,
            business_name, business_address, country`` (as from
            ``io_utils.read_source``).
        source: ``"S1"``, ``"S2"`` or ``"S3"``.

    Returns:
        One row per record with columns ``contracts.RECORDS_COLUMNS``:
        strings, ``postal_tokens`` / ``num_tokens`` as lists of str, and bool
        ``landmark_flag`` / ``name_empty`` / ``addr_empty``.

    Raises:
        ValueError: If a source column is missing.
    """
    io_utils.require_columns(src.columns, io_utils.SOURCE_COLUMNS, f"build_records({source})")
    name_norm = normalize_text(src["business_name"])
    addr_norm = normalize_text(src["business_address"])
    return pd.DataFrame({
        "entity_id": src["entity_id"],
        "source": source,
        "country": src["country"],
        "name_raw": src["business_name"],
        "addr_raw": src["business_address"],
        "name_norm": name_norm,
        "name_core": name_norm,
        "legal_suffix": "",
        "name_acronym": acronym(name_norm),
        "addr_norm": addr_norm,
        "postal_tokens": postal_tokens(addr_norm),
        "num_tokens": num_tokens(addr_norm),
        "landmark_flag": landmark_flag(addr_norm),
        "name_empty": name_norm.eq(""),
        "addr_empty": addr_norm.eq(""),
    })


def run_stage(paths: Paths, split: str) -> None:
    """Build ``<artifacts>/records_{split}.parquet`` from the split's S1/S2/S3 TSVs.

    Writes to a ``.tmp`` file and renames at the end, so a crash never leaves
    a partial file that the runner would skip.

    Args:
        paths: Resolved run directories (reads ``paths.data_dir/{split}/``).
        split: ``"train"`` or ``"test"``.

    Raises:
        ValueError: If a source file fails the ``io_utils`` checks or contains
            an ID whose prefix does not match the file's source.
    """
    out = paths.artifacts_dir / f"records_{split}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with pq.ParquetWriter(tmp, RECORDS_SCHEMA) as writer:
        for s in (1, 2, 3):
            source = f"S{s}"
            src = io_utils.read_source(paths.data_dir / split / f"{split}_source{s}.tsv")
            bad = ~src["entity_id"].str.startswith(f"{source}-")
            if bad.any():
                raise ValueError(
                    f"{split}_source{s}.tsv: {int(bad.sum())} id(s) without prefix {source}-, "
                    f"e.g. {src.loc[bad, 'entity_id'].iloc[0]}"
                )
            n_name_empty = n_addr_empty = 0
            for start in range(0, len(src), PREP_CHUNK_ROWS):
                rec = build_records(src.iloc[start : start + PREP_CHUNK_ROWS], source)
                n_name_empty += int(rec["name_empty"].sum())
                n_addr_empty += int(rec["addr_empty"].sum())
                if start == 0 and len(rec):
                    logger.debug("Sample %s record: %s", source, rec.iloc[0].to_dict())
                writer.write_table(pa.Table.from_pandas(rec, schema=RECORDS_SCHEMA, preserve_index=False))
            total += len(src)
            logger.info("%s %s: %d records | countries %s", split, source, len(src),
                        src["country"].value_counts().to_dict())
            if n_name_empty or n_addr_empty:
                logger.warning("%s %s: %d empty names, %d empty addresses after normalisation",
                               split, source, n_name_empty, n_addr_empty)
            del src
            gc.collect()
    tmp.replace(out)
    logger.info("Wrote %s: %d records", out, total)
