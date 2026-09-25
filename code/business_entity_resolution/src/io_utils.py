"""TSV and parquet I/O. All pipeline code reads and writes data through here.

TSVs are read with ``quoting=csv.QUOTE_NONE`` (a stray ``"`` cannot swallow
lines; CSV-escaped quotes such as ``\"\"\"ehpad Club SAS\"`` stay literal) and
``keep_default_na=False`` (a business named "NA" or an empty ID list stays a
string). Every read is checked against the file's physical line count.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

logger = logging.getLogger(__name__)

SOURCE_COLUMNS: tuple[str, ...] = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS: tuple[str, str] = ("source1_entity_id", "matched_entity_ids")
MATCHING_HEADER: tuple[str, str] = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER: tuple[str, str] = ("source1_entity_id", "candidate_entity_ids")

_READ_CHUNK_BYTES = 1 << 20


def require_columns(columns: Iterable[str], required: Iterable[str], context: str) -> None:
    """Raise if any required column is missing.

    Args:
        columns: Column names that are present.
        required: Column names that must be present.
        context: Where the check happens (file or stage name), for the message.

    Raises:
        ValueError: Naming every missing column.
    """
    present = set(columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise ValueError(f"{context}: missing required column(s) {missing}; found {sorted(present)}")


def count_data_lines(path: Path) -> int:
    """Count physical lines in a file, counting a final line without '\\n'.

    Reads in 1 MiB chunks, so memory use is constant.

    Args:
        path: File to count.

    Returns:
        Number of lines, including the header line.
    """
    n = 0
    last = b"\n"
    with path.open("rb") as f:
        while chunk := f.read(_READ_CHUNK_BYTES):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (last != b"\n")


def _read_tsv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """Read a TSV as all-string columns and verify row count and header.

    Args:
        path: TSV file with a header row.
        required: Columns that must be in the header.

    Returns:
        One row per data line; every column has pandas string dtype.

    Raises:
        ValueError: If a required column is missing or the parsed row count
            differs from the file's line count minus 1.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    require_columns(df.columns, required, str(path))
    expected = count_data_lines(path) - 1
    if len(df) != expected:
        raise ValueError(
            f"{path}: parsed row count {len(df)} != file line count - 1 ({expected}); "
            "check for blank lines or stray line breaks"
        )
    return df


def read_source(path: Path) -> pd.DataFrame:
    """Read a source TSV (S1, S2 or S3).

    Loads the whole file into memory (~2x file size for string columns).

    Args:
        path: ``*_source{1,2,3}.tsv`` file.

    Returns:
        DataFrame, one row per record, string columns
        ``entity_id, business_name, business_address, country``. Empty fields
        are ``""``, never NaN.

    Raises:
        ValueError: If a column is missing or the row count check fails.
    """
    df = _read_tsv(path, SOURCE_COLUMNS)
    logger.info("Read %s: %d rows", path.name, len(df))
    return df


def read_ground_truth(path: Path) -> dict[str, set[str]]:
    """Read the ground-truth TSV into a mapping.

    Args:
        path: TSV with columns ``source1_entity_id, matched_entity_ids``
            (comma-separated IDs, empty for singletons).

    Returns:
        ``{s1_id: set of matched S2/S3 IDs}``; singletons map to an empty set.

    Raises:
        ValueError: If a column is missing, the row count check fails, or an
            S1 ID appears more than once.
    """
    df = _read_tsv(path, GROUND_TRUTH_COLUMNS)
    s1_col, match_col = GROUND_TRUTH_COLUMNS
    if df[s1_col].duplicated().any():
        raise ValueError(f"{path}: duplicate {s1_col} rows")
    truth = {
        s1: set(ids.split(",")) if ids else set()
        for s1, ids in zip(df[s1_col].tolist(), df[match_col].tolist(), strict=True)
    }
    logger.info("Read %s: %d S1 rows", path.name, len(truth))
    return truth


def write_id_list_tsv(
    mapping: Mapping[str, Iterable[str]],
    s1_ids: Sequence[str],
    path: Path,
    header: tuple[str, str],
) -> None:
    """Write a submission-style TSV: one row per S1 with comma-joined IDs.

    Rows follow ``s1_ids`` order. IDs are de-duplicated keeping first-seen
    order; an S1 missing from ``mapping`` gets an empty list. UTF-8, "\\n"
    line endings. Streams rows to disk.

    Args:
        mapping: ``{s1_id: iterable of S2/S3 IDs}``.
        s1_ids: Every S1 ID to write, each exactly once.
        path: Output file; parent folders are created.
        header: Exact two column names, e.g. ``MATCHING_HEADER``.

    Raises:
        ValueError: If ``s1_ids`` contains duplicates.
    """
    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError(f"{path}: s1_ids contains duplicate IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = dict.fromkeys(mapping.get(s1, ()))
            f.write(f"{s1}\t{','.join(ids)}\n")
    logger.info("Wrote %s: %d rows", path, len(s1_ids))


def write_source_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write records in the original source TSV format, values unchanged.

    Values are written verbatim (no quoting or escaping), so the file reads
    back identically through ``read_source``. UTF-8, "\\n" line endings.
    Holds one joined line per row in memory while writing.

    Args:
        df: One row per record, string columns ``entity_id, business_name,
            business_address, country`` (other columns are ignored).
        path: Output file; parent folders are created.

    Raises:
        ValueError: If a column is missing or a value contains a tab or
            line break.
    """
    require_columns(df.columns, SOURCE_COLUMNS, f"write_source_tsv({path})")
    cols = df[list(SOURCE_COLUMNS)]
    for c in SOURCE_COLUMNS:
        if cols[c].str.contains(r"[\t\r\n]", regex=True).any():
            raise ValueError(f"write_source_tsv({path}): column {c} contains a tab or line break")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = cols[SOURCE_COLUMNS[0]]
    for c in SOURCE_COLUMNS[1:]:
        lines = lines + "\t" + cols[c]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(SOURCE_COLUMNS) + "\n")
        f.writelines(line + "\n" for line in lines)
    logger.info("Wrote %s: %d rows", path, len(df))


def save_parquet(df: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    """Save a DataFrame (or a subset of its columns) to parquet without the index.

    Args:
        df: Any DataFrame.
        path: Output ``.parquet`` file; parent folders are created.
        columns: Columns to save, in order; all columns if None.

    Raises:
        ValueError: If a requested column is not in ``df``.
    """
    if columns is not None:
        require_columns(df.columns, columns, f"save_parquet({path})")
        df = df[list(columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s: %d rows x %d cols", path, len(df), df.shape[1])


def load_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a parquet file or a folder of parquet parts, reading only ``columns``.

    Args:
        path: ``.parquet`` file or folder of ``part-*.parquet`` files.
        columns: Columns to read; all columns if None.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If a requested column is not in the parquet schema.
    """
    if columns is not None:
        schema_names = pads.dataset(path, format="parquet").schema.names
        require_columns(schema_names, columns, f"load_parquet({path})")
    df = pd.read_parquet(path, columns=None if columns is None else list(columns))
    logger.info("Loaded %s: %d rows x %d cols", path, len(df), df.shape[1])
    return df
