"""Name-blocking recall at K per country for several name columns (E9 method).

Usage (from code/business_entity_resolution/, after ``--stage prep --split train``):
    python -m src.name_recall --data-dir PATH --artifacts-dir PATH

For each name column, fits char 3-gram TF-IDF on every train record, then for
the same seeded sample of S1 with matches ranks each true match among ALL
same-country S2/S3 records by cosine (``eda.true_match_ranks``; cosine 0 =
miss). Columns: ``name_v0`` (prep v0 ``normalize_text(name_raw)``, the E9
baseline) plus any records columns, default ``name_norm`` and ``name_key``;
``a+b+c`` means the text ``a | b | c`` (default ``name_norm+name_norm+name_key``,
the recommended blocking text).
Writes ``<artifacts-dir>/name_recall.md`` and logs the table at INFO.

Memory: records (entity_id, source, country and the name columns) plus one
country's S2/S3 pool as a transposed float32 CSR matrix at a time (the same
footprint as eda E9, < 6 GB on full train).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from src import config, eda, io_utils, normalize
from src.logging_utils import setup_logging, track_stage

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS: tuple[str, ...] = ("name_v0", "name_norm", "name_key", "name_norm+name_norm+name_key")
KS: tuple[int, ...] = (10, 20, 50)


def _ranks_for_column(vec: eda.NameVectorizer, names: pd.Series, cc: NDArray[np.int32],
                      is_s1: NDArray[np.bool_], sample: NDArray[np.intp],
                      true_cols: dict[int, NDArray[np.intp]], code: int) -> NDArray[np.float64]:
    """Ranks of the true matches of ``sample`` S1 (all of country ``code``) in their country pool.

    Args:
        vec: Vectorizer fitted on ``names``.
        names: Name text per record (positional).
        cc: Country code per record.
        is_s1: True for S1 records.
        sample: Record positions of the sampled S1.
        true_cols: Record positions of each sampled S1's true matches.
        code: Country code of the pool.

    Returns:
        Ranks, concatenated in sample order.
    """
    pool = np.flatnonzero(~is_s1 & (cc == code))
    col_of = np.full(len(names), -1, dtype=np.int64)
    col_of[pool] = np.arange(len(pool))
    blocks = [vec.transform(names.iloc[pool[i : i + eda.SCAN_CHUNK_ROWS]])
              for i in range(0, len(pool), eda.SCAN_CHUNK_ROWS)]
    xt = sp.vstack(blocks, format="csr").T.tocsr()
    del blocks
    out: list[NDArray[np.float64]] = []
    for i in range(0, len(sample), eda.RANK_BATCH):
        batch = sample[i : i + eda.RANK_BATCH]
        out.append(eda.true_match_ranks(vec.transform(names.iloc[batch]), xt,
                                        [col_of[true_cols[int(p)]] for p in batch]))
    return np.concatenate(out) if out else np.array([])


def name_recall(records: pd.DataFrame, gt: pd.DataFrame, columns: Sequence[str],
                n_rank_s1: int, seed: int = config.SEED) -> pd.DataFrame:
    """Recall at KS of true matches by name TF-IDF, per country and column.

    Args:
        records: One row per train record with str ``entity_id, source,
            country`` and every column in ``columns``.
        gt: Ground-truth long table (``io_utils.read_ground_truth_pairs``):
            str ``s1_id, cand_id``, ``cand_id == ""`` for singletons.
        columns: Name columns to compare.
        n_rank_s1: S1 with matches sampled per country (same sample for all
            columns).
        seed: Sampling seed.

    Returns:
        One row per (country, column): ``country, column, n_s1, n_pairs`` and
        ``R@K`` floats.

    Raises:
        ValueError: If a required column is missing.
    """
    io_utils.require_columns(records.columns, ("entity_id", "source", "country", *columns), "name_recall")
    io_utils.require_columns(gt.columns, ("s1_id", "cand_id"), "name_recall ground truth")
    rng = np.random.default_rng(seed)
    pos = pd.Series(np.arange(len(records)), index=records["entity_id"].to_numpy())
    gt = gt[gt["cand_id"] != ""]
    p1 = pos.reindex(gt["s1_id"].to_numpy()).to_numpy()
    p2 = pos.reindex(gt["cand_id"].to_numpy()).to_numpy()
    ok = ~(np.isnan(p1) | np.isnan(p2))
    if not ok.all():
        logger.warning("name_recall: %d ground-truth pairs reference unknown IDs, skipped", int((~ok).sum()))
    p1, p2 = p1[ok].astype(np.intp), p2[ok].astype(np.intp)
    country = records["country"].astype("category")
    cc: NDArray[np.int32] = np.asarray(country.cat.codes, dtype=np.int32)
    is_s1 = (records["source"] == "S1").to_numpy(dtype=bool)
    order = np.argsort(p1, kind="stable")
    with_match, starts = np.unique(p1[order], return_index=True)
    true_cols = dict(zip(with_match.tolist(), np.split(p2[order], starts[1:]), strict=True))
    samples: dict[int, NDArray[np.intp]] = {}
    for code in range(len(country.cat.categories)):
        cand = with_match[cc[with_match] == code]
        if len(cand):
            samples[code] = np.sort(rng.choice(cand, size=min(n_rank_s1, len(cand)), replace=False))
    rows = []
    for col in columns:
        vec = eda.NameVectorizer()
        for i in range(0, len(records), eda.SCAN_CHUNK_ROWS):
            vec.fit(records[col].iloc[i : i + eda.SCAN_CHUNK_ROWS])
        for code, sample in samples.items():
            name = country.cat.categories[code]
            with track_stage(f"name_recall-{name}-{col}"):
                r = _ranks_for_column(vec, records[col], cc, is_s1, sample, true_cols, code)
            rec = eda.recall_at_k(r, KS)
            rows.append({"country": str(name), "column": col, "n_s1": len(sample), "n_pairs": len(r),
                         **{f"R@{k}": v for k, v in rec.items()}})
            logger.info("%s %s: %s", name, col, "  ".join(f"R@{k}={v:.3f}" for k, v in rec.items()))
    return pd.DataFrame(rows).sort_values("country", kind="stable", ignore_index=True)


def to_markdown(df: pd.DataFrame) -> str:
    """Render the recall table as a GitHub markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train/train_ground_truth.tsv")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="folder with records_train.parquet")
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument("--n-rank-s1", type=int, default=eda.N_RANK_S1, help="sampled S1 per country")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    paths = config.get_paths(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    setup_logging(args.log_level, paths.log_dir)
    stored = sorted({p for c in args.columns for p in c.split("+")} - {"name_v0"})
    rec = io_utils.load_parquet(paths.artifacts_dir / "records_train.parquet",
                                columns=["entity_id", "source", "country", "name_raw", *stored])
    if "name_v0" in args.columns:
        rec["name_v0"] = normalize.normalize_text(rec["name_raw"])
    for c in args.columns:
        if "+" in c:
            parts = c.split("+")
            rec[c] = rec[parts[0]].str.cat([rec[p] for p in parts[1:]], sep=" | ")
    gt = io_utils.read_ground_truth_pairs(paths.data_dir / "train" / "train_ground_truth.tsv")
    table = to_markdown(name_recall(rec, gt, args.columns, args.n_rank_s1))
    out = paths.artifacts_dir / "name_recall.md"
    out.write_text(table + "\n", encoding="utf-8", newline="\n")
    for line in table.splitlines():
        logger.info("%s", line)


if __name__ == "__main__":
    main()
