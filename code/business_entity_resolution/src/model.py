"""Stages `train` (split=train) and `predict` (split=test): LightGBM.

train:   features_train/ -> <artifacts>/oof_train.parquet (contracts.OOF_COLUMNS),
         models/model_fold{k}.txt, feature_importance.tsv (mean gain).
         GroupKFold(config.N_FOLDS) by S1 over ALL train S1. Each fold trains
         on whole S1 groups sampled from the other folds, capped at
         config.TRAIN_MAX_ROWS pairs, early-stopping on VALID_FRACTION of those
         groups; then every pair of the held-out fold is predicted, so the OOF
         covers all train pairs. Deterministic, seed config.SEED,
         config.LGBM_NUM_THREADS threads.
predict: features_test/ -> <artifacts>/pred_test.parquet (contracts.PRED_COLUMNS),
         mean probability of the fold models.

Memory: one pass reads only s1_id/label (int32 S1 code + int8 label per pair,
5 bytes/pair, ~0.55 GB for 110M pairs). The sampled training rows are loaded
once as float32 (~cap * 1.25 * n_features * 4 bytes, ~3.5 GB at 10M rows and
69 features), binned into one LightGBM Dataset, then freed; each fold uses a
``Dataset.subset`` of it (no raw copy). Prediction streams one feature part at
a time. Relies on features parts keeping each S1's pairs contiguous (the feat
stage guarantees it) to number S1 groups without loading records.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from src import config, contracts, io_utils
from src.config import Paths
from src.experiment_log import log_experiment

logger = logging.getLogger(__name__)

OWNER = "model.py (lane feat)"

MODEL_DIR = "models"
IMPORTANCE_FILE = "feature_importance.tsv"
NON_FEATURES: tuple[str, ...] = (*contracts.FEATURES_KEY_COLUMNS, "label")
DEFAULT_TRAIN_MAX_ROWS = 10_000_000  # until config.TRAIN_MAX_ROWS exists (lane A)
VALID_FRACTION = 0.1  # share of sampled training S1 groups held out for early stopping
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100
# Dataset-level params must be identical for the binned parent and its subsets.
DATASET_PARAMS: dict[str, object] = {"max_bin": 255, "verbose": -1, "seed": config.SEED}
LGBM_PARAMS: dict[str, object] = {
    **DATASET_PARAMS,
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "deterministic": True, "force_row_wise": True,
}


def _parts(folder: Path) -> list[Path]:
    """Sorted part-*.parquet files of a features folder.

    Raises:
        ValueError: If the folder has no parts.
    """
    parts = sorted(folder.glob("part-*.parquet"))
    if not parts:
        raise ValueError(f"{folder}: no part-*.parquet files")
    return parts


def _train_max_rows() -> int:
    """config.TRAIN_MAX_ROWS if the lead has added it, else DEFAULT_TRAIN_MAX_ROWS."""
    return int(getattr(config, "TRAIN_MAX_ROWS", DEFAULT_TRAIN_MAX_ROWS))


def group_folds(sizes: NDArray[np.int64], n_folds: int) -> NDArray[np.int8]:
    """Fold per group, identical to sklearn GroupKFold (no shuffle) on the expanded groups.

    Largest groups first, each to the fold with the fewest samples so far.
    Works on group sizes, so it never materialises per-pair index arrays.

    Args:
        sizes: Pairs per group (group code = position).
        n_folds: Number of folds.

    Returns:
        int8 fold per group.
    """
    order = np.argsort(sizes, kind="stable")[::-1]
    load = np.zeros(n_folds)
    fold = np.zeros(len(sizes), dtype=np.int8)
    for g in order:  # ponytail: Python loop, ~2 s for 2.2M S1; vectorise if it ever matters
        k = int(np.argmin(load))
        load[k] += sizes[g]
        fold[g] = k
    return fold


def read_groups(parts: list[Path], with_label: bool) -> tuple[NDArray[np.int32], NDArray[np.int8] | None]:
    """S1 group code per pair (consecutive runs of s1_id) and, on train, the label.

    Args:
        parts: Feature parts; each S1's pairs are contiguous and never span parts.
        with_label: Also read ``label``.

    Returns:
        ``(codes, labels)``: int32 code per pair in file order, int8 labels or None.

    Raises:
        ValueError: If a part lacks s1_id / label.
    """
    codes, labels = [], []
    offset = 0
    for p in parts:
        cols = ["s1_id", "label"] if with_label else ["s1_id"]
        t = pq.read_table(p, columns=cols)
        s1 = t.column("s1_id")
        starts = np.ones(len(s1), dtype=bool)
        if len(s1) > 1:
            starts[1:] = pc.not_equal(s1.slice(1), s1.slice(0, len(s1) - 1)).to_numpy(zero_copy_only=False)
        c = np.cumsum(starts, dtype=np.int64) - 1 + offset
        codes.append(c.astype(np.int32))
        offset = int(c[-1]) + 1 if len(c) else offset
        if with_label:
            labels.append(t.column("label").to_numpy().astype(np.int8))
    return np.concatenate(codes), (np.concatenate(labels) if with_label else None)


def _load_rows(parts: list[Path], columns: list[str], mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Stack the float32 feature rows selected by ``mask`` (over all pairs), one part at a time."""
    out = np.empty((int(mask.sum()), len(columns)), dtype=np.float32)
    pos = start = 0
    for p in parts:
        n = pq.ParquetFile(p).metadata.num_rows
        m = mask[start : start + n]
        if m.any():
            t = pq.read_table(p, columns=columns)
            block = np.column_stack([t.column(c).to_numpy() for c in columns])[m]
            out[pos : pos + len(block)] = block
            pos += len(block)
            del t
            pa.default_memory_pool().release_unused()  # mimalloc otherwise keeps every part's pages
        start += n
    return out


def _feature_columns(part: Path) -> list[str]:
    """Feature columns of a part (everything but s1_id, cand_id, label)."""
    names: list[str] = pq.ParquetFile(part).schema_arrow.names
    io_utils.require_columns(names, contracts.FEATURES_KEY_COLUMNS, str(part))
    return [c for c in names if c not in NON_FEATURES]


def _write_parquet(path: Path, tables: Iterator[pa.Table]) -> None:
    """Stream tables (same schema) to ``path`` via ``path.tmp`` + rename; holds one table at a time."""
    tmp = path.with_name(path.name + ".tmp")
    first = next(tables)
    with pq.ParquetWriter(tmp, first.schema) as w:
        w.write_table(first)
        for t in tables:
            w.write_table(t)
    tmp.replace(path)


def run_train(paths: Paths) -> None:
    """Cross-validated training with OOF predictions for every train pair.

    Args:
        paths: Run directories (reads features_train/, writes oof_train.parquet,
            models/, feature_importance.tsv and an experiment-log row).

    Raises:
        ValueError: If features_train is missing parts or required columns.
    """
    t0 = time.perf_counter()
    art = paths.artifacts_dir
    parts = _parts(art / "features_train")
    cols = _feature_columns(parts[0])
    io_utils.require_columns(pq.ParquetFile(parts[0]).schema_arrow.names, ["label"], str(parts[0]))
    codes, labels = read_groups(parts, with_label=True)
    assert labels is not None
    sizes = np.bincount(codes).astype(np.int64)
    n_s1 = len(sizes)
    fold_of = group_folds(sizes, config.N_FOLDS)
    logger.info("Train: %d pairs, %d S1, %d features, %d positives; fold pairs %s", len(codes), n_s1, len(cols),
                int(labels.sum()), np.bincount(fold_of[codes], minlength=config.N_FOLDS).tolist())

    # One sample of whole S1 groups serves every fold: each fold drops its own
    # S1 and keeps ~(n_folds-1)/n_folds of the sample, i.e. ~TRAIN_MAX_ROWS pairs.
    rng = np.random.default_rng(config.SEED)
    cap = _train_max_rows() * config.N_FOLDS // (config.N_FOLDS - 1)
    perm = rng.permutation(n_s1)
    sampled = np.zeros(n_s1, dtype=bool)
    sampled[perm[np.cumsum(sizes[perm]) <= cap]] = True
    is_valid = rng.random(n_s1) < VALID_FRACTION
    row_mask = sampled[codes]
    row_s1 = codes[row_mask]
    x = _load_rows(parts, cols, row_mask)
    full = lgb.Dataset(x, label=labels[row_mask], feature_name=cols, params=DATASET_PARAMS,
                       free_raw_data=True).construct()
    del x
    logger.info("Binned %d sampled pairs (%d S1) in %.1fs", len(row_s1), int(sampled.sum()), time.perf_counter() - t0)

    (art / MODEL_DIR).mkdir(parents=True, exist_ok=True)
    params = LGBM_PARAMS | {"num_threads": config.LGBM_NUM_THREADS}
    boosters: list[lgb.Booster] = []
    for k in range(config.N_FOLDS):
        other = fold_of[row_s1] != k
        tr = np.flatnonzero(other & ~is_valid[row_s1])
        va = np.flatnonzero(other & is_valid[row_s1])
        booster = lgb.train(params, full.subset(tr.tolist()), NUM_BOOST_ROUND, valid_sets=[full.subset(va.tolist())],
                            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        booster.save_model(art / MODEL_DIR / f"model_fold{k}.txt", num_iteration=booster.best_iteration)
        boosters.append(booster)
        logger.info("Fold %d: %d train / %d valid pairs, best iteration %d, valid logloss %.5f (%.1fs)", k, len(tr),
                    len(va), booster.best_iteration, booster.best_score["valid_0"]["binary_logloss"],
                    time.perf_counter() - t0)
    del full

    # OOF: every pair predicted by the model that did not see its S1.
    prob = np.empty(len(codes), dtype=np.float32)

    def oof_tables() -> Iterator[pa.Table]:
        """Predict one part with its folds' models; fills ``prob`` as a side effect."""
        start = 0
        for p in parts:
            t = pq.read_table(p, columns=["s1_id", "cand_id", *cols])
            n = t.num_rows
            x = np.column_stack([t.column(c).to_numpy() for c in cols])
            fold = fold_of[codes[start : start + n]]
            for k, booster in enumerate(boosters):
                m = fold == k
                if m.any():
                    prob[start : start + n][m] = np.asarray(booster.predict(
                        x[m], num_iteration=booster.best_iteration, num_threads=config.LGBM_NUM_THREADS))
            yield pa.table({"s1_id": t.column("s1_id"), "cand_id": t.column("cand_id"), "fold": pa.array(fold),
                            "prob": pa.array(prob[start : start + n]), "label": pa.array(labels[start : start + n])})
            start += n
            del t
            pa.default_memory_pool().release_unused()

    _write_parquet(art / "oof_train.parquet", oof_tables())

    gain = pd.DataFrame({f"fold{k}": b.feature_importance("gain", iteration=b.best_iteration)
                         for k, b in enumerate(boosters)}, index=cols)
    gain.insert(0, "gain_mean", gain.mean(axis=1))
    gain = gain.sort_values("gain_mean", ascending=False).rename_axis("feature").reset_index()
    gain.to_csv(art / IMPORTANCE_FILE, sep="\t", index=False, lineterminator="\n", encoding="utf-8")
    logger.info("Top features by gain: %s", ", ".join(gain["feature"].head(10)))

    auc = float(roc_auc_score(labels, prob))
    logger.info("OOF AUC %.5f over %d pairs; train stage %.1fs", auc, len(prob), time.perf_counter() - t0)
    log_experiment(
        f"feat-model-{pd.Timestamp.now():%Y%m%d-%H%M%S}", "feat", "model v0: LightGBM GroupKFold OOF",
        notes=f"OOF AUC={auc:.5f}; pairs={len(prob)}; sampled train pairs={len(row_s1)}; "
              f"best iters={[b.best_iteration for b in boosters]}; data={paths.data_dir.name}",
    )


def run_predict(paths: Paths) -> None:
    """Mean fold-model probability for every test pair -> pred_test.parquet.

    Raises:
        FileNotFoundError: If a fold model is missing.
        ValueError: If features_test lacks a feature the models were trained on.
    """
    art = paths.artifacts_dir
    boosters = [lgb.Booster(model_file=art / MODEL_DIR / f"model_fold{k}.txt") for k in range(config.N_FOLDS)]
    cols: list[str] = boosters[0].feature_name()
    parts = _parts(art / "features_test")
    io_utils.require_columns(pq.ParquetFile(parts[0]).schema_arrow.names, cols, str(parts[0]))

    def pred_tables() -> Iterator[pa.Table]:
        """Mean fold-model probability for one part at a time."""
        for p in parts:
            t = pq.read_table(p, columns=["s1_id", "cand_id", *cols])
            x = np.column_stack([t.column(c).to_numpy() for c in cols])
            prob = np.mean([b.predict(x, num_threads=config.LGBM_NUM_THREADS) for b in boosters], axis=0)
            yield pa.table({"s1_id": t.column("s1_id"), "cand_id": t.column("cand_id"),
                            "prob": pa.array(prob.astype(np.float32))})
            del t
            pa.default_memory_pool().release_unused()

    _write_parquet(art / "pred_test.parquet", pred_tables())
    n_pairs = pq.ParquetFile(art / "pred_test.parquet").metadata.num_rows
    logger.info("Wrote pred_test.parquet: %d pairs, mean of %d fold models", n_pairs, len(boosters))


def run_stage(paths: Paths, split: str) -> None:
    """Train with OOF predictions (split=train) or predict test (split=test).

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.
    """
    if split == "train":
        run_train(paths)
    else:
        run_predict(paths)
