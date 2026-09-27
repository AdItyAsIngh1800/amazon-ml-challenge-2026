"""Tests for model.py on synthetic feature parts."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src import config, contracts, model
from src.config import Paths


def test_group_folds_match_sklearn() -> None:
    """Size-based fold assignment equals GroupKFold on the expanded per-pair groups."""
    sizes = np.random.default_rng(0).integers(1, 60, size=500).astype(np.int64)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    expected = np.empty(len(sizes), dtype=np.int8)
    for k, (_, test) in enumerate(GroupKFold(5).split(groups, groups=groups)):
        expected[groups[test]] = k
    assert (model.group_folds(sizes, 5) == expected).all()


def _write_parts(folder: Path, n_s1: int, n_cand: int, n_parts: int, seed: int, label: bool) -> pd.DataFrame:
    """Feature parts with two informative and one noise feature; S1 groups never span parts."""
    rng = np.random.default_rng(seed)
    s1 = np.repeat([f"S1-{i}" for i in range(n_s1)], n_cand)
    y = (rng.random(len(s1)) < 0.2).astype(np.int8)
    df = pd.DataFrame({
        "s1_id": s1, "cand_id": [f"S2-{i}" for i in range(len(s1))],
        "sim_a": (y + rng.normal(0, 0.4, len(s1))).astype(np.float32),
        "sim_b": np.where(rng.random(len(s1)) < 0.1, np.nan, y * 0.5 + rng.random(len(s1))).astype(np.float32),
        "noise": rng.random(len(s1)).astype(np.float32),
    })
    if label:
        df["label"] = y
    folder.mkdir(parents=True)
    for k, idx in enumerate(np.array_split(np.arange(n_s1), n_parts)):
        df[df["s1_id"].isin([f"S1-{i}" for i in idx])].to_parquet(folder / f"part-{k:05d}.parquet", index=False)
    return df


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Paths, pd.DataFrame]:
    """Synthetic features_train trained with a row cap below the data size."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    monkeypatch.setattr(config, "TRAIN_MAX_ROWS", 2000)
    monkeypatch.setattr(config, "LGBM_NUM_THREADS", 2)
    paths = Paths(tmp_path / "data", tmp_path / "art", tmp_path / "out")
    df = _write_parts(paths.artifacts_dir / "features_train", 400, 8, 3, seed=1, label=True)
    model.run_stage(paths, "train")
    return paths, df


def test_train_oof_covers_every_pair(trained: tuple[Paths, pd.DataFrame]) -> None:
    """OOF has every pair in file order, one fold per S1, informative probabilities."""
    paths, df = trained
    oof = pd.read_parquet(paths.artifacts_dir / "oof_train.parquet")
    assert set(contracts.OOF_COLUMNS) <= set(oof.columns)
    assert oof[["s1_id", "cand_id"]].equals(df[["s1_id", "cand_id"]])
    assert (oof["label"].to_numpy() == df["label"].to_numpy()).all()
    assert (oof.groupby("s1_id")["fold"].nunique() == 1).all()
    assert sorted(oof["fold"].unique()) == list(range(config.N_FOLDS))
    assert oof["prob"].dtype == np.float32 and oof["prob"].between(0, 1).all()
    assert oof.loc[oof["label"] == 1, "prob"].mean() > oof.loc[oof["label"] == 0, "prob"].mean() + 0.3
    for k in range(config.N_FOLDS):
        assert (paths.artifacts_dir / model.MODEL_DIR / f"model_fold{k}.txt").exists()
    imp = pd.read_csv(paths.artifacts_dir / model.IMPORTANCE_FILE, sep="\t")
    assert imp["feature"].iloc[-1] == "noise" and set(imp["feature"]) == {"sim_a", "sim_b", "noise"}
    assert "OOF AUC" in (paths.artifacts_dir.parent / "shared" / "experiments.tsv").read_text(encoding="utf-8")


def test_train_is_deterministic(trained: tuple[Paths, pd.DataFrame]) -> None:
    """Retraining on the same parts reproduces the OOF probabilities exactly."""
    paths, _ = trained
    first = pd.read_parquet(paths.artifacts_dir / "oof_train.parquet")["prob"].to_numpy()
    model.run_stage(paths, "train")
    again = pd.read_parquet(paths.artifacts_dir / "oof_train.parquet")["prob"].to_numpy()
    assert (first == again).all()


def test_predict_is_mean_of_fold_models(trained: tuple[Paths, pd.DataFrame]) -> None:
    """pred_test.parquet = average of the fold models, one row per test pair in order."""
    paths, _ = trained
    test = _write_parts(paths.artifacts_dir / "features_test", 50, 6, 2, seed=2, label=False)
    model.run_stage(paths, "test")
    pred = pd.read_parquet(paths.artifacts_dir / "pred_test.parquet")
    assert list(pred.columns) == list(contracts.PRED_COLUMNS)
    assert pred[["s1_id", "cand_id"]].equals(test[["s1_id", "cand_id"]])
    x = test[["sim_a", "sim_b", "noise"]].to_numpy()
    boosters = [lgb.Booster(model_file=paths.artifacts_dir / model.MODEL_DIR / f"model_fold{k}.txt")
                for k in range(config.N_FOLDS)]
    expected = np.mean([b.predict(x) for b in boosters], axis=0)
    np.testing.assert_allclose(pred["prob"].to_numpy(), expected, rtol=1e-5)


def test_read_groups_spans_parts(tmp_path: Path) -> None:
    """S1 codes are consecutive runs, continuing across parts."""
    df = _write_parts(tmp_path / "f", 5, 3, 2, seed=0, label=True)
    codes, labels = model.read_groups(model._parts(tmp_path / "f"), with_label=True)
    assert codes.tolist() == np.repeat(np.arange(5), 3).tolist()
    assert labels is not None and (labels == df["label"].to_numpy()).all()


def test_predict_missing_feature_raises(trained: tuple[Paths, pd.DataFrame]) -> None:
    """A test part without a trained feature is rejected, naming it."""
    paths, _ = trained
    folder = paths.artifacts_dir / "features_test"
    _write_parts(folder, 5, 2, 1, seed=3, label=False)
    part = folder / "part-00000.parquet"
    pd.read_parquet(part).drop(columns=["sim_b"]).to_parquet(part, index=False)
    with pytest.raises(ValueError, match="sim_b"):
        model.run_stage(paths, "test")


def test_train_saves_feature_list(trained: tuple[Paths, pd.DataFrame]) -> None:
    """models/feature_list.json holds the exact trained feature columns in order."""
    paths, _ = trained
    saved = json.loads((paths.artifacts_dir / model.MODEL_DIR / model.FEATURE_LIST_FILE).read_text(encoding="utf-8"))
    assert saved == ["sim_a", "sim_b", "noise"]


def test_predict_extra_feature_raises(trained: tuple[Paths, pd.DataFrame]) -> None:
    """Test features with a column the models never saw (e.g. v2 parts, v1 models) are refused."""
    paths, _ = trained
    folder = paths.artifacts_dir / "features_test"
    _write_parts(folder, 5, 2, 1, seed=3, label=False)
    part = folder / "part-00000.parquet"
    pd.read_parquet(part).assign(unit_conflict=np.float32(0)).to_parquet(part, index=False)
    with pytest.raises(ValueError, match="unit_conflict"):
        model.run_stage(paths, "test")


def test_predict_feature_list_mismatch_raises(trained: tuple[Paths, pd.DataFrame]) -> None:
    """A feature_list.json that disagrees with the saved models is refused."""
    paths, _ = trained
    _write_parts(paths.artifacts_dir / "features_test", 5, 2, 1, seed=3, label=False)
    (paths.artifacts_dir / model.MODEL_DIR / model.FEATURE_LIST_FILE).write_text('["sim_a"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="feature_list.json"):
        model.run_stage(paths, "test")


def test_predict_fold0_only(trained: tuple[Paths, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> None:
    """PREDICT_MODELS=fold0 predicts with fold 0's model alone; PREDICT_THREADS is accepted."""
    paths, _ = trained
    test = _write_parts(paths.artifacts_dir / "features_test", 50, 6, 2, seed=2, label=False)
    monkeypatch.setattr(config, "PREDICT_MODELS", "fold0")
    monkeypatch.setattr(config, "PREDICT_THREADS", 1)
    model.run_stage(paths, "test")
    pred = pd.read_parquet(paths.artifacts_dir / "pred_test.parquet")
    b0 = lgb.Booster(model_file=paths.artifacts_dir / model.MODEL_DIR / "model_fold0.txt")
    expected = np.asarray(b0.predict(test[["sim_a", "sim_b", "noise"]].to_numpy()))
    np.testing.assert_allclose(pred["prob"].to_numpy(), expected, rtol=1e-5)
