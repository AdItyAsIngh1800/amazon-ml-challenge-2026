"""Tests for the run_pipeline skeleton: stage selection, skipping, guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config, run_pipeline
from src.config import Paths


def _paths(tmp_path: Path, data: str = "data") -> Paths:
    """Paths rooted in a temp folder."""
    return Paths(tmp_path / data, tmp_path / "artifacts", tmp_path / "output")


def test_all_stages_per_split() -> None:
    """'all' expands to the split-specific stage list."""
    train = [s.name for s in run_pipeline.stages_for("all", "train")]
    test = [s.name for s in run_pipeline.stages_for("all", "test")]
    assert train == ["prep", "block", "feat", "train", "decide"]
    assert test == ["prep", "block", "feat", "predict", "write"]


def test_stage_wrong_split_raises() -> None:
    """Training on the test split is rejected."""
    with pytest.raises(ValueError, match="train"):
        run_pipeline.stages_for("train", "test")


def test_stages_wired_to_owner_modules() -> None:
    """Each stage calls run_stage from its owner's module."""
    from src import blocking, decide, features, model, normalize

    expected = {"prep": normalize, "block": blocking, "feat": features, "train": model,
                "predict": model, "decide": decide, "write": decide}
    for name, module in expected.items():
        assert run_pipeline.STAGES[name].run is module.run_stage
        assert run_pipeline.STAGES[name].owner == module.OWNER


@pytest.mark.parametrize(("stage", "split", "owner"), [
    ("feat", "test", "features.py"), ("train", "train", "model.py"),
    ("predict", "test", "model.py"), ("decide", "train", "decide.py"), ("write", "test", "decide.py"),
])
def test_unimplemented_stages_name_owner(tmp_path: Path, stage: str, split: str, owner: str) -> None:
    """Stub stages raise NotImplementedError naming the owning module."""
    with pytest.raises(NotImplementedError, match=owner):
        run_pipeline.run(stage, split, _paths(tmp_path))


def test_existing_outputs_are_skipped_unless_force(tmp_path: Path) -> None:
    """prep is skipped when records_train.parquet exists; --force reruns it."""
    paths = _paths(tmp_path)
    paths.artifacts_dir.mkdir(parents=True)
    (paths.artifacts_dir / "records_train.parquet").touch()
    run_pipeline.run("prep", "train", paths)  # skipped, no error
    with pytest.raises(ValueError, match="Parquet"):  # prep skipped, block reads the empty file
        run_pipeline.run("all", "train", paths)
    with pytest.raises(FileNotFoundError):  # prep really runs; no data in tmp
        run_pipeline.run("prep", "train", paths, force=True)


def test_artifacts_tied_to_data_dir(tmp_path: Path) -> None:
    """Reusing an artifacts folder with another data dir is refused."""
    run_pipeline._check_run_meta(_paths(tmp_path, "dev_sample"))
    with pytest.raises(ValueError, match="different --artifacts-dir"):
        run_pipeline._check_run_meta(_paths(tmp_path, "dataset"))


def test_cli_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--block-top-k sets config.BLOCK_TOP_K before stages run."""
    monkeypatch.setattr(config, "BLOCK_TOP_K", config.BLOCK_TOP_K)  # restored after test
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "records_train.parquet").touch()
    run_pipeline.main(["--stage", "prep", "--split", "train", "--data-dir", str(tmp_path / "d"),
                       "--artifacts-dir", str(art), "--out-dir", str(tmp_path / "o"), "--block-top-k", "7"])
    assert config.BLOCK_TOP_K == 7
