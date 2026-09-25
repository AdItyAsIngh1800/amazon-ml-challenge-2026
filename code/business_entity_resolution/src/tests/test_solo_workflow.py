"""Tests for the full-data lock, worktree detection and the experiment log."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import config, experiment_log, run_pipeline
from src.fulldata_lock import LOCK_NAME, fulldata_lock


@pytest.fixture
def full_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake full dataset dir and shared artifacts dir, wired into config."""
    data, shared = tmp_path / "dataset", tmp_path / "shared"
    data.mkdir()
    monkeypatch.setattr(config, "FULL_DATA_DIR", data.resolve())
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", shared)
    return data, shared


def _dead_pid() -> int:
    """PID of a process that has already exited."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_main_checkout_from_worktree_git_file(tmp_path: Path) -> None:
    """A worktree's .git file points back to the main checkout."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {tmp_path / 'main' / '.git' / 'worktrees' / 'wt'}\n", encoding="utf-8")
    assert config._main_checkout(wt) == tmp_path / "main"
    assert config._main_checkout(tmp_path) == tmp_path  # no .git file -> itself


def test_dev_data_never_locks(full_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Non-full data dirs yield False and create no lock."""
    _, shared = full_dirs
    with fulldata_lock(tmp_path / "dev_sample", "cmd") as held:
        assert held is False
    assert not (shared / LOCK_NAME).exists()


def test_full_data_lock_written_and_removed(full_dirs: tuple[Path, Path]) -> None:
    """Lock holds pid/lane/command/start during the run and is gone after."""
    data, shared = full_dirs
    with fulldata_lock(data, "src.eda --data-dir dataset") as held:
        assert held
        info = json.loads((shared / LOCK_NAME).read_text(encoding="utf-8"))
        assert info["pid"] == os.getpid() and info["command"] == "src.eda --data-dir dataset"
        assert info["lane"] == config.REPO_ROOT.name and info["start"]
    assert not (shared / LOCK_NAME).exists()


def test_live_lock_refused_with_holder(full_dirs: tuple[Path, Path]) -> None:
    """A lock held by a live process blocks and names the holder; it is left intact."""
    data, shared = full_dirs
    shared.mkdir()
    holder = {"pid": os.getppid(), "lane": "amazon_ml_challenge_block", "command": "src.run_pipeline x", "start": "t"}
    (shared / LOCK_NAME).write_text(json.dumps(holder), encoding="utf-8")
    with pytest.raises(RuntimeError, match="amazon_ml_challenge_block"), fulldata_lock(data, "cmd"):
        pass
    assert json.loads((shared / LOCK_NAME).read_text(encoding="utf-8"))["pid"] == os.getppid()


def test_stale_lock_cleared(full_dirs: tuple[Path, Path]) -> None:
    """A lock whose PID is dead is replaced by ours."""
    data, shared = full_dirs
    shared.mkdir()
    (shared / LOCK_NAME).write_text(json.dumps({"pid": _dead_pid()}), encoding="utf-8")
    with fulldata_lock(data, "cmd"):
        assert json.loads((shared / LOCK_NAME).read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not (shared / LOCK_NAME).exists()


def test_lock_removed_on_crash(full_dirs: tuple[Path, Path]) -> None:
    """An exception inside the run still releases the lock."""
    data, shared = full_dirs
    with pytest.raises(ValueError), fulldata_lock(data, "cmd"):
        raise ValueError("boom")
    assert not (shared / LOCK_NAME).exists()


def test_run_pipeline_refuses_when_locked(full_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """run_pipeline on the full dataset stops before any stage while another run holds the lock."""
    data, shared = full_dirs
    shared.mkdir()
    (shared / LOCK_NAME).write_text(json.dumps({"pid": os.getppid(), "lane": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="held by PID"):
        run_pipeline.main(["--stage", "prep", "--split", "train", "--data-dir", str(data),
                           "--artifacts-dir", str(tmp_path / "art"), "--out-dir", str(tmp_path / "out")])
    assert not (tmp_path / "art" / "run_meta.json").exists()


def test_experiment_log_appends_rows(tmp_path: Path) -> None:
    """Header once, rows appended, None -> empty, tabs/newlines flattened."""
    log = tmp_path / "experiments.tsv"
    experiment_log.log_experiment("A-001", "A", "decide v0", f05_overall=0.81234, accepted=True, path=log)
    experiment_log.log_experiment("block-001", "block", "pass A\tK=50\n", pair_recall=0.7, notes="dev", path=log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == list(experiment_log.EXPERIMENT_COLUMNS)
    r1, r2 = (dict(zip(experiment_log.EXPERIMENT_COLUMNS, ln.split("\t"), strict=True)) for ln in lines[1:])
    assert (r1["id"], r1["f05_overall"], r1["accepted"], r1["pair_recall"]) == ("A-001", "0.8123", "Y", "")
    assert (r2["change"], r2["pair_recall"], r2["accepted"], r2["notes"]) == ("pass A K=50", "0.7000", "", "dev")


def test_experiment_log_rejects_foreign_header(tmp_path: Path) -> None:
    """Appending to a file with another header is refused."""
    log = tmp_path / "experiments.tsv"
    log.write_text("something\telse\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected header"):
        experiment_log.log_experiment("x", "A", "y", path=log)
