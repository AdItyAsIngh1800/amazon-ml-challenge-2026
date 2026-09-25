"""Smoke tests for config path resolution and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src import config
from src.logging_utils import setup_logging


def test_paths_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI argument beats env var, env var beats the repo default."""
    monkeypatch.setenv("BER_DATA_DIR", str(tmp_path / "env"))
    monkeypatch.delenv("BER_OUTPUT_DIR", raising=False)
    paths = config.get_paths(artifacts_dir=tmp_path / "cli")
    assert paths.data_dir == (tmp_path / "env").resolve()
    assert paths.artifacts_dir == (tmp_path / "cli").resolve()
    assert paths.output_dir == config.REPO_ROOT / "output"


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    """A run log file is created and receives messages."""
    setup_logging("info", tmp_path)
    logging.getLogger("x").info("hello")
    logs = list(tmp_path.glob("run_*.log"))
    assert len(logs) == 1 and "| INFO | x | hello" in logs[0].read_text(encoding="utf-8")


def test_setup_logging_rejects_bad_level(tmp_path: Path) -> None:
    """Unknown level names raise ValueError."""
    with pytest.raises(ValueError):
        setup_logging("LOUD", tmp_path)
