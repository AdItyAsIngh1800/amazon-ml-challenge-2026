"""Smoke tests for config path resolution and logging setup."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from src import config
from src.logging_utils import setup_logging, track_stage


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


def test_track_stage_sees_allocation() -> None:
    """A ~200 MB buffer held during the stage shows up in peak RSS."""
    before = psutil.Process().memory_info().rss / 1024**3
    with track_stage("alloc", interval_s=0.01) as stats:
        buf = np.ones(25_000_000, dtype=np.float64)  # 200 MB, pages touched
        time.sleep(0.05)
        del buf
    assert stats.runtime_s >= 0.05
    assert stats.peak_rss_gb >= before + 0.15


def test_config_defaults() -> None:
    """Hardware-sized defaults requested by the lead."""
    assert config.BLOCK_CHUNK_S1_ROWS == 10_000
    assert config.BLOCK_TOP_K == 20 and config.MAX_CANDIDATES_PER_S1 == 50
    assert config.FEATURE_CHUNK_PAIRS == 300_000
    assert config.FLOAT_DTYPE is np.float32 and config.INDEX_DTYPE is np.int32
    assert config.RAPIDFUZZ_WORKERS == -1
    assert config.LGBM_NUM_THREADS >= 1
