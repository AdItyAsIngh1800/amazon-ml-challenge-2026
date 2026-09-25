"""Logging setup and per-stage runtime / peak-memory tracking."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

from src import config

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_GB = 1024**3

logger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_dir: Path) -> None:
    """Configure the root logger to write to the console and a run log file.

    Call once per process (from ``run_pipeline``). Calling again replaces the
    previous handlers instead of duplicating them.

    Args:
        log_level: Level name such as ``"DEBUG"`` or ``"INFO"`` (case-insensitive).
        log_dir: Folder for ``run_<timestamp>.log``; created if missing.

    Raises:
        ValueError: If ``log_level`` is not a valid logging level name.
    """
    level = logging.getLevelNamesMapping().get(log_level.upper())
    if level is None:
        raise ValueError(f"Unknown log level: {log_level!r}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("Logging to %s", log_file)


@dataclass
class StageStats:
    """Runtime and peak resident memory of one stage (filled in on exit)."""

    runtime_s: float = 0.0
    peak_rss_gb: float = 0.0


@contextmanager
def track_stage(name: str, interval_s: float = 0.2) -> Iterator[StageStats]:
    """Log a stage's start, end, runtime and peak RSS.

    Peak RSS is measured with psutil by sampling this process every
    ``interval_s`` seconds in a background thread, which behaves the same on
    macOS, Linux and Windows (unlike ``resource.ru_maxrss``, whose unit differs
    by OS and which only reports the lifetime peak). Allocations shorter than
    ``interval_s`` can be missed. Logs a WARNING above
    ``config.PEAK_RAM_TARGET_GB``.

    Args:
        name: Stage name used in log messages.
        interval_s: Sampling interval in seconds.

    Yields:
        A ``StageStats`` that is populated when the block exits.
    """
    proc = psutil.Process()
    peak = proc.memory_info().rss
    stop = threading.Event()

    def _sample() -> None:
        """Record the max RSS until ``stop`` is set."""
        nonlocal peak
        while not stop.wait(interval_s):
            peak = max(peak, proc.memory_info().rss)

    stats = StageStats()
    sampler = threading.Thread(target=_sample, name=f"rss-{name}", daemon=True)
    logger.info("Stage %s: start", name)
    start = time.perf_counter()
    sampler.start()
    status = "FAILED"
    try:
        yield stats
        status = "end"
    finally:
        stop.set()
        sampler.join()
        peak = max(peak, proc.memory_info().rss)
        stats.runtime_s = time.perf_counter() - start
        stats.peak_rss_gb = peak / _GB
        logger.info(
            "Stage %s: %s | runtime %.1fs | peak RSS %.2f GB",
            name, status, stats.runtime_s, stats.peak_rss_gb,
        )
        if stats.peak_rss_gb > config.PEAK_RAM_TARGET_GB:
            logger.warning(
                "Stage %s: peak RSS %.2f GB exceeds target %.1f GB",
                name, stats.peak_rss_gb, config.PEAK_RAM_TARGET_GB,
            )
