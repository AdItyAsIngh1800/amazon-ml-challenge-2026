"""Logging setup shared by every pipeline entry point."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


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
    logging.getLogger(__name__).info("Logging to %s", log_file)
