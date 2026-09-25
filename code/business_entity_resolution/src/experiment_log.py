"""Append-only experiment log shared by all lanes (plan Section 11).

One TSV row per experiment in ``<SHARED_ARTIFACTS_DIR>/experiments.tsv`` (the
main checkout's ``artifacts/``, so every lane worktree writes to the same log).
Metrics left as None are written as empty cells.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "timestamp", "id", "lane", "change", "pair_recall", "avg_cands_per_s1",
    "f05_overall", "f05_singleton", "f05_non_singleton", "f05_us", "f05_india",
    "loco", "accepted", "notes",
)


def _text(value: str) -> str:
    """Make free text safe for one TSV cell (no tabs or line breaks)."""
    return " ".join(value.split())


def _num(value: float | None) -> str:
    """Format a metric to 4 decimals; None becomes an empty cell."""
    return "" if value is None else f"{value:.4f}"


def log_experiment(
    exp_id: str,
    lane: str,
    change: str,
    *,
    pair_recall: float | None = None,
    avg_cands_per_s1: float | None = None,
    f05_overall: float | None = None,
    f05_singleton: float | None = None,
    f05_non_singleton: float | None = None,
    f05_us: float | None = None,
    f05_india: float | None = None,
    loco: float | None = None,
    accepted: bool | None = None,
    notes: str = "",
    path: Path | None = None,
) -> Path:
    """Append one experiment row, writing the header if the log is new.

    Args:
        exp_id: Experiment id, e.g. ``"block-003"``.
        lane: Lane that ran it (``A``, ``norm``, ``block``, ``feat``).
        change: One-line description of what changed.
        pair_recall: Blocking pair recall.
        avg_cands_per_s1: Average candidates per S1.
        f05_overall: Macro F0.5 over all S1.
        f05_singleton: Macro F0.5 over singleton S1.
        f05_non_singleton: Macro F0.5 over matched S1.
        f05_us: Macro F0.5 over US S1.
        f05_india: Macro F0.5 over India S1.
        loco: Leave-one-country-out score.
        accepted: Accepted per plan Section 8 (Y/N); None if undecided.
        notes: Free text.
        path: Log file; ``<SHARED_ARTIFACTS_DIR>/experiments.tsv`` if None.

    Returns:
        The log file path.

    Raises:
        ValueError: If the existing file has a different header.
    """
    out = path or config.SHARED_ARTIFACTS_DIR / "experiments.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(EXPERIMENT_COLUMNS)
    if out.exists() and out.stat().st_size:
        with out.open(encoding="utf-8") as f:
            existing = f.readline().rstrip("\n")
        if existing != header:
            raise ValueError(f"{out}: unexpected header {existing!r}; expected {header!r}")
        new = False
    else:
        new = True
    row = [
        datetime.now().isoformat(timespec="seconds"), _text(exp_id), _text(lane), _text(change),
        _num(pair_recall), _num(avg_cands_per_s1), _num(f05_overall), _num(f05_singleton),
        _num(f05_non_singleton), _num(f05_us), _num(f05_india), _num(loco),
        "" if accepted is None else ("Y" if accepted else "N"), _text(notes),
    ]
    with out.open("a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write(header + "\n")
        f.write("\t".join(row) + "\n")
    logger.info("Experiment %s logged to %s", exp_id, out)
    return out
