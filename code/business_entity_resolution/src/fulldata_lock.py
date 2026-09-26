"""Machine-wide lock so only one full-data run happens at a time.

Several lane worktrees share one 16 GB Mac. Any entry point that reads the
full dataset (``config.FULL_DATA_DIR``) holds ``<SHARED_ARTIFACTS_DIR>/.fulldata.lock``
for its whole run. Dev-sample and synthetic data dirs never lock.

The lock file is JSON: pid, lane (checkout folder), command, start time. It is
removed on normal exit, on exceptions (incl. Ctrl-C) and on SIGTERM. A lock
whose PID is no longer alive (e.g. after SIGKILL or a power loss) is stale and
is cleared automatically with a WARNING.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import FrameType

import psutil

from src import config

logger = logging.getLogger(__name__)

LOCK_NAME = ".fulldata.lock"


def is_full_data(data_dir: Path) -> bool:
    """True if ``data_dir`` is the full dataset folder (``config.FULL_DATA_DIR``)."""
    return data_dir.expanduser().resolve() == config.FULL_DATA_DIR


def _read_holder(lock: Path) -> dict[str, object] | None:
    """Parsed lock contents, or None if missing or unreadable."""
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _acquire(lock: Path, command: str) -> None:
    """Create the lock atomically, clearing a stale one once.

    Raises:
        RuntimeError: If a live process holds the lock.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "lane": config.REPO_ROOT.name, "command": command,
            "start": datetime.now().isoformat(timespec="seconds")}
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = _read_holder(lock) or {}
            pid = holder.get("pid")
            if isinstance(pid, int) and psutil.pid_exists(pid):
                raise RuntimeError(
                    f"Full-data lock {lock} is held by PID {pid} (lane {holder.get('lane')}, "
                    f"started {holder.get('start')}): {holder.get('command')}. "
                    "Wait for it to finish; only one full-data run at a time."
                ) from None
            logger.warning("Clearing stale full-data lock %s (holder %s is not running)", lock, holder)
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(info, f)
        logger.info("Acquired full-data lock %s", lock)
        return
    raise RuntimeError(f"Could not acquire full-data lock {lock}")


def _release(lock: Path) -> None:
    """Remove the lock if this process holds it."""
    holder = _read_holder(lock)
    if holder and holder.get("pid") == os.getpid():
        lock.unlink(missing_ok=True)
        logger.info("Released full-data lock %s", lock)


@contextmanager
def fulldata_lock(data_dir: Path, command: str, lock_dir: Path | None = None) -> Iterator[bool]:
    """Hold the full-data lock while the block runs, if ``data_dir`` is the full dataset.

    Args:
        data_dir: The run's ``--data-dir``.
        command: Command line recorded in the lock for other lanes to see.
        lock_dir: Folder for the lock; ``config.SHARED_ARTIFACTS_DIR`` if None.

    Yields:
        True if the lock is held, False for non-full data (no lock taken).

    Raises:
        RuntimeError: If another live process holds the lock.
    """
    if not is_full_data(data_dir):
        yield False
        return
    lock = (lock_dir or config.SHARED_ARTIFACTS_DIR) / LOCK_NAME
    _acquire(lock, command)
    in_main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if in_main else None

    def _on_sigterm(signum: int, frame: FrameType | None) -> None:
        """Turn SIGTERM into SystemExit so the finally block releases the lock."""
        raise SystemExit(128 + signum)

    if in_main:
        signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        yield True
    finally:
        if in_main:
            signal.signal(signal.SIGTERM, previous)
        _release(lock)
