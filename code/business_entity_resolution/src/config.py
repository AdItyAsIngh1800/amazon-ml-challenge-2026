"""Central configuration: paths, seeds, chunk sizes and K values.

Paths are never hard-coded to a machine. Each directory is resolved in this
order: explicit argument (from the CLI) > environment variable > default
relative to the repository root.

Environment variables:
    BER_DATA_DIR: folder containing ``train/`` and ``test/`` TSV folders.
    BER_ARTIFACTS_DIR: folder for parquet artifacts, models and logs.
    BER_OUTPUT_DIR: folder for the two submission TSVs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# code/business_entity_resolution/src/config.py -> repo root is parents[3].
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

SEED: int = 42

# Blocking: S1 rows per chunk in chunked sparse top-K (plan: 5k-20k).
CHUNK_SIZE: int = 10_000
# Features: candidate pairs per parquet part.
FEATURE_CHUNK_SIZE: int = 1_000_000

# Blocking K values (starting points; owner: blocking.py).
TOP_K_PER_PASS: int = 50
MAX_CANDIDATES_PER_S1: int = 50
RECALL_K_VALUES: tuple[int, ...] = (5, 10, 20, 50)
PASS_C_MIN_NAME_COSINE: float = 0.2

N_FOLDS: int = 5
MEMORY_BUDGET_GB: float = 12.0


@dataclass(frozen=True)
class Paths:
    """Resolved directories for one pipeline run.

    Attributes:
        data_dir: Folder containing ``train/`` and ``test/`` TSV folders.
        artifacts_dir: Folder for parquet artifacts, models and logs.
        output_dir: Folder for ``matching_results.tsv`` and
            ``candidate_pairs.tsv``.
    """

    data_dir: Path
    artifacts_dir: Path
    output_dir: Path

    @property
    def log_dir(self) -> Path:
        """Folder for run logs (``<artifacts_dir>/logs``)."""
        return self.artifacts_dir / "logs"


def _resolve(arg: Path | str | None, env_var: str, default: Path) -> Path:
    """Pick the CLI argument, else the environment variable, else the default."""
    value = arg if arg is not None else os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


def get_paths(
    data_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Paths:
    """Resolve run directories from CLI arguments, env vars or defaults.

    Args:
        data_dir: Explicit data folder, usually from ``--data-dir``.
        artifacts_dir: Explicit artifacts folder.
        output_dir: Explicit output folder, usually from ``--out-dir``.

    Returns:
        A ``Paths`` instance. Directories are not created here.
    """
    return Paths(
        data_dir=_resolve(data_dir, "BER_DATA_DIR", REPO_ROOT / "dataset"),
        artifacts_dir=_resolve(artifacts_dir, "BER_ARTIFACTS_DIR", REPO_ROOT / "artifacts"),
        output_dir=_resolve(output_dir, "BER_OUTPUT_DIR", REPO_ROOT / "output"),
    )
