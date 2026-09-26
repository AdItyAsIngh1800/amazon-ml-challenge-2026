"""Central configuration: paths, seeds, chunk sizes and K values.

Paths are never hard-coded to a machine. Each directory is resolved in this
order: explicit argument (from the CLI) > environment variable > default
relative to the repository root.

Environment variables:
    BER_DATA_DIR: folder containing ``train/`` and ``test/`` TSV folders.
    BER_ARTIFACTS_DIR: folder for parquet artifacts, models and logs.
    BER_OUTPUT_DIR: folder for the two submission TSVs.

The numeric values below are defaults sized for a 16 GB MacBook Air
(pipeline peak RAM target 10 GB); run_pipeline exposes CLI flags to
override them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# code/business_entity_resolution/src/config.py -> repo root is parents[3].
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def _main_checkout(root: Path) -> Path:
    """Main working tree when ``root`` is a git worktree (its ``.git`` is a file), else ``root``.

    A worktree's ``.git`` file reads ``gitdir: <main>/.git/worktrees/<name>``.
    """
    git = root / ".git"
    if git.is_file():
        text = git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            if gitdir.parent.name == "worktrees":
                return gitdir.parent.parent.parent
    return root


# Shared across all lane worktrees: the full dataset, the full-data lock and
# the experiment log live in the main checkout.
MAIN_ROOT: Path = _main_checkout(REPO_ROOT)
SHARED_ARTIFACTS_DIR: Path = MAIN_ROOT / "artifacts"
FULL_DATA_DIR: Path = Path(os.environ.get("BER_FULL_DATA_DIR", str(MAIN_ROOT / "dataset"))).expanduser().resolve()

SEED: int = 42
PEAK_RAM_TARGET_GB: float = 10.0

# Dtypes: every score/feature is float32; internal ID indices are int32.
FLOAT_DTYPE = np.float32
INDEX_DTYPE = np.int32

# Blocking (owner: blocking.py).
BLOCK_CHUNK_S1_ROWS: int = 10_000
BLOCK_TOP_K: int | None = None  # overrides every pass's K; None = each pass's own
MAX_CANDIDATES_PER_S1: int = 50
RECALL_K_VALUES: tuple[int, ...] = (5, 10, 20, 50)
PASS_C_MIN_NAME_COSINE: float = 0.2

# Features (owner: features.py): candidate pairs per parquet part.
FEATURE_CHUNK_PAIRS: int = 300_000
RAPIDFUZZ_WORKERS: int = -1  # all cores, for rapidfuzz.process.cdist

N_FOLDS: int = 5
# Model (owner: model.py): training pairs per fold (whole S1 groups sampled).
TRAIN_MAX_ROWS: int = 10_000_000

# Decision layer (owner: decide.py). "prob" reads model OOF / test predictions;
# "score" reads a rule-baseline score column from the same files until the
# model exists (run_pipeline --decide-score-column).
DECIDE_SCORE_COLUMN: str = "prob"
DECIDE_ONE_OWNER: bool = True  # EDA E3: 0 matched IDs shared between S1 lists
# Every threshold grid (t, t_empty, t_s2/t_s3) = this many quantiles of the decided
# score column (adapts to rrf_score ~0.017-0.18) UNION a fixed DECIDE_GRID_STEP grid
# over (0, 1) (probabilities: near-0 bulk leaves no quantiles in 0.02-0.99, PR #20).
DECIDE_GRID_QUANTILES: int = 200
DECIDE_GRID_STEP: float = 0.005
# Decision rule tuned by decide (run_pipeline --decide-method):
#   "threshold"    one t for all candidates (+ t_empty gate)
#   "per_source"   t_s2 / t_s3 by candidate ID prefix, coordinate descent from the global t
#   "expected_f05" per S1, the probability-sorted prefix with the highest expected F0.5
DECIDE_METHOD: str = "threshold"
DECIDE_CD_ROUNDS: int = 5  # max coordinate-descent rounds for "per_source"
# True: tune with and without the one-owner rule, keep the better (--decide-one-owner-auto).
DECIDE_ONE_OWNER_AUTO: bool = False
# Pairs per chunk in expected_f05 selection (chunks end on S1 boundaries).
DECIDE_EXPECTED_CHUNK_PAIRS: int = 4_000_000
# write (test) reads this decision_config.json instead of <artifacts>/decision_config.json
# when set (run_pipeline --decision-config), e.g. one tuned on another artifacts dir.
DECISION_CONFIG_PATH: Path | None = None


def _performance_cores() -> int:
    """Number of performance cores on macOS, else ``os.cpu_count()`` (min 1)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return max(1, int(out.stdout.strip()))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return os.cpu_count() or 1


LGBM_NUM_THREADS: int = _performance_cores()


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
