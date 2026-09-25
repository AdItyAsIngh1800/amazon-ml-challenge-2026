"""Stage `feat`: pairwise, group-context and blocking features.

Reads:  <artifacts>/records_{split}.parquet, candidates_{split}.parquet
Writes: <artifacts>/features_{split}/part-*.parquet (contracts.FEATURES_KEY_COLUMNS
        + float32 features, + label on train), config.FEATURE_CHUNK_PAIRS
        pairs per part. No country identity features.
"""

from __future__ import annotations

import logging

from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "features.py (Member 4)"


def run_stage(paths: Paths, split: str) -> None:
    """Build features_{split}/part-*.parquet in chunks.

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.

    Raises:
        NotImplementedError: Until features.py (Member 4) implements this stage.
    """
    raise NotImplementedError(f"{__name__}.run_stage ({split}) is not implemented yet; owner: {OWNER}")
