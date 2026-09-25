"""Stage `block`: candidate generation (Passes A, B, C, F) and reverse features.

Reads:  <artifacts>/records_{split}.parquet (contracts.RECORDS_COLUMNS)
Writes: <artifacts>/candidates_{split}.parquet (contracts.CANDIDATES_COLUMNS)
Chunked top-K: config.BLOCK_CHUNK_S1_ROWS S1 rows per chunk, config.BLOCK_TOP_K
per pass, capped at config.MAX_CANDIDATES_PER_S1.
"""

from __future__ import annotations

import logging

from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "blocking.py (lane block)"


def run_stage(paths: Paths, split: str) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet.

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.

    Raises:
        NotImplementedError: Until blocking.py (lane block) implements this stage.
    """
    raise NotImplementedError(f"{__name__}.run_stage ({split}) is not implemented yet; owner: {OWNER}")
