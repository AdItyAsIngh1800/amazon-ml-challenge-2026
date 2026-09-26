"""Stages `train` (split=train) and `predict` (split=test): LightGBM.

train:   features_train/ -> <artifacts>/oof_train.parquet (contracts.OOF_COLUMNS),
         5 fold models, gain importance. GroupKFold(config.N_FOLDS) by s1_id,
         deterministic, seed config.SEED, config.LGBM_NUM_THREADS threads.
predict: features_test/ -> <artifacts>/pred_test.parquet (contracts.PRED_COLUMNS),
         mean of the fold models.
"""

from __future__ import annotations

import logging

from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "model.py (lane feat)"


def run_stage(paths: Paths, split: str) -> None:
    """Train with OOF predictions (split=train) or predict test (split=test).

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.

    Raises:
        NotImplementedError: Until model.py (lane feat) implements this stage.
    """
    raise NotImplementedError(f"{__name__}.run_stage ({split}) is not implemented yet; owner: {OWNER}")
