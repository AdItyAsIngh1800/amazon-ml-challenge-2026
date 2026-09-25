"""Stages `decide` (split=train) and `write` (split=test): decision layer and outputs.

decide: oof_train.parquet -> <artifacts>/decision_config.json (t, t_empty,
        one_owner), grid-searched for macro F0.5 on all train OOF at once.
write:  pred_test.parquet + decision_config.json + candidates_test.parquet ->
        <out>/matching_results.tsv and <out>/candidate_pairs.tsv via
        io_utils.write_id_list_tsv, one row per test S1.
"""

from __future__ import annotations

import logging

from src.config import Paths

logger = logging.getLogger(__name__)

OWNER = "decide.py (lane A, lead)"


def run_stage(paths: Paths, split: str) -> None:
    """Tune thresholds (split=train) or write the submission TSVs (split=test).

    Args:
        paths: Resolved run directories.
        split: ``"train"`` or ``"test"``.

    Raises:
        NotImplementedError: Until decide.py (lane A, lead) implements this stage.
    """
    raise NotImplementedError(f"{__name__}.run_stage ({split}) is not implemented yet; owner: {OWNER}")
