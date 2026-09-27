"""Data contracts: the columns each stage artifact must contain (see CLAUDE.md).

Stages validate their inputs against these with ``io_utils.require_columns``.
Changing a contract needs the lead's approval.
"""

from __future__ import annotations

# records_{split}.parquet — one row per S1/S2/S3 record.
# name_translit / addr_translit: raw text with Indic scripts romanised
# (transliterate.py); name_norm / addr_norm are built from them.
# name_key: phonetic key of name_norm (transliterate.phonetic_key).
RECORDS_COLUMNS: tuple[str, ...] = (
    "entity_id", "source", "country", "name_raw", "addr_raw",
    "name_norm", "name_core", "legal_suffix", "name_acronym", "addr_norm",
    "postal_tokens", "num_tokens", "landmark_flag", "name_empty", "addr_empty",
    "name_translit", "addr_translit", "name_key",
)

# candidates_{split}.parquet — one row per (S1, S2/S3 candidate) pair.
# rrf_score: reciprocal rank fusion of the pass ranks, the value the cap ranks by.
CANDIDATES_COLUMNS: tuple[str, ...] = (
    "s1_id", "cand_id", "cand_source", "country_match",
    "pass_A_score", "pass_A_rank", "pass_B_score", "pass_B_rank",
    "pass_C_score", "pass_C_rank", "pass_F_score", "pass_F_rank",
    "n_passes", "best_block_score", "rrf_score", "rev_n_s1", "rev_rank", "rev_gap",
)

# features_{split}/part-*.parquet — key columns; feature columns are float32,
# plus "label" on the train split.
FEATURES_KEY_COLUMNS: tuple[str, ...] = ("s1_id", "cand_id")

# oof_train / pred_test: until the model exists, the rule baseline may write a
# "score" column instead of "prob" (config.DECIDE_SCORE_COLUMN selects it).
OOF_COLUMNS: tuple[str, ...] = ("s1_id", "cand_id", "fold", "prob", "label")
PRED_COLUMNS: tuple[str, ...] = ("s1_id", "cand_id", "prob")
DECISION_CONFIG_KEYS: tuple[str, ...] = ("t", "t_empty", "one_owner", "score_column", "train_f05")
# Optional decision_config keys (decide v1); absent = the v0 global-threshold rule:
#   method ("threshold" | "per_source" | "expected_f05"), t_s2 / t_s3 (per_source;
#   "t" then holds the global-threshold start point), one_owner_auto (bool),
#   f05_by_one_owner ({"true": F, "false": F}). expected_f05 ignores t / t_empty.
# decide v2: method "conditional_extra" adds t_conf / t_extra; one_owner_delta
#   (float) makes one_owner soft (absent = hard one-owner rule).
DECISION_CONFIG_OPTIONAL_KEYS: tuple[str, ...] = ("method", "t_s2", "t_s3", "one_owner_auto", "f05_by_one_owner",
                                                  "t_conf", "t_extra", "one_owner_delta")
