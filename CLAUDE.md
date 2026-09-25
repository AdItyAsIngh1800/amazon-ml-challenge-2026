# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

Source of truth: `plan/final_plan.txt`. Implement it; don't re-plan. Ambiguous or conflicts with data -> ask the lead.

## Problem
For every Source 1 (S1) record, output all Source 2 / Source 3 records that
are the same real-world business. S1 is deduplicated; an S1 matches 0..n
records. Columns: entity_id, business_name, business_address, country.
Train = US + India. Test adds France (never seen in train), ~1.7M S1.

## Metric (F0.5 per S1, macro-averaged over ALL S1)
- truth empty & pred empty -> 1.0; exactly one of them empty -> 0.0
- else F0.5 = 1.25PR / (0.25P + R); 0 if P = R = 0
Precision weighs 2x recall. Singletons (5.6% of train S1) are a full point each.

## Outputs (tab-separated, UTF-8, "\n" endings, exact headers)
- `matching_results.tsv`: `source1_entity_id` | `matched_entity_ids`
- `candidate_pairs.tsv`: `source1_entity_id` | `candidate_entity_ids`
One row per test S1 (France included). Comma-joined IDs, no spaces, no
duplicates, S2-/S3- only, empty allowed. Matches must be a subset of
candidates. candidate_pairs = exact set the model scores.
Always run `utils/validate_submission.py` on outputs.

## Data facts (Checkpoint 1)
- No tabs/newlines inside fields; every row has 4 fields. No BOM, no CR.
- ~830 lines have CSV-escaped quotes (`"""ehpad Club SAS"`); QUOTE_NONE keeps the
  literal `"`, normalisation strips it.
- Names never empty; ~3.4% of S2/S3 addresses empty; some names are "NA".
- One-owner rule holds in train: every matched S2/S3 ID is in exactly one S1 list.
- Test S2+S3 : S1 = 5.75 vs train 4.68 -> test is denser (risk R3).

## Repository layout
```
code/business_entity_resolution/
  src/  config.py io_utils.py eda.py make_dev_sample.py normalize.py
        blocking.py features.py model.py decide.py evaluate.py
        logging_utils.py contracts.py run_pipeline.py fulldata_lock.py
        experiment_log.py tests/
  README.md  requirements.txt  pyproject.toml
artifacts/ (gitignored; dev_sample/, .fulldata.lock, experiments.tsv)
artifacts_<lane>/ (lane outputs)   output/ (TSVs gitignored)
```

## TSV reading — ONLY via `io_utils.read_source`
`sep="\t", dtype=str, keep_default_na=False, na_filter=False,
quoting=csv.QUOTE_NONE, encoding="utf-8"`, then assert rows == lines - 1.

## Data contracts (parquet, {split} = train | test)
- `records_{split}.parquet`: entity_id, source (S1/S2/S3), country, name_raw,
  addr_raw, name_norm, name_core, legal_suffix, name_acronym, addr_norm,
  postal_tokens (list), num_tokens (list), landmark_flag, name_empty, addr_empty
- `candidates_{split}.parquet`: s1_id, cand_id, cand_source, country_match
  (bool agreement flag), pass_{A,B,C,F}_score, pass_{A,B,C,F}_rank, n_passes,
  best_block_score, rev_n_s1, rev_rank, rev_gap
- `features_{split}/part-*.parquet`: s1_id, cand_id, float32 features, label (train)
- `oof_train.parquet`: s1_id, cand_id, fold, prob, label
- `pred_test.parquet`: s1_id, cand_id, prob
- `decision_config.json`: t, t_empty, one_owner, optional per-source
  thresholds. Tuned on train, applied unchanged to test.
Contract changes go in `src/contracts.py`; the PR title starts with [CONTRACT].

## Runner (single entry point, run from code/business_entity_resolution/)
```
python -m src.run_pipeline --stage {prep,block,feat,train,predict,decide,write,all}
  --split {train,test} --data-dir PATH --out-dir PATH [--artifacts-dir PATH]
  [--force] [--log-level INFO] [--block-top-k N ... config overrides]
```
- train split `all` = prep, block, feat, train, decide; test split `all` =
  prep, block, feat, predict, write. Run train first, then test.
- Each stage skips if its outputs exist unless --force; logs runtime + peak RSS.
- An artifacts dir is tied to the --data-dir that built it (run_meta.json).
  Dev runs: `--data-dir ../../artifacts/dev_sample --artifacts-dir ../../artifacts/dev_run`.
- Stage code is `run_stage(paths, split)` in the owning lane's module; model.py
  (train/predict) and decide.py (decide/write) branch on split.
- Read tunables as `config.NAME` at call time (never `from src.config import
  NAME`) so CLI overrides apply.

## Hard rules
- NO network calls in src/: no external APIs, geocoding, registries, lookups,
  hosted models. Disqualification rule.
- Dataset never leaves this machine: never commit, upload or paste it.
- No country hard-coding, filtering, one-hot or country identity features.
  Country is an open set. `country_match` (agreement flag) is allowed.
- Postal-like tokens = every standalone 5-6 digit number, all countries.
- UTF-8 on every open, "\n" on write. pathlib for all paths.
- multiprocessing only under `if __name__ == "__main__":`.
- MacBook Air 16 GB, peak RAM <= 10 GB. Chunk anything that scales with pairs
  (config.BLOCK_CHUNK_S1_ROWS, config.FEATURE_CHUNK_PAIRS); measure with
  `logging_utils.track_stage`. No full similarity matrix. float32 scores/features,
  int32 ID indices (config.FLOAT_DTYPE, config.INDEX_DTYPE).
- Fixed seeds (config.SEED = 42), deterministic LightGBM, fixed num_threads.
- Models: MIT or Apache 2.0, <= 8B params; record name/license/params.

## Allowed libraries (new deps need lead approval + license check)
pandas (BSD), pyarrow (Apache 2.0), numpy (BSD), scipy (BSD),
scikit-learn (BSD), rapidfuzz (MIT), lightgbm (MIT), pytest (MIT),
mypy (MIT), psutil (BSD-3). Avoid: unidecode (GPL; use unicodedata NFKD), libpostal.

## Code quality standards (every file, every checkpoint)
Docstrings
- Google-style docstrings on every module/class/function incl. private helpers
  (one line OK for small ones). Public functions: Args, Returns, Raises.
- DataFrame/parquet in or out: list required columns, dtypes, what one row is.
- Anything scaling with data size states its memory behaviour
  (e.g. "processes S1 rows in chunks of config.BLOCK_CHUNK_S1_ROWS").
Types
- `from __future__ import annotations` in every module. Full hints on all
  params and returns. Precise types (pd.DataFrame, NDArray[np.float32],
  scipy.sparse.csr_matrix, Path, dict[str, set[str]]). Avoid Any; if
  unavoidable, comment why.
- Stage boundaries validate input columns/dtypes and raise ValueError naming
  the missing/wrong column. No checks inside per-row or per-pair loops.
Logging
- `logger = logging.getLogger(__name__)` in every module. No print() in src/.
- run_pipeline calls `logging_utils.setup_logging` once (--log-level).
- INFO: stage start/end, row counts, runtime, peak memory, key metrics.
- WARNING: anomalies (empty fields, dropped rows, fallbacks). DEBUG: sample
  records only. Never log whole DataFrames or inside per-pair loops.
Definition of done: `pytest` passes AND `mypy` reports 0 errors on src/.
Tests first (TDD) for evaluate.py and io_utils.py.

## Solo lanes (one person runs four Claude Code lanes, reviews and merges)
| Lane | Folder (~/codes/) | Branch | Edits only |
|---|---|---|---|
| A (lead) | amazon_ml_challenge | feat/<task> | decide.py, run_pipeline.py, shared modules, docs |
| norm | amazon_ml_challenge_norm | feat/normalize-* | normalize.py |
| block | amazon_ml_challenge_block | feat/blocking-* | blocking.py |
| feat | amazon_ml_challenge_feat | feat/features-* | features.py, model.py |
- Each lane edits only its own modules and their tests. Shared modules (config,
  io_utils, evaluate, contracts, run_pipeline, eda) change only in lane A.
- Lanes use the dev sample only, outputs in their own folder:
  `--data-dir ~/codes/amazon_ml_challenge/artifacts/dev_sample
  --artifacts-dir ~/codes/amazon_ml_challenge/artifacts_<lane>`.
- Full-data runs are started only by the lead, in a separate terminal with
  caffeinate -i. run_pipeline / eda / make_dev_sample hold
  artifacts/.fulldata.lock while reading the full dataset; a second run is refused.
- One PR per task: tests + mypy pass, dev-sample run shown, metric impact, and a
  row via `experiment_log.log_experiment` (artifacts/experiments.tsv).
- Metric impact = pair recall, % S1 fully covered, avg cands/S1, F0.5 overall /
  singleton / non-singleton / per country, LOCO. Accept only if global F0.5
  +>= 0.002 and LOCO drops <= 0.005. Dev-sample scores overstate precision;
  thresholds come only from full data.
- Never push to main (branch protection unavailable on our plan); the lead
  merges. Branches feat/<module>-<desc>, fix/<module>-<desc>.

## Commands
```
cd code/business_entity_resolution && python -m pytest -q && python -m mypy
```
