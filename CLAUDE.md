# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

Single source of truth: `plan/final_plan.txt`. Implement it; do not re-plan.
If the plan is ambiguous or conflicts with the data, stop and ask the lead.

## Problem
For every Source 1 (S1) record, output all Source 2 / Source 3 records that
are the same real-world business. S1 is deduplicated; an S1 matches 0..n
records. Columns: entity_id, business_name, business_address, country.
Train = US + India. Test adds France (never seen in train), ~1.7M S1.

## Metric (F0.5 per S1, macro-averaged over ALL S1)
- truth empty, pred empty -> 1.0
- truth empty, pred non-empty -> 0.0
- truth non-empty, pred empty -> 0.0
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
- ~830 lines (mostly test) contain CSV-escaped quotes (`"""ehpad Club SAS"`).
  We read with QUOTE_NONE, so the literal `"` stays in the value;
  normalisation strips punctuation.
- Names are never empty; ~3.4% of S2/S3 addresses are empty.
- Some businesses are literally named "NA" -> keep_default_na=False.
- One-owner rule holds in train: every matched S2/S3 ID is in exactly one S1 list.
- Test S2+S3 : S1 = 5.75 vs train 4.68 -> test is denser (risk R3).

## Repository layout
```
code/business_entity_resolution/
  src/  config.py io_utils.py eda.py make_dev_sample.py normalize.py
        blocking.py features.py model.py decide.py evaluate.py
        logging_utils.py run_pipeline.py tests/
  README.md  requirements.txt  pyproject.toml
notebooks/   artifacts/ (gitignored)   output/ (TSVs gitignored)
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
- `features_{split}/part-*.parquet`: s1_id, cand_id, float32 features,
  label (train only)
- `oof_train.parquet`: s1_id, cand_id, fold, prob, label
- `pred_test.parquet`: s1_id, cand_id, prob
- `decision_config.json`: t, t_empty, one_owner, optional per-source
  thresholds. Tuned on train, applied unchanged to test.
Nobody changes a contract without telling the lead.

## Runner (single entry point, run from code/business_entity_resolution/)
```
python -m src.run_pipeline --stage {prep,block,feat,train,predict,decide,write,all}
  --split {train,test} --data-dir PATH --out-dir PATH [--force] [--log-level INFO]
```
Each stage skips if its output exists unless --force. Logs runtime + peak RAM.

## Hard rules
- NO network calls in src/: no external APIs, geocoding, registries, lookups,
  hosted models. Disqualification rule.
- Dataset never leaves this machine: never commit, upload or paste it.
- No country hard-coding, filtering, one-hot or country identity features.
  Country is an open set. `country_match` (agreement flag) is allowed.
- Postal-like tokens = every standalone 5-6 digit number, all countries.
- UTF-8 on every open, "\n" on write. pathlib for all paths.
- multiprocessing only under `if __name__ == "__main__":`.
- Whole pipeline < 12 GB RAM. Chunk anything that scales with pairs.
  Never build a full similarity matrix. float32. Integer IDs internally.
- Fixed seeds (config.SEED = 42), deterministic LightGBM, fixed num_threads.
- Models: MIT or Apache 2.0, <= 8B params; record name/license/params.

## Allowed libraries (new deps need lead approval + license check)
pandas (BSD), pyarrow (Apache 2.0), numpy (BSD), scipy (BSD),
scikit-learn (BSD), rapidfuzz (MIT), lightgbm (MIT), pytest (MIT),
mypy (MIT). Avoid: unidecode (GPL; use unicodedata NFKD), libpostal.

## Code quality standards (every file, every checkpoint)
Docstrings
- Google-style docstrings on every module, class and function, incl. private
  helpers (one line is fine for small helpers).
- Public functions document Args, Returns, Raises.
- DataFrame/parquet in or out: list required columns, dtypes, what one row is.
- Anything scaling with data size states its memory behaviour
  (e.g. "processes S1 rows in chunks of config.CHUNK_SIZE").
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
- WARNING: anomalies (empty fields, dropped rows, unexpected countries,
  fallbacks). DEBUG: sample records only. Never log whole DataFrames or
  inside per-pair loops.
Definition of done: `pytest` passes AND `mypy` reports 0 errors on src/.
Tests first (TDD) for evaluate.py and io_utils.py.

## Workflow
- Dev sample first (`artifacts/dev_sample/`, via --data-dir), then full data
  on Colab. Dev-sample scores overstate precision; thresholds come only from
  full data.
- Every change reports metric impact: pair recall, % S1 fully covered, avg
  cands/S1, F0.5 overall/singleton/non-singleton/per country, LOCO.
- Accept a change only if global F0.5 +>= 0.002 and LOCO drops <= 0.005.
- Git: main protected, lead merges. Branches feat/<module>-<desc>,
  fix/<module>-<desc>. Commits "<module>: <what changed>". PR checklist:
  runs on dev sample, tests pass, metric impact stated.

## Commands
```
cd code/business_entity_resolution
python -m pytest -q
python -m mypy
```
