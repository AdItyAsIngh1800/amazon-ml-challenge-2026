# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 (S1) business record, find all Source 2 / Source 3 records that are the
same real-world business. The pipeline produces `output/matching_results.tsv` (final
matches) and `output/candidate_pairs.tsv` (the exact candidate set the model scores).

Python 3.13 (tested on 3.13.13, macOS). Everything runs locally: no network calls, no
external APIs, geocoders or registries, no hosted models. The only model is LightGBM
(MIT licence), trained from scratch on the provided training data.

## Overview

| Stage | Module | What it does | Output (in `--artifacts-dir`) |
|---|---|---|---|
| prep | `normalize.py`, `transliterate.py` | Unicode NFKD normalisation, Indic → Latin transliteration, legal-suffix / abbreviation dictionaries, postal and number tokens | `records_{split}.parquet` |
| block | `blocking.py` | TF-IDF candidate passes within each country label (name char 3-grams, address words), fused by reciprocal rank, capped at 50 candidates per S1 | `candidates_{split}.parquet` |
| feat | `features.py` | float32 pair features (string similarities, token overlaps, per-S1 group context, block scores), streamed in chunks | `features_{split}/part-*.parquet` |
| train | `model.py` | LightGBM, 5-fold GroupKFold by S1, out-of-fold probabilities for every train pair | `oof_train.parquet`, `models/` |
| predict | `model.py` | Mean probability of the 5 fold models on the test pairs | `pred_test.parquet` |
| compare | `decide.py` | Optional: tunes every decision variant on the same train OOF and tabulates F0.5 (overall / singleton / non-singleton / per country) | `decide_compare.tsv` |
| decide | `decide.py` | Tunes the decision rule for macro F0.5 on the train OOF: one threshold `t`, per-source `t_s2`/`t_s3`, per-S1 expected-F0.5 selection, or `conditional_extra` (lower `t_extra` for S1 whose best candidate is >= `t_conf`) (`--decide-method`); empty-list gate `t_empty`; one-owner rule on, off, auto or soft (`--decide-one-owner-delta`) | `decision_config.json` |
| write | `decide.py` | Applies `decision_config.json` unchanged to test and writes both TSVs, then runs the validator | `--out-dir`/`*.tsv` |

There is no country-specific code: blocking compares records that carry the same country
label (whatever the label is), and the model sees only whether two records' countries
agree. France (test only) goes through the same pipeline as US and India.

## Environment setup

From a clone of the repository:

```bash
git clone https://github.com/AdItyAsIngh1800/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
python3.13 -m venv .venv            # or: uv venv --python 3.13 .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```

From the submission zip: unzip it, then run the same `venv` and `pip install -r
code/business_entity_resolution/requirements.txt` commands in the unzipped folder.

macOS: LightGBM needs OpenMP (`brew install libomp`) if it is not already installed.

## Data folder layout

The default `--data-dir` is `dataset/` at the repository (or unzipped package) root, i.e.
`../../dataset` seen from `code/business_entity_resolution/`:

```
<root>/
  dataset/
    train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
  code/business_entity_resolution/   (this folder)
  output/                            (created: the two submission TSVs)
  artifacts/                         (created: intermediate parquet files, models, logs)
```

Any other location works with `--data-dir PATH` (or the `BER_DATA_DIR` environment
variable); likewise `--out-dir` / `BER_OUTPUT_DIR` and `--artifacts-dir` / `BER_ARTIFACTS_DIR`.

## Reproducing the submission end to end

All commands run from `code/business_entity_resolution/`. Run the train split first (it
trains the models and tunes the decision rule), then the test split.

```bash
cd code/business_entity_resolution
D="--data-dir ../../dataset --out-dir ../../output --artifacts-dir ../../artifacts"
# Decision rule: threshold (default) | per_source | expected_f05 | conditional_extra,
# optionally with --decide-one-owner-auto or --decide-one-owner-delta D (soft one-owner). Pick it from the compare table below, e.g.
# DECIDE="--decide-method expected_f05 --decide-one-owner-auto"
DECIDE="--decide-method threshold"

# Train split: prep, block, feat, train, (compare), decide
python -m src.run_pipeline --split train --stage prep    $D
python -m src.run_pipeline --split train --stage block   $D
python -m src.run_pipeline --split train --stage feat    $D
python -m src.run_pipeline --split train --stage train   $D
python -m src.run_pipeline --split train --stage compare $D   # optional: every variant -> decide_compare.tsv
python -m src.run_pipeline --split train --stage decide  $D $DECIDE

# Test split: prep, block, feat, predict, write
python -m src.run_pipeline --split test  --stage prep    $D
python -m src.run_pipeline --split test  --stage block   $D
python -m src.run_pipeline --split test  --stage feat    $D
python -m src.run_pipeline --split test  --stage predict $D
python -m src.run_pipeline --split test  --stage write   $D
```

Equivalent short form: `--stage all --split train $DECIDE`, then `--stage all --split test`
(`all` never runs `compare`).

- A stage is skipped when its outputs already exist, so a crashed run resumes where it
  stopped; `--force` reruns it.
- Each run logs the runtime and peak RSS of every stage to the console and to
  `<artifacts>/logs/run_<timestamp>.log`.
- An artifacts folder is tied to the `--data-dir` that built it (`run_meta.json`); use a
  new `--artifacts-dir` for a different dataset.
- Seeds are fixed (42), LightGBM runs in deterministic mode with a fixed thread count,
  and `decision_config.json` is tuned on train only and applied unchanged to test. It
  records the chosen rule (`method`, thresholds, `one_owner`, and with auto one-owner the
  F0.5 of both settings), so `write` needs no decide flags.

Useful overrides (defaults in `src/config.py`): `--block-chunk-s1-rows N` and
`--feature-chunk-pairs N` (lower them to reduce peak RAM), `--lgbm-num-threads N`,
`--train-max-rows N` (training pairs per fold), `--decision-config PATH` (write: apply a
decision config tuned elsewhere). `python -m src.run_pipeline --help` lists all of them.

### Rule baseline without a model (optional)

```bash
python -m src.run_pipeline --split train --stage baseline $D
python -m src.run_pipeline --split train --stage decide   $D --decide-score-column score
python -m src.run_pipeline --split test  --stage baseline $D
python -m src.run_pipeline --split test  --stage write    $D
```

## Stages, runtime and peak RAM

Hardware: MacBook Air, Apple M4 (4 performance + 6 efficiency cores), 16 GB RAM,
macOS 26.6, Python 3.13.13. Every stage is designed to stay under 10 GB peak RSS.

Measured on the full data (2026-09-26/27 run). Train blocked a stratified 50% of train
S1 (`--block-s1-fraction 0.5`, 1,103,410 S1, 55.2M pairs); test covers all 1,732,544 S1
(86.6M pairs).

| Stage | Train split | Test split | Notes |
|---|---|---|---|
| prep | 257 s, 1.86 GB | 252 s, 1.78 GB | |
| block | 15,886 s (4.4 h), 6.48 GB | 35,211 s (9.8 h), 5.17 GB | train: 50% of S1, `--block-workers 4`; test: 1 worker |
| feat | 998 s, 8.05 GB | 1,894 s, 5.59 GB | 69 features |
| train | 4,459 s (1.2 h), 7.57 GB | — | 5 folds, 12.5M sampled pairs |
| predict | — | 15,371 s (4.3 h), 0.94 GB | 5 fold models × 86.6M pairs |
| compare (optional) | 180 s, 4.40 GB | — | 6 decision variants |
| decide | 42 s, 5.89 GB | — | |
| write | — | 95 s, 4.51 GB | includes the validator |

Peak RSS over all stages: 8.05 GB (train feat).

To refresh this table after a run: `grep -h "Stage .*: end" ../../artifacts/logs/*.log`.

Peak RAM scales with the pair count; if memory is tight, lower `--block-chunk-s1-rows`
(e.g. 1000) and `--feature-chunk-pairs` (e.g. 100000). Runtime then grows slightly.

## Expected outputs

`../../output/` (or `--out-dir`), UTF-8, tab-separated, `\n` line endings:

- `matching_results.tsv`: header `source1_entity_id	matched_entity_ids`, one row per test S1
  (France included), comma-joined S2-/S3- IDs, empty when the S1 has no match.
- `candidate_pairs.tsv`: header `source1_entity_id	candidate_entity_ids`, one row per test
  S1, the exact candidate set the model scored. Every match is also a candidate.

## Validating the output

`write` runs the challenge validator automatically when `utils/validate_submission.py`
exists at the repository root, and refuses to publish the TSVs unless it prints `PASS`.
To run it by hand (from the repository / `student_resource` root):

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Building the submission zip

```bash
python -m src.make_submission --team-name TEAM $D
```

This runs the validator (and refuses to build unless it prints `PASS`), then writes
`<root>/TEAM_submission.zip`. The zip contains `output/` (both TSVs),
`code/business_entity_resolution/` (`src/*.py`, this README and `requirements.txt`) and
`Documentation_template.md`. The script unzips the result to a temporary folder and
checks the tree matches exactly.

## Development samples (optional)

```bash
# Dev sample (10% of train S1, fixed seed) -> ../../artifacts/dev_sample/train/
python -m src.make_dev_sample --data-dir ../../dataset
# Mini sample (~1% of train S1), drawn from the dev sample; pipeline runs stay < 1 GB
# with --block-chunk-s1-rows 1000
python -m src.make_dev_sample --data-dir ../../artifacts/dev_sample --out-dir ../../artifacts_shared/mini_sample --frac 0.1
```

Scores on the samples overstate precision (fewer competing candidates); thresholds for
the submission are always tuned on the full training data.

## Tests and type checks

```bash
cd code/business_entity_resolution
python -m pytest -q src/tests
python -m mypy src          # strict settings live in pyproject.toml (repository only, not in the zip)
```

## Reproducibility

- Dependencies are pinned in `requirements.txt`; a test checks the pins against what `src/` imports.
- Fixed seed 42 throughout: sampling, fold assignment, LightGBM (`deterministic=True`, fixed `num_threads`).
- No step reads anything outside `--data-dir`, and no step uses the network.
