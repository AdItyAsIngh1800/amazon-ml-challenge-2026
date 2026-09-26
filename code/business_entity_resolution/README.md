# Business Entity Resolution — Amazon ML Challenge 2026

Python 3.13 (tested on 3.13.13 macOS).

## Overview

## Environment setup

```bash
git clone https://github.com/AdItyAsIngh1800/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
python3.13 -m venv .venv            # or: uv venv --python 3.13 .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```

macOS: LightGBM needs OpenMP (`brew install libomp`) if it is not already installed.

## Data folder layout

```
dataset/
  train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Running the pipeline

All commands run from `code/business_entity_resolution/`.

```bash
# Dev sample (10% of train S1, fixed seed) -> ../../artifacts/dev_sample/train/
python -m src.make_dev_sample --data-dir ../../dataset
# Mini sample (~1% of train S1, same strata / seed / ratio), drawn from the dev sample;
# pipeline runs stay < 1 GB with --block-chunk-s1-rows 1000
python -m src.make_dev_sample --data-dir ../../artifacts/dev_sample --out-dir ../../artifacts_shared/mini_sample --frac 0.1

# Train split, then test split
python -m src.run_pipeline --stage all --split train --data-dir ../../dataset --out-dir ../../output
python -m src.run_pipeline --stage all --split test  --data-dir ../../dataset --out-dir ../../output

# M1 rule baseline (no model): blocking's rrf_score replaces train/predict
python -m src.run_pipeline --stage baseline --split train --data-dir ../../dataset --out-dir ../../output
python -m src.run_pipeline --stage decide   --split train --data-dir ../../dataset --out-dir ../../output --decide-score-column score
python -m src.run_pipeline --stage baseline --split test  --data-dir ../../dataset --out-dir ../../output
python -m src.run_pipeline --stage write    --split test  --data-dir ../../dataset --out-dir ../../output
# ...or apply a decision_config.json tuned elsewhere (e.g. on the dev sample)
python -m src.run_pipeline --stage write    --split test  --data-dir ../../dataset --out-dir ../../output --decision-config /abs/path/decision_config.json

# Decision variants: compare all on the same oof_train -> <artifacts>/decide_compare.tsv
python -m src.run_pipeline --stage compare --split train --data-dir ../../dataset --out-dir ../../output
# then tune the chosen one (threshold | per_source | expected_f05; optional auto one-owner)
python -m src.run_pipeline --stage decide  --split train --data-dir ../../dataset --out-dir ../../output --force \
    --decide-method expected_f05 --decide-one-owner-auto
```

## Stages, runtime and peak RAM

## Expected outputs

## Tests and type checks

```bash
cd code/business_entity_resolution
python -m pytest -q
python -m mypy
```

## Reproducibility
