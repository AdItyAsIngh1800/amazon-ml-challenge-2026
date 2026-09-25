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

# Train split, then test split
python -m src.run_pipeline --stage all --split train --data-dir ../../dataset --out-dir ../../output
python -m src.run_pipeline --stage all --split test  --data-dir ../../dataset --out-dir ../../output
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
