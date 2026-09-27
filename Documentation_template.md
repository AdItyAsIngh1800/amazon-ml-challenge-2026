# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TODO  
**Team Members:** TODO  
**Submission Date:** TODO

> Draft status: every number below comes from `artifacts/eda/eda_report.txt` (full-data
> EDA), `artifacts/experiments.tsv` or a merged/open PR description, and is labelled with
> the data it was measured on. The **dev sample** is 10% of train S1 (220,683 S1); the
> **mini sample** is ~1% (22,068 S1). Both overstate precision (fewer competing
> candidates), so every decision threshold in the submission is tuned on full train.
> Full-data numbers come from the 2026-09-27 run (`artifacts/overnight2_nohup.out`,
> `artifacts/block_test_nohup.out`); the train side blocked a stratified 50% of train S1
> (`--block-s1-fraction 0.5`, Section 3). Anything still open is marked **TODO**.

---

## 1. Executive Summary

A four-stage, country-agnostic pipeline: TF-IDF blocking within each country label (name
character 3-grams on transliterated + phonetic text, address words), fused by reciprocal
rank and capped at 50 candidates per S1; 69 pairwise similarity, group-context and
blocking features; a LightGBM classifier trained with 5-fold GroupKFold by S1 so every
train pair gets an out-of-fold probability; and a decision layer that tunes thresholds
directly for macro F0.5 over all train S1, including singletons and a one-owner rule. The
main innovations are an in-house Indic-script → Latin transliterator with a phonetic name
key (for cross-script India pairs) and a decision layer optimised for the exact challenge
metric rather than for pair-level accuracy.

---

## 2. Methodology

### 2.1 Problem Analysis

Full-data EDA (`artifacts/eda/eda_report.txt`, sections E1–E13; summary numbers only):

| Finding | Number | Implication |
|---|---|---|
| Train size (S1 / S2 / S3) | 2,206,821 / 5,034,616 / 5,285,603 (US + India) | ~110M candidate pairs at 50 per S1 |
| Test size (S1 / S2 / S3) | 1,732,544 / 4,887,273 / 5,082,316 (US + India + **France**) | France (259,452 S1) never appears in train |
| S2+S3 records per S1 | train 4.68, test 5.75 | Test is denser: more distractors per S1 |
| Singleton S1 (no match) (E2) | 5.58% | Each is a full point: predicting nothing must be possible |
| Mean matches per S1 (E2) | 3.46 (S2 1.67, S3 1.79); 76.8% of S1 have 2+ matches from one source | Many-to-one within a source is normal; no 1:1 assumption |
| One-owner check (E3) | 0 of 7,638,365 matched S2/S3 IDs appear in more than one S1 list | Enforce "each S2/S3 ID belongs to at most one S1" |
| Distractors (E4) | 25.99% of train S2/S3 records are in no ground-truth list | Blocking must tolerate unmatched near-duplicates |
| Country agreement (E5) | 100.00% of true pairs share the country label (0 mismatches) | Block within country label; no cross-country pass |
| Empty fields (E6) | Names never empty; 2.3–3.7% of S2/S3 addresses empty (all countries) | Features must handle a missing side (NaN) |
| Postal-like tokens (E7) | Standalone 5–6 digit token in ~11% of US records, 0.3–1.1% of India, 0.4–0.5% of France (test); 5.08% of true pairs share one | Useful signal, but only for a minority of pairs |
| Exact name match (E8, lowercase letters+digits) | 22.94% of true pairs (India 16.29%, US 27.39%) | Exact-key blocking would miss ~77% of matches |
| Name-only TF-IDF recall (E9) | R@50 = 0.700, R@100 = 0.730 over all same-country S2/S3 | Name alone cannot block; an address pass is needed |
| Non-ASCII records (E12) | India 33.47%, US 5.56% (train); France 41.76% (test) | Unicode normalisation and transliteration are required |
| Cross-script India names (PR #8, dev sample) | 16.9% of India true pairs pair a Latin name with an Indic-script name; their name R@20 with plain normalisation was 0.002 | Transliteration + phonetic key |
| Ground-truth integrity (E13) | 0 duplicate S1, 0 missing IDs, 0 S1 IDs inside lists | Labels can be used as-is |

Noise patterns seen in the ground truth (E10/E11 token statistics): legal-form variants
(pvt/private, ltd/limited, llc/inc, transliterated legal forms), injected noise words after
the legal form ("center", "services", ".com"), leading honorifics (mr/dr/smt), word-order
changes, character typos and accents, abbreviated vs. spelled-out street types and state
names, and address components in a different order.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (+ metric-optimised decision layer)  
**Core Innovation:** Indic-script transliteration with a phonetic name key for
cross-script matching, reciprocal-rank-fused multi-pass blocking with reverse
(candidate-side) features, and a decision layer tuned for macro F0.5 over all S1
(singletons included) under a one-owner constraint.

Pipeline (one entry point, `python -m src.run_pipeline --stage ... --split ...`):

1. **prep:** normalise and tokenise names and addresses (Section 4.1).
2. **block:** candidate generation, at most 50 S2/S3 per S1 (Section 3).
3. **feat:** 69 float32 features per candidate pair (Section 4.2).
4. **train / predict:** LightGBM, GroupKFold by S1; out-of-fold probabilities on train,
   fold-mean probabilities on test (Section 4.3).
5. **decide / write:** tune the decision rule on train OOF, apply it unchanged to test,
   write and validate both TSVs (Section 4.5).

**Country-agnostic design for unseen France.** Country is treated as an open set: no
country filtering, one-hot or identity features, and no per-country code paths. Blocking
runs separately for each country label found in the split, whatever it is, so France gets
the same passes as US and India. The only country-derived feature is `country_match`, an
agreement flag. TF-IDF vectorisers are fitted per split and per country on that split's
own records, and feature IDF uses a hashing vectoriser, so France's vocabulary has no
out-of-vocabulary tokens. The normalisation dictionaries include French legal forms and
street types taken from the test-side token statistics (E10) and are applied to every
record without a country condition. Leave-one-country-out (LOCO) results are in
Section 4.4.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** two TF-IDF cosine passes, each fitted per country label on that
  split's S1+S2+S3 records and never crossing countries (E5):
  - **Pass A (name):** character 3-grams on the composite text
    `name_norm | name_norm | name_key` (normalised name weighted twice plus the phonetic
    key); top K = 50 per S1.
  - **Pass B (address):** word unigrams on `addr_norm` (empty addresses skipped); top
    K = 50 per S1. On the dev sample, words beat character 3-grams for addresses on both
    recall and RAM (single-pass R@50 India 0.900 vs 0.853, US 0.928 vs 0.883; PR #7).
  - The candidates contract reserves passes C (postal) and F; they are not implemented and
    are written as all-NaN.
- **Candidate pairs generated:** dev sample 11.03M pairs for 220,683 S1 (50.0 per S1);
  mini sample 1,103,400 pairs for 22,068 S1. Full train (50% of S1, below): 55,170,500
  pairs for 1,103,410 S1; full test: 86,627,200 pairs for 1,732,544 S1 (50.0 per S1).
- **Train S1 fraction.** Full-train blocking runs on a deterministic 50% of train S1,
  stratified by country × singleton (`--block-s1-fraction 0.5`: 1,103,410 of 2,206,821
  S1), with 4 worker processes (`--block-workers 4`). The candidate pool (all S2/S3 of the
  country) is not subsampled, so each kept S1 faces the full competition. The other half
  gets no candidates and is left out of training, decide and every train-side report.
  Test blocking always covers every S1.
- **How you ensured true matches were not lost:**
  - *No full similarity matrix.* Query-side pruning keeps each S1 row's n-grams with
    document frequency ≤ `max_df` × pool plus always its `min_query_terms` rarest ones
    (Pass A: 0.001 / 16; Pass B: 0.003 / 3). Sparse products are chunked by an nnz upper
    bound (≤ 50M nnz per product). The top 200 partial scores per S1 are rescored with the
    exact cosine, then the top K are kept. Plain feature-level `max_df` pruning had cut
    India R@50 from 0.697 to 0.479, so it was rejected (PR #7). Raising Pass A's
    `min_query_terms` from 4 to 16 took Pass A R@50 on 5,000 S1/country from
    0.695 → 0.732 (India) and 0.752 → 0.870 (US) (PR #12).
  - *Union with a reciprocal-rank-fusion cap.* The pass results are unioned, and the
    50-per-S1 cap keeps the pairs with the highest RRF score Σ 1/(10 + rank). On the dev
    sample, RRF at cap 50 keeps pair recall 0.9773 against 0.9511 for a max-score cap; RRF
    at cap 20 (0.9693) already beats max-score at cap 50 (PR #7).
  - *Reverse features.* For each candidate, computed over all S1 of the split:
    `rev_n_s1` (how many S1 retrieved it), `rev_rank` (its rank among them) and `rev_gap`
    (score gap to its best S1). These carry the one-owner structure into the model.
  - *Recall tracking* on every train run (`blocking_recall_train.tsv`, experiment log).

**Dev-sample pair recall (train, 220,683 S1, 763,722 true pairs; experiment log rows
block-v0 04:46 and block-v1 15:02; PR #12).** v0 = Pass A on `name_norm` only, prep v0;
v1 = Pass A on `name_norm | name_norm | name_key`, `min_query_terms` 16, prep v1.

| Variant | block-v0 | block-v1 | Δ |
|---|---|---|---|
| Pass A (ALL) | 0.7514 | 0.8125 | +0.0611 |
| Pass B (ALL) | 0.9169 | 0.9199 | +0.0030 |
| Union, uncapped | 0.9804 | 0.9858 | +0.0054 |
| Union @20 | 0.9693 | 0.9752 | +0.0059 |
| **Union @50 (shipped)** | **0.9773** | **0.9826** | **+0.0053** |
| Union @50, India | 0.9668 | 0.9713 | +0.0045 |
| Union @50, US | 0.9843 | 0.9902 | +0.0059 |
| S1 fully covered @50 | 93.48% | 94.96% | +1.48 pp |

**Full-train pair recall** (50% of train S1: 1,103,410 S1, 3,818,728 true pairs;
experiment row block-v1 2026-09-27T08:39:03, `blocking_recall_train.tsv`):

| Variant | ALL | India | US |
|---|---|---|---|
| Pass A | 0.6859 | 0.6222 | 0.7284 |
| Pass B | 0.8812 | 0.8515 | 0.9011 |
| Union, uncapped (~97 per S1) | 0.9625 | 0.9427 | 0.9758 |
| Union @20 | 0.9313 | 0.9098 | 0.9457 |
| Union @30 | 0.9431 | 0.9213 | 0.9577 |
| **Union @50 (shipped)** | **0.9535** | **0.9323** | **0.9677** |
| Union @80 | 0.9602 | 0.9399 | 0.9738 |
| S1 fully covered @50 | 87.75% | 82.98% | 90.93% |

This is below the 98% target and below the dev-sample 0.9826; Section 5.3 explains the
gap. Raising the cap from 50 to 80 would add only +0.0067.

**Miss analysis (PR #15, mini sample, block-v1; experiment row block-v1 17:31).** 530 of
76,280 true pairs (0.69%) are not candidates: India 444 of 30,434 (1.46%), US 86 of 45,846
(0.19%); S2 290, S3 240. Primary cause (first match in this order):

| Primary cause | Pairs | Share of misses |
|---|---|---|
| Cross-script names (one side pure Latin, the other pure Indic; all India) | 258 | 48.7% |
| Pushed out by the 50 cap (India 69, US 30) | 99 | 18.7% |
| Empty address on one side | 89 | 16.8% |
| Low name and address similarity (both < 0.3) | 9 | 1.7% |
| Other | 75 | 14.2% |

Cross-script pairs are 58% of India misses, so Indic-side transliteration/phonetics is the
largest remaining recall lever; the cap costs 0.13 pp of recall.

**Full-train miss analysis** (`python -m src.blocking_misses`,
`artifacts/blocking_misses/report.txt`; same 50% S1 subset). 177,381 of 3,818,728 true
pairs (4.65%) are not candidates: India 103,470 of 1,529,471 (6.77%), US 73,911 of
2,289,257 (3.23%); S2 80,805 (4.38%), S3 96,576 (4.90%).

| Primary cause | India | US | All | Share of misses |
|---|---|---|---|---|
| Cross-script names (Latin S1, Indic candidate) | 42,228 | 0 | 42,228 | 23.8% |
| Empty address on one side | 11,464 | 24,286 | 35,750 | 20.2% |
| Pushed out by the 50 cap | 15,837 | 18,424 | 34,261 | 19.3% |
| Low name and address similarity (both < 0.3) | 621 | 157 | 778 | 0.4% |
| Other | 33,320 | 31,044 | 64,364 | 36.3% |

At full scale the cap costs 0.90 pp of recall (34,261 pairs) against 0.13 pp on the mini
sample, and "other" misses (median name similarity 0.57, address 0.43: moderately similar
pairs that simply rank below 50 closer-looking distractors in each pass) become the
largest group. Cross-script names still account for 44.5% of India misses (46,026 flagged
pairs).

---

## 4. Matching Model

### 4.1 Normalisation and transliteration (prep)

- Unicode NFKD with accent stripping (standard library `unicodedata`), lower-casing,
  punctuation removal; CSV-escaped quotes kept literally when reading and stripped here.
- **Indic → Latin transliteration** (own code, `transliterate.py`, no dependencies):
  Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam are mapped to
  Devanagari by Unicode block offset (non-aligned code points are mapped explicitly), then
  Devanagari is romanised (inherent vowel, virama, vowel signs, final-schwa deletion,
  anusvara/visarga/nukta rules). Indic digits become ASCII, so Indic postcodes become
  postal tokens.
- **Phonetic name key** (`name_key`): the same rules for every record, e.g. collapse vowel
  length and aspiration, merge letter variants (x→ks, q→k, z→j, f→p, c→k, w→v), drop
  doubled letters and all vowels except a word's first letter. On real cross-script true
  pairs, the median rapidfuzz ratio of `name_key` is 85–95 in every script, against ~10
  on raw names (PR #8).
- **Dictionaries as domain rules** (`dictionaries.py`; data only; every entry commented
  with its EDA/dev-sample evidence): legal forms and their transliterated and French
  variants (→ `legal_suffix`, with `name_core` = name without legal forms), noise words
  injected after legal forms, leading honorifics, name abbreviations (shri/shree → sri),
  address abbreviations (street types, US and India state names → codes) and landmark
  words. They encode general knowledge about business names and addresses and are applied
  to every record regardless of country; no external data or lookup is used.
- **Postal tokens:** every standalone 5–6 digit number, all countries, plus joined
  `ddd ddd` pairs (E7 pattern). This raised the share of India true pairs sharing a postal
  token from 0.25% to 1.88% on the dev sample (PR #9).

Name TF-IDF recall on the dev sample (E9 method, 5,000 S1 per country, R@20; PR #8, #9;
rows norm-001, norm-002): India 0.661 (prep v0) → 0.683 (transliteration + key) → 0.692
(dictionaries); US 0.829 → 0.835 → 0.836. India cross-script pairs: R@20 0.002 → 0.158.
Address TF-IDF R@20: India 0.864 → 0.879, US 0.920 → 0.925.

### 4.2 Features (69, all float32; `features.py`)

**Features used:**
- **Name features (similarity group, part of 37):** rapidfuzz ratio, token_set,
  token_sort, partial, Jaro-Winkler and normalised Levenshtein on both `name_core` and
  `name_norm`; `name_key` ratio; `name_translit` ratio and token_set; character 3-gram
  TF-IDF cosine; IDF-weighted and plain token Jaccard; acronym match both ways; legal
  suffix equal / conflict / missing; length ratio; first-token match; digit-token Jaccard
  and a one-side-digits flag.
- **Address features (similarity group):** postal match / conflict / missing; number-token
  overlap and conflict; word TF-IDF cosine; IDF-weighted and plain token Jaccard;
  token_set on `addr_norm` and `addr_translit`.
- **Record flags (7):** landmark flag, empty name and empty address for both sides;
  `cand_is_s3`.
- **Group context within each S1 (10):** candidate count; rank, gap to the best candidate
  and z-score of `name_char3_cos`, `name_core_token_set` and `addr_tok_cos`.
- **Blocking (15):** `country_match` (agreement flag), per-pass score and rank for passes
  A, B, C, F (C and F are all-NaN), `n_passes`, `best_block_score`, `rrf_score`, and the
  reverse features `rev_n_s1`, `rev_rank`, `rev_gap`.

Undefined similarities (an empty side) are NaN, which LightGBM handles natively. Feature
IDF uses a `HashingVectorizer` (2^20 buckets) with document frequencies counted over every
record of the split. No feature identifies a country (a unit test checks this).
Strongest single features on the mini sample (AUC): `addr_norm_token_set` 0.994,
`rrf_score` 0.994, `pass_B_score` 0.990 (row feat-001).

### 4.3 Model

**Model type:** LightGBM 4.7.0 binary classifier (gradient-boosted trees, trained from
scratch; no pretrained model). Parameters: learning rate 0.05, 127 leaves,
`min_data_in_leaf` 100, feature and bagging fraction 0.8, L2 1.0, `max_bin` 255, up to
3,000 rounds with early stopping after 100 rounds on 10% of the sampled S1 groups;
`deterministic=True`, seed 42, fixed thread count.

- **Folds:** 5-fold GroupKFold by S1 over **all** train S1, so no S1 appears in both the
  training and the held-out part of a fold (all *blocked* train S1 on full data).
- **Out-of-fold predictions:** every train pair is predicted by the model of its held-out
  fold, so the OOF file covers all ~110M train pairs and the decision layer is tuned on
  exactly the pair distribution the test predictions will have. On full train this is
  55.2M pairs (the blocked 50% of S1).
- **Training-row cap:** one shared sample of whole S1 groups
  (`config.TRAIN_MAX_ROWS` = 10M × 5/4 pairs) is binned once; each fold trains on its
  ~10M-row subset. This bounds peak RAM (measured 7.57 GB at full scale; PR #14).
- **Test:** mean probability of the 5 fold models.

Mini sample (row feat-002, PR #14): OOF AUC 0.99998; best iterations 176–195 per fold.
Full train (row feat-model-20260927-101003): 55,170,500 pairs (1,103,410 S1, 3,641,347
positives); a shared sample of 12.5M pairs (250,000 S1 groups), ~9.0M training / ~1.0M
early-stopping pairs per fold; best iterations 2374, 2562, 2393, 2767, 2196 (validation
log-loss 0.0086–0.0087); **OOF AUC 0.99979**. Top features by gain: `rrf_score`,
`grp_addr_tok_cos_z`, `addr_norm_token_set`, `num_overlap`, `name_core_partial`,
`name_core_ratio`. OOF calibration: expected calibration error 0.0003, every 0.1-bin's
positive rate within 0.023 of its mean probability.

### 4.4 Leakage audit and LOCO (PR #20, mini sample)

- **No leak.** The 69 model features exclude `s1_id`, `cand_id`, `fold`, `label` and
  `prob`, and match the saved model's feature names. The ground truth is read only to
  create labels and to tune the decision layer; blocking and normalisation never read it.
  IDF, reverse and group features are unsupervised and computed the same way on test. No
  single feature separates the labels (best single-feature AUC 0.994).
- **Feature-group ablation** (same CV recipe; macro F0.5): full 69 features 0.9920;
  name + address similarity only (37) 0.9905; full without blocking features 0.9914;
  full without group features 0.9920. Blocking features dominate gain but add only
  +0.0006 F0.5; name and address similarity carry the model.
- **LOCO** (train on one country's S1, tune thresholds on that country's OOF, score the
  other country with the fold-mean model):

| Features | US → India | India → US | Mean |
|---|---|---|---|
| In-distribution (reference) | 0.9879 | 0.9948 | 0.9914 |
| **Full (69)** | **0.9831** | **0.9933** | **0.9882** |
| Similarity only (37) | 0.9813 | 0.9920 | 0.9867 |
| Full without blocking features | 0.9825 | 0.9926 | 0.9876 |

  The drop for an unseen country is 0.005 (India) and 0.0015 (US). Source-country
  thresholds are within 0.001 of thresholds re-tuned on the target country, so
  calibration transfers. Drop-one-feature LOCO (69 runs): deltas −0.0003 to +0.0006, none
  reaches the +0.002 acceptance bar, so no feature was removed.
- **Caveat:** the mini sample keeps all true matches plus *random* distractors, so hard
  negatives are rare; these ~0.99 scores will not carry over to full data. Full-train LOCO was **not
  run**: it needs a retrain per held-out country at full scale (Section 6).

### 4.5 Decision layer

**Threshold selection method:** direct maximisation of the challenge metric (macro F0.5
over **all** train S1, singletons included) on the full-train out-of-fold probabilities.
True-match counts come from the ground truth, so pairs lost in blocking count as false
negatives. The tuned rule is saved in `decision_config.json` and applied unchanged to
test.

- **One-owner rule:** each S2/S3 ID is kept only for the S1 that gives it the highest
  score (ties → lowest S1), following E3. It can be on, off, or chosen automatically by
  train F0.5.
- **Rules** (`--decide-method`):
  - `threshold`: keep candidates with prob ≥ t; return an empty list when the S1's best
    prob < `t_empty` (protects singletons).
  - `per_source`: separate `t_s2` / `t_s3` (source read from the ID prefix), tuned with
    `t_empty` by coordinate descent from the global optimum, so never worse than it.
  - `expected_f05`: per S1, sort candidates by prob and keep the prefix (the empty set
    included) with the highest expected F0.5, treating probabilities as calibrated.
- **Threshold grid:** the union of 200 score quantiles (for rank-fusion scores) and a
  fixed 0.005-step grid on (0, 1). The quantile-only grid had no values between 0.02 and
  0.99 on probabilities, because almost all pairs score near 0 (PR #20); the fixed grid
  closes that gap (PR #21, merged). The search sorts pairs once and
  sweeps the grid incrementally (O(n log n)), and unit tests check it against brute force.
- **Calibration:** expected calibration error 0.0003 on the mini OOF (PR #18) and 0.0003
  on the full-train OOF (55.2M pairs, `compare_run.log`):

| Probability bin | Pairs | Mean prob | Positive rate |
|---|---|---|---|
| [0.0, 0.1) | 51,291,576 | 0.0005 | 0.0007 |
| [0.1, 0.2) | 121,497 | 0.1434 | 0.1655 |
| [0.2, 0.3) | 66,682 | 0.2464 | 0.2666 |
| [0.3, 0.4) | 48,357 | 0.3478 | 0.3635 |
| [0.4, 0.5) | 40,077 | 0.4489 | 0.4567 |
| [0.5, 0.6) | 36,914 | 0.5498 | 0.5521 |
| [0.6, 0.7) | 38,936 | 0.6510 | 0.6512 |
| [0.7, 0.8) | 48,123 | 0.7526 | 0.7481 |
| [0.8, 0.9) | 77,596 | 0.8557 | 0.8526 |
| [0.9, 1.0] | 3,400,742 | 0.9958 | 0.9956 |

Mini sample, model v0 OOF (rows A-decide-20260926-1739*, -205011, -205447; PR #18, #21):

| Rule (+ one-owner) | Quantile grid only | Quantile ∪ 0.005 grid |
|---|---|---|
| threshold | 0.9827 | 0.9922 (t = 0.745, t_empty = 0.875) |
| per_source | 0.9827 | 0.9922 (t_s2 = 0.80, t_s3 = 0.73) |
| expected_f05 | 0.9916 | 0.9916 (no grid) |

On the same mini sample, the rule baseline (`rrf_score` only, no model) scores 0.9358 with
the same decision layer. The final rule is chosen on full train with `--stage compare` (`decide_compare.tsv`,
full-train OOF, 1,103,410 S1; runtime 180 s):

| Variant | Params | Overall | Singleton | Non-singleton | India | US |
|---|---|---|---|---|---|---|
| **threshold + one-owner (chosen)** | t = 0.69, t_empty = 0.73 | **0.9597** | 0.9555 | 0.9600 | 0.9512 | 0.9654 |
| per_source + one-owner | t_s2 = t_s3 = 0.69, t_empty = 0.73 | 0.9597 | 0.9555 | 0.9600 | 0.9512 | 0.9654 |
| expected_f05 + one-owner | — | 0.9588 | 0.9099 | 0.9617 | 0.9505 | 0.9644 |
| threshold | t = 0.705, t_empty = 0.75 | 0.9593 | 0.9566 | 0.9595 | 0.9506 | 0.9651 |
| per_source | t_s2 = 0.69, t_s3 = 0.71, t_empty = 0.75 | 0.9593 | 0.9566 | 0.9595 | 0.9506 | 0.9651 |
| expected_f05 | — | 0.9582 | 0.9024 | 0.9615 | 0.9497 | 0.9638 |

**Chosen: `threshold` + one-owner (t = 0.69, t_empty = 0.73, F0.5 0.9597).** Per-source
thresholds give no gain: with one-owner the search lands on t_s2 = t_s3, and without it
t_s2 = 0.69 / t_s3 = 0.71 score the same as the single threshold. Expected-F0.5 is 0.0009
lower overall even though the probabilities are very well calibrated (ECE 0.0003): it is
slightly better on non-singletons (0.9617 vs 0.9600) but drops singletons to 0.910, most likely
because a single candidate with moderate probability already makes a non-empty list the
expected-F0.5 optimum, while `t_empty` is tuned directly on the all-or-nothing singleton
score. One-owner adds +0.0004 (0.9597 vs 0.9593).

---

## 5. Results & Error Analysis

### 5.1 Full-train OOF F0.5

`threshold` rule, one-owner on, tuned on the full-train OOF probabilities (row
A-decide-20260927-101046): **t = 0.69, t_empty = 0.73**. Scored over the 1,103,410
blocked train S1; blocking misses count as false negatives.

| Segment | S1 | F0.5 |
|---|---|---|
| **Overall** | 1,103,410 | **0.9597** |
| Singleton S1 | 61,624 | 0.9555 |
| Non-singleton S1 | 1,041,786 | 0.9600 |
| US | 661,816 | 0.9654 |
| India | 441,594 | 0.9512 |
| LOCO mean (**mini sample**, Section 4.4) | — | 0.9882 |

Full-train LOCO was not run (it needs a retrain per held-out country); the mini-sample
LOCO above uses random distractors and overstates the level, not just the drop.

Without the one-owner rule the best F0.5 is 0.9593 (t = 0.705, t_empty = 0.75), so the
rule adds +0.0004. On test, the same config gives 5,528,207 matches for 1,732,544 S1 and
110,078 empty lists (6.4%, against 5.6% singletons in train).

### 5.2 Leaderboard

| Submission | Description | Leaderboard F0.5 |
|---|---|---|
| M1 | Rule baseline: `rrf_score` + decision layer, no model (dev F0.5 0.888) | 0.677 |
| **M2 (final)** | LightGBM model + `threshold` + one-owner (this document) | **0.953** |

M2 scores 0.953 against 0.9597 full-train OOF, a gap of 0.007. Likely reasons:

- **Denser test:** 5.75 S2/S3 per S1 against 4.68 in train, so more distractors per S1
  than the thresholds were tuned on.
- **Unseen France:** 259,452 test S1 (15%) come from a country never seen in training or
  threshold tuning.
- **50% train fraction:** with only half of train S1 blocked, fewer S1 compete for each
  S2/S3 ID in validation, so the one-owner rule and the thresholds were tuned under
  weaker competition than on test, where every S1 is blocked.

### 5.3 Dev sample vs. full data

Every metric drops from the dev sample to full data: blocking recall @50 0.9826 (dev) vs
0.9535 (full train), and M1 0.888 (dev) vs 0.677 (leaderboard). The cause is denser
competition at full scale. The dev sample keeps 10% of train S1 with all their true
matches plus *random* S2/S3 distractors (`make_dev_sample.py`), so its pool is ~10% of the
full pool and holds few near-duplicates of any S1's true matches. At full scale:

- **Blocking:** the TF-IDF passes rank each S1 against every S2/S3 of its country (3.8M US,
  4.7M India records in test). Many more similar-looking businesses compete for the 50
  slots: Pass A recall falls from 0.8125 (dev) to 0.6859, cap push-outs rise from 0.13 pp
  to 0.90 pp, and moderately similar true pairs ("other" misses) are outranked.
- **Decision:** more high-scoring distractors per S1 cost precision, which F0.5 weighs
  twice. M1 ranks by `rrf_score` alone, which cannot tell a true match from a close
  distractor, so it lost most (0.888 → 0.677). Test is denser still (5.75 S2/S3 per S1
  vs 4.68 in train) and adds unseen France.
- **Model:** the model's full-train OOF F0.5 (0.9597) is below its mini-sample score
  (0.9922) for the same reason, but the drop is far smaller than M1's because its features
  (name/address similarity, group context, reverse features) separate close distractors.

Thresholds are therefore tuned only on full train, never on a sample.

### 5.4 Error analysis

- **Decision variants:** full-train comparison in Section 4.5 (threshold + one-owner
  chosen; per-source equal; expected-F0.5 −0.0009, singletons 0.910).
- **Common false positives (wrong merges):** TODO (from full-train OOF; describe patterns
  only, no raw records).
- **Common false negatives (missed matches):** 4.65% of true pairs are never scored
  because blocking missed them (Section 3: cross-script names, empty addresses, the cap,
  moderately similar pairs outranked by distractors). A breakdown of the remaining
  decision/model misses: TODO.

---

## 6. Conclusion

The pipeline runs end to end on full data on a 16 GB laptop (peak RSS 8.05 GB, README
runtime table) and reaches full-train OOF F0.5 0.9597 (US 0.9654, India 0.9512,
singletons 0.9555) and **0.953 on the leaderboard** (M1 rule baseline: 0.677). Blocking
is the main ceiling: 4.65% of true pairs are never scored.

**Limitations**

- **Cross-script recall.** 42,228 of 177,381 full-train blocking misses (23.8%) are
  Latin S1 names whose true candidate is written in an Indic script. Transliteration
  helps inside Pass A, but these pairs still lose to Latin-script distractors.
- **The 50-candidate cap** pushes out 34,261 true pairs (0.90 pp recall) at full scale.
- **50% train fraction.** Full-train blocking, training and threshold tuning used half of
  the train S1 to fit the time budget. The candidate pool was not reduced, so recall and
  precision are measured under full competition, but the one-owner rule was validated
  with only half of the competing S1 present: an S2/S3 ID can be claimed only by blocked
  S1, so conflicts between S1 are undercounted and its measured gain (+0.0004) may differ
  from its effect on test, where every S1 is blocked.
- **No full-train LOCO.** Transfer to an unseen country (France) was measured only on
  the mini sample (drop ≤ 0.005); a full-scale LOCO needs one retrain per held-out
  country and was not run.
- **Runtime.** Test blocking took 35,211 s (9.8 h, single worker) and predict 15,371 s
  (4.3 h, 5 fold models × 86.6M pairs); both dominate the end-to-end time.

**Future work**

- **Pass T:** a dedicated transliterated-name blocking pass (`name_translit` / `name_key`
  of both sides, restricted to pairs where one side is Indic-script) to recover the
  cross-script misses, fused into the RRF cap.
- An adaptive cap (more slots for S1 with many close candidates) instead of a flat 50.
- Blocking all train S1 (fraction 1.0) with `--block-workers` to validate one-owner under
  the test-time competition.
- Faster predict: fewer trees (early-stopped refit on all data instead of 5 fold models),
  or LightGBM `num_threads` tuned for inference.

---

## Appendix

### A. Code Artefacts

Code lives in `code/business_entity_resolution/`: all source in `src/`, plus `README.md`
and `requirements.txt` (Python 3.13).

| Module | Role |
|---|---|
| `run_pipeline.py` | Single entry point; stage dispatch, config overrides, logging, run metadata |
| `config.py`, `contracts.py` | Tunables and paths; parquet/JSON column contracts |
| `io_utils.py` | The only TSV reader/writer (QUOTE_NONE, UTF-8, row-count checks) |
| `normalize.py`, `transliterate.py`, `dictionaries.py` | Stage prep → `records_{split}.parquet` |
| `blocking.py` | Stage block → `candidates_{split}.parquet` |
| `features.py` | Stage feat → `features_{split}/part-*.parquet` |
| `model.py` | Stages train (→ `oof_train.parquet`, `models/`) and predict (→ `pred_test.parquet`) |
| `decide.py` | Stages decide (→ `decision_config.json`), compare, write (→ both TSVs, validated), baseline |
| `evaluate.py` | Official F0.5 scorer, segment tables, blocking recall |
| `make_submission.py` | Builds and checks the submission zip |
| `eda.py`, `make_dev_sample.py`, `name_recall.py`, `blocking_misses.py` | Analysis and sampling tools |
| `logging_utils.py`, `experiment_log.py`, `fulldata_lock.py` | Runtime/RAM tracking, experiment log, full-data lock |

Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
`code/business_entity_resolution/` (train split first, then test):

```bash
pip install -r requirements.txt
D="--data-dir ../../dataset --out-dir ../../output --artifacts-dir ../../artifacts"
python -m src.run_pipeline --stage all --split train $D --decide-method threshold   # prep, block, feat, train, decide
python -m src.run_pipeline --stage all --split test  $D                             # prep, block, feat, predict, write
python -m src.make_submission --team-name TEAM $D                                   # optional: validated zip
```

Each stage skips if its outputs exist (`--force` reruns it) and logs runtime and peak RSS.
The decision method is set to the full-train winner of `--stage compare` (`threshold` +
one-owner). `write`
runs `utils/validate_submission.py` and publishes the TSVs only on `PASS`. The README
gives stage-by-stage commands, data layout, overrides and measured runtimes. Hardware: MacBook
Air (Apple M4, 16 GB); every stage is designed for ≤ 10 GB peak RAM. Tests:
`python -m pytest -q src/tests && python -m mypy src`.

### B. Additional Results

Tables instead of charts; all full train (50% of S1), from `blocking_recall_train.tsv`,
`compare_run.log` and `decide_compare.tsv`.

**B.1 Blocking recall vs. cap** (pair recall; % S1 fully covered in brackets)

| Cap | ALL | India | US |
|---|---|---|---|
| 20 | 0.9313 (82.7%) | 0.9098 (78.1%) | 0.9457 (85.8%) |
| 30 | 0.9431 (85.3%) | 0.9213 (80.6%) | 0.9577 (88.5%) |
| **50 (shipped)** | **0.9535 (87.7%)** | **0.9323 (83.0%)** | **0.9677 (90.9%)** |
| 80 | 0.9602 (89.3%) | 0.9399 (84.7%) | 0.9738 (92.4%) |
| uncapped (~97 per S1) | 0.9625 (89.9%) | 0.9427 (85.3%) | 0.9758 (92.9%) |

**B.2 Calibration of OOF probabilities** (ECE 0.0003)

| Probability bin | Pairs | Mean prob | Positive rate |
|---|---|---|---|
| [0.0, 0.1) | 51,291,576 | 0.0005 | 0.0007 |
| [0.1, 0.2) | 121,497 | 0.1434 | 0.1655 |
| [0.2, 0.3) | 66,682 | 0.2464 | 0.2666 |
| [0.3, 0.4) | 48,357 | 0.3478 | 0.3635 |
| [0.4, 0.5) | 40,077 | 0.4489 | 0.4567 |
| [0.5, 0.6) | 36,914 | 0.5498 | 0.5521 |
| [0.6, 0.7) | 38,936 | 0.6510 | 0.6512 |
| [0.7, 0.8) | 48,123 | 0.7526 | 0.7481 |
| [0.8, 0.9) | 77,596 | 0.8557 | 0.8526 |
| [0.9, 1.0] | 3,400,742 | 0.9958 | 0.9956 |

**B.3 Decision variants** (macro F0.5)

| Variant | Params | Overall | Singleton | Non-singleton | India | US |
|---|---|---|---|---|---|---|
| **threshold + one-owner (chosen)** | t = 0.69, t_empty = 0.73 | **0.9597** | 0.9555 | 0.9600 | 0.9512 | 0.9654 |
| per_source + one-owner | t_s2 = t_s3 = 0.69, t_empty = 0.73 | 0.9597 | 0.9555 | 0.9600 | 0.9512 | 0.9654 |
| expected_f05 + one-owner | — | 0.9588 | 0.9099 | 0.9617 | 0.9505 | 0.9644 |
| threshold | t = 0.705, t_empty = 0.75 | 0.9593 | 0.9566 | 0.9595 | 0.9506 | 0.9651 |
| per_source | t_s2 = 0.69, t_s3 = 0.71, t_empty = 0.75 | 0.9593 | 0.9566 | 0.9595 | 0.9506 | 0.9651 |
| expected_f05 | — | 0.9582 | 0.9024 | 0.9615 | 0.9497 | 0.9638 |

### C. Models and Libraries

| Component | Version | Licence | Use |
|---|---|---|---|
| LightGBM | 4.7.0 | MIT | Pair classifier (trained from scratch; tree ensemble, far below 8B parameters; 5 fold models, 12,292 trees in total) |
| scikit-learn | 1.9.1 | BSD-3-Clause | TF-IDF / hashing vectorisers |
| rapidfuzz | 3.14.6 | MIT | String similarity features |
| pandas | 3.0.6 | BSD-3-Clause | Tabular I/O |
| pyarrow | 25.0.1 | Apache-2.0 | Parquet and Arrow storage |
| numpy | 2.5.3 | BSD-3-Clause | Arrays |
| scipy | 1.18.1 | BSD-3-Clause | Sparse matrix products |
| psutil | 7.2.2 | BSD-3-Clause | Peak-RAM logging |
| pytest, mypy (dev only) | 9.1.1, 2.3.1 | MIT | Tests and type checks |

No pretrained, hosted or external models are used. Transliteration, phonetic keys and
dictionaries are our own code (standard-library `unicodedata`; no GPL `unidecode`, no
libpostal).

### D. Fair Play Statement

- No external data, APIs, geocoding, business registries, lookups or hosted models; no
  network calls anywhere in `src/`. The dataset never left the local machine.
- Every learned statistic comes only from the provided challenge data: TF-IDF vectorisers
  for blocking are fitted per split (and per country label) on that split's own records,
  feature IDF is counted per split, and the model and decision thresholds are trained and
  tuned on the train split only, then applied unchanged to test. Test ground truth was
  never available or used.
- Normalisation dictionaries are hand-written domain rules (legal forms, abbreviations,
  honorifics, state names) justified by token statistics of the provided data.
- Country is never used as an identity feature or filter; only a same-country agreement
  flag is used.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
