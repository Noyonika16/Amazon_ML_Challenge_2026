# Business Entity Resolution Pipeline

## Overview
This repository contains a high-performance, multi-stage supervised Business Entity Resolution pipeline designed for the Amazon ML Challenge 2026. The solution resolves noisy, unstandardized business records from Source 2 and Source 3 against the deduplicated reference Source 1 across multi-lingual and multi-country partitions (US, India, France).

### Key Architectural Components
1. **Multilingual Unicode & Structural Normalization** (`src/normalize.py`):
   - Strips combining accents/diacritics across French and English records while preserving Devanagari, Tamil, and Indic scripts.
   - Cleans and normalizes legal suffixes (`Pvt Ltd`, `LLP`, `Inc`, `LLC`, `SARL`, `SAS`, `SCI`, etc.).
   - Extracts numeric structural anchors (building numbers, postal codes, phone numbers).

2. **High-Recall Multi-Channel Blocking Engine** (`src/pipeline.py`):
   - Strict dynamic country partitioning (open-set support for France, US, India).
   - Multi-key candidate generation (exact core name, prefix-4, rare tokens, address bigrams, numeric anchors).
   - Achieves > 96% candidate recall ceiling while pruning 99.999% of unpromising pairs.

3. **High-Throughput C++ Feature Engineering** (`src/features.py`):
   - Accelerated with `rapidfuzz` (C++): Levenshtein, Jaro-Winkler, token sort, token set, and partial ratios.
   - Numeric similarity, street number matching, and address conflict penalty detection.

4. **Hard-Negative Trained LightGBM Matcher** (`src/train.py`, `models/lgbm_matcher.txt`):
   - Supervised gradient-boosted pairwise classifier trained on ground-truth matches and retrieval-mined hard negatives.

5. **Macro F_0.5 Optimized Set Decision Engine** (`src/set_decision.py`):
   - High-precision singleton gating protecting entities with no true matches (earning 1.0 instead of fatal false merge penalty).
   - Strict subset constraint enforcement: every matched entity ID is guaranteed to be in `candidate_pairs.tsv`.

---

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── normalize.py            # Multilingual text and address normalizer
│   ├── features.py             # Vectorized pairwise feature extractor
│   ├── set_decision.py         # Entity-level Macro F_0.5 decision engine
│   ├── train.py                # Model training and threshold calibration
│   └── pipeline.py             # End-to-end vectorized inference pipeline
├── README.md                   # Complete reproduction instructions
└── requirements.txt            # Pinned dependencies
```

---

## Reproduction Instructions

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training & Calibration (Optional / Pre-trained)
To retrain the model and calibrate the decision threshold on the training dataset:
```bash
python src/train.py
```
This generates:
- `models/lgbm_matcher.txt`: Trained LightGBM model weights
- `models/optimal_threshold.txt`: Macro $F_{0.5}$ calibrated threshold (0.80)

### 3. Generate Submission Outputs
To run the full end-to-end vectorized inference on the test set:
```bash
python src/pipeline.py
```
This produces:
- `output/candidate_pairs.tsv` (Blocking candidate set)
- `output/matching_results.tsv` (Final entity matches)

### 4. Validate Submission
Run the official competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Validation output must print `PASS` with exit code 0.
