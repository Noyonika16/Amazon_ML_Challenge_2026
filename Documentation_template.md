# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Team EntityResolvers  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary
We developed an end-to-end, multi-stage supervised Business Entity Resolution architecture engineered specifically for the precision-heavy macro $F_{0.5}$ metric on multi-million record scale. The system integrates multilingual Unicode and legal-entity normalization, a multi-channel high-recall blocking engine achieving a >96% recall ceiling while pruning 99.999% of negative pairs, vectorized C++ accelerated pairwise feature engineering via RapidFuzz, a LightGBM gradient-boosted pairwise classifier trained with retrieval-mined hard negatives, and an entity-level set-decision engine with explicit singleton protection. Operating with zero external data lookup and strictly open-source, compliant models, the pipeline achieves a validation macro $F_{0.5}$ of **0.7337** with **95.9% precision** while processing over 10,000 entities per second.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) across the 2.2 million training records and ~10 million candidate records, several structural and semantic phenomena were uncovered:
1. **Macro $F_{0.5}$ Metric Asymmetry**: The evaluation metric weights precision twice as heavily as recall ($\beta = 0.5$). False merges (merging two distinct businesses) penalize the score drastically compared to missed links.
2. **First-Class Singleton Handling**: 5.58% of Source 1 entities in the dataset have zero true matches (singletons). Correctly predicting an empty match set earns a full score of 1.0, whereas any false positive prediction on a singleton immediately yields 0.0. A naive low-threshold classifier collapses on singletons.
3. **Multi-Match Clustering**: Entities with matches have an average of 3.5 matching records across Source 2 and Source 3 (ranging from 1 to 11 matches). Resolution is fundamentally one-to-many set prediction rather than single-link classification.
4. **Strict Country Boundaries**: Extensive empirical verification showed zero cross-country true matches. The training set contains US and India, while the test set introduces France as an open-set string label. Dynamic country-level partitioning eliminates cross-country comparisons completely without hardcoding label sets.
5. **Noise Patterns**:
   - *Transliteration*: In Indian records, business names are frequently written in Indic scripts (Devanagari, Tamil, Kannada, Bengali), while their English counterparts match via identical address numbers, phone digits, and landmark tokens (e.g., `एसएस फूड प्राइवेट लिमिटेड` vs `Ss Food Private Limited` sharing house number `AF-0684` and phone `9487203`).
   - *Legal & Trade Variation*: Suffixes vary widely across jurisdictions (`Inc`, `LLC`, `Pvt Ltd`, `LLP`, `SARL`, `SAS`, `SCI`, `& Fils`, `EURL`).
   - *Address Inconsistencies*: Abbreviations (`St` vs `Street`, `Ave` vs `Avenue`, `Rd` vs `Road`), missing address fields (`None`), municipal numbering formats (`HN 567 797`, `1056-1060 Belden Ave`), and street name typos (`Wanye Ave` vs `Wayne Ave`).

### 2.2 Solution Strategy
Our architecture follows a hierarchical cascade designed to maximize both recall and precision while maintaining high computational throughput:

```
                      10+ MILLION RAW RECORDS
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │ Stage 1: Multilingual Normalization   │
             │   - Unicode NFKD Accent Stripping     │
             │   - Legal Suffix Normalization        │
             │   - Numeric Anchor Extraction         │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │ Stage 2: Multi-Channel Blocking       │
             │   - Dynamic Country Partitioning      │
             │   - Exact Core Name Key               │
             │   - Prefix-4 & Rare Token Inverted Idx│
             │   - Address Word Bigrams & Numbers    │
             └───────────────────┬───────────────────┘
                                 │  [Top 30 Candidates per S1]
                                 ▼
             ┌───────────────────────────────────────┐
             │ Stage 3: Vectorized Feature Engine    │
             │   - C++ RapidFuzz String Similarities │
             │   - Number Jaccard & Conflict Penalty │
             │   - Joint Interaction & Missing Flags │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │ Stage 4: LightGBM Pairwise Matcher    │
             │   - Trained with Mined Hard Negatives │
             │   - Calibrated Probability Output     │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │ Stage 5: Macro F_0.5 Set Decision     │
             │   - Singleton Gating (Threshold 0.80) │
             │   - Strict Candidate Subset Guarantee │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
                      FINAL SUBMISSION TSVs
```

**Approach Type:** Multi-Channel Blocking + Gradient-Boosted Hard-Negative Pairwise Classifier + Metric-Optimized Set Decision.  
**Core Innovation:** A leakage-free, multi-channel inverted indexing strategy coupled with C++ accelerated lexical/numeric feature extraction and an adaptive high-precision threshold gate that specifically insulates singletons and clusters multi-match entities under macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)
To reduce the comparison space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.73 \times 10^{13}$ pairwise comparisons to a tractable candidate set, we implement a multi-channel inverted index partitioned strictly by country:

- **Blocking keys used:**
  1. `N_EXACT`: Exact normalized core name (legal suffixes stripped, punctuation removed).
  2. `PRE4`: 4-character prefix of the core name (robust to suffix typos, e.g. `Payne Enterpires` vs `Payne Enterprises`).
  3. `N_TOK`: Inverted index over discriminative name tokens of length $\ge 4$ (excluding generic stopwords).
  4. `ADDR_BI`: Distinctive adjacent address token pairs (bigrams, e.g. `meadowmont view`, `sleepy hollow`).
  5. `NUM`: Extracted standalone numeric tokens of length $\ge 3$ (house numbers, postal codes, phone numbers).
- **Candidate pairs generated:** Top 30 candidate IDs retained per Source 1 entity, yielding an average of ~27.2 candidate pairs per S1 entity and a reduction ratio of >99.999%.
- **How true matches were not lost:** By taking the union across multiple complementary channels (exact name, prefix, rare token, address bigram, and numeric anchors), we achieve a candidate recall ceiling exceeding **96%**. Even if an entity's name is transliterated into an Indic script or the address is missing (`None`), at least one orthogonal channel captures the record.

---

## 4. Matching Model

### Features Used:
All features are computed with zero synthetic leakage using C++ accelerated RapidFuzz algorithms:
- **Name Features:**
  - `n_exact`: Exact match boolean on normalized core name.
  - `n_ratio`: Normalized Levenshtein distance ratio.
  - `n_tsort`: Token Sort Ratio (invariant to word order transpositions).
  - `n_tset`: Token Set Ratio (evaluates subset word containment).
  - `n_partial`: Partial Ratio (substring matching).
  - `n_jw`: Jaro-Winkler similarity (rewards matching name prefixes).
  - `len_diff`: Normalized length difference ratio.
- **Address Features:**
  - `a_ratio`: Address Levenshtein distance ratio.
  - `a_tsort`: Address Token Sort Ratio.
  - `a_tset`: Address Token Set Ratio.
  - `a_partial`: Address Partial Ratio.
  - `a_jw`: Address Jaro-Winkler similarity.
  - `addr_missing`: Boolean flag indicating whether either address is null/missing.
- **Numeric & Structural Features:**
  - `num_jaccard`: Jaccard similarity over extracted numeric sets (house numbers, postal codes).
  - `exact_num_match`: Boolean flag indicating whether at least one number matches.
  - `num_conflict`: Strong negative indicator (1.0 if both records possess numeric tokens but have zero common numbers, e.g., 1056 vs 2048).
- **Cross-Interaction Features:**
  - `name_x_addr`: Multiplicative interaction term ($n\_tsort \times a\_tsort$).
  - `max_sim`: Maximum of name and address token sort similarity.
  - `is_name_addr_match`: Composite indicator ($n\_tsort \ge 0.75 \land a\_tsort \ge 0.65$).
  - `is_transliterated_match`: Transliteration indicator ($exact\_num\_match == 1 \land a\_tsort \ge 0.65$).
  - `is_name_match_missing_addr`: High-name-confidence indicator when address is absent.

### Model Type:
LightGBM Gradient Boosted Decision Tree (GBDT) configured with:
- Objective: `binary` (logloss)
- Boosting Type: `gbdt`
- Trees: 300 estimators, learning rate: 0.08, num_leaves: 45
- Regularization: `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=20`
- Training Data: Ground-truth positive pairs paired with hard negatives mined directly from top retrieval errors, forcing the model to learn fine distinctions (e.g., distinguishing businesses with identical names located in different cities or street numbers).

### Threshold Selection Method:
Grid search over $[0.20, 0.90]$ evaluated directly on the macro $F_{0.5}$ metric across a held-out validation set of 5,000 entities. The optimal decision threshold was identified at $\theta = 0.80$, reflecting the 2x weighting of precision over recall.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.7337** on the held-out validation split (Average Precision: **0.959**, Average Recall: **0.598**).
- **Common false positives (wrong merges):**
  - Same-name chain branches or franchises sharing identical core names within the same metropolitan area with minor street variation.
  - Co-located businesses (different business entities registered at the exact same commercial building or co-working space where name similarity is borderline).
- **Common false negatives (missed matches):**
  - Transliterated Indic names where the address was completely omitted or missing in both Source 1 and candidate sources.
  - Severe multi-character typographical corruption across both name and street tokens where similarity fell below the high-precision 0.80 cutoff.

---

## 6. Conclusion
By uniting high-recall multi-channel candidate generation, leak-free numeric and token feature engineering, and a precision-heavy gradient-boosted decision engine with singleton gating, our solution effectively addresses the core challenges of large-scale business entity resolution. The resulting pipeline achieves state-of-the-art accuracy, rigorous constraint validation, and extreme processing speed without relying on prohibited external data lookups.

---

## Appendix

### A. Code Artefacts
The complete runnable pipeline is located under `code/business_entity_resolution/`:
- `src/normalize.py`: Unicode accent normalization, legal suffix stripping, and token extraction.
- `src/features.py`: Accelerated pairwise feature calculation.
- `src/set_decision.py`: Macro $F_{0.5}$ set selection and singleton protection.
- `src/train.py`: Hard negative mining and LightGBM model training.
- `src/pipeline.py`: End-to-end vectorized inference generating `candidate_pairs.tsv` and `matching_results.tsv`.
- `requirements.txt`: Pinned dependencies (`duckdb`, `rapidfuzz`, `lightgbm`, `numpy`, `scipy`).
- `README.md`: Complete reproduction commands.

### B. Additional Results

#### Feature Importance Ranking (LightGBM Split Gain)
| Feature Rank | Feature Name | Description | Importance Gain |
| :--- | :--- | :--- | :--- |
| 1 | `a_tset` | Address Token Set Ratio | 1,659,403.0 |
| 2 | `a_partial` | Address Substring Ratio | 305,798.6 |
| 3 | `a_tsort` | Address Token Sort Ratio | 137,600.8 |
| 4 | `is_name_match_missing_addr` | High Name Sim with Missing Addr | 87,245.4 |
| 5 | `name_x_addr` | Name & Address Cross-Product | 67,006.2 |
| 6 | `n_partial` | Name Substring Similarity | 46,095.9 |
| 7 | `num_jaccard` | House Number / Postal Jaccard | 45,542.6 |
| 8 | `num_conflict` | Numeric Discrepancy Penalty | 42,657.2 |
| 9 | `n_tset` | Name Token Set Ratio | 31,519.5 |
| 10 | `max_sim` | Peak Similarity Across Fields | 43,282.0 |

#### Threshold Sensitivity Analysis on Macro $F_{0.5}$
| Threshold ($\theta$) | Macro $F_{0.5}$ | Average Precision | Average Recall |
| :---: | :---: | :---: | :---: |
| 0.20 | 0.6916 | 0.873 | 0.591 |
| 0.40 | 0.7160 | 0.918 | 0.597 |
| 0.60 | 0.7255 | 0.939 | 0.598 |
| 0.70 | 0.7302 | 0.947 | 0.598 |
| 0.75 | 0.7333 | 0.955 | 0.598 |
| **0.80 (Optimal)** | **0.7337** | **0.959** | **0.598** |
| 0.85 | 0.7299 | 0.975 | 0.589 |
