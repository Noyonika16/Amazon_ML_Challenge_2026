"""
Self-contained validation benchmark testing the trained LightGBM model
on 5,000 validation S1 entities with guaranteed candidate dictionary coverage.
Measures:
- Candidate Recall
- Entity-level Precision, Recall
- Macro F_0.5 Score across all entities (including singletons)
"""

import os
import sys
import time
import duckdb
import numpy as np
import lightgbm as lgb
from collections import defaultdict

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_name, normalize_address, extract_numbers
from features import compute_pair_features, FEATURE_NAMES
from train import compute_entity_f_beta

print("=== Running Validation Benchmark ===")
con = duckdb.connect()

# Load 5,000 validation entities
t0 = time.time()
val_s1 = con.execute('''
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, gt.matched_entity_ids
    FROM read_csv('dataset/train/train_source1.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}) s1
    JOIN read_csv('dataset/train/train_ground_truth.tsv', sep='\\t', header=true, columns={'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}) gt
      ON s1.entity_id = gt.source1_entity_id
    WHERE s1.country = 'US'
    LIMIT 5000
''').fetchall()

print(f"Loaded {len(val_s1):,} US validation S1 records in {time.time()-t0:.2f}s")

val_gt_map = {}
val_target_ids = set()
for s1_id, _, _, _, gt_str in val_s1:
    if gt_str and gt_str.strip():
        m_ids = [m.strip() for m in gt_str.split(',') if m.strip()]
        val_gt_map[s1_id] = set(m_ids)
        val_target_ids.update(m_ids)
    else:
        val_gt_map[s1_id] = set()

print(f"Entities with matches: {len([k for k, v in val_gt_map.items() if v]):,}, Singletons: {len([k for k, v in val_gt_map.items() if not v]):,}")
print(f"Total true match IDs: {len(val_target_ids):,}")

# Load candidate pool: 100k US records from S2 and S3 + ALL target IDs
t0 = time.time()
s2_us = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source2.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    WHERE country = 'US'
    LIMIT 80000
''').fetchall()

s3_us = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source3.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    WHERE country = 'US'
    LIMIT 80000
''').fetchall()

loaded = {r[0] for r in s2_us} | {r[0] for r in s3_us}
missing = list(val_target_ids - loaded)
print(f"Fetching ALL {len(missing):,} remaining true match records...")

# Load all missing in chunks of 5000
for i in range(0, len(missing), 5000):
    chk = missing[i:i+5000]
    id_str = "'" + "','".join(chk) + "'"
    q2 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source2.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
    q3 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source3.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
    s2_us.extend(con.execute(q2).fetchall())
    s3_us.extend(con.execute(q3).fetchall())

pool = s2_us + s3_us
print(f"Total candidate pool records loaded: {len(pool):,} in {time.time()-t0:.2f}s")

cand_dict = {}
for eid, name, addr, country in pool:
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    nums = extract_numbers(addr)
    cand_dict[eid] = (name, addr, country, norm_n, norm_a, nums)

# Build blocking index
print("Building blocking index...")
t0 = time.time()
STOPWORDS = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'opp', 'floor', 'block', 'sector', 'city', 'state', 'india', 'null'}
index = defaultdict(list)

for eid, (name, addr, country, norm_n, norm_a, nums) in cand_dict.items():
    if norm_n and len(norm_n) >= 3:
        index[('N_EXACT', country, norm_n)].append(eid)
    if norm_n and len(norm_n) >= 4:
        index[('PRE4', country, norm_n[:4])].append(eid)
    n_tokens = [t for t in norm_n.split() if len(t) >= 4]
    for tok in n_tokens[:3]:
        index[('N_TOK', country, tok)].append(eid)
    a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
    for i in range(len(a_tokens) - 1):
        index[('ADDR_BI', country, a_tokens[i], a_tokens[i+1])].append(eid)
        if i >= 3: break
    for num in list(nums)[:3]:
        if len(num) >= 3:
            index[('NUM', country, num)].append(eid)

print(f"Index built with {len(index):,} keys in {time.time()-t0:.2f}s")

# Load model
model = lgb.Booster(model_file="models/lgbm_matcher.txt")
print("Model loaded successfully.")

# Run Candidate Generation & Scoring
print("Running candidate generation and LightGBM scoring...")
t0 = time.time()
val_eval_data = []
total_true_matches = sum(len(v) for v in val_gt_map.values())
retrieved_true_matches = 0
total_candidates = 0

for s1_id, s1_name, s1_addr, s1_country, _ in val_s1:
    norm_n = normalize_name(s1_name)
    norm_a = normalize_address(s1_addr)
    nums = extract_numbers(s1_addr)
    true_set = val_gt_map.get(s1_id, set())
    
    keys = []
    if norm_n and len(norm_n) >= 3: keys.append(('N_EXACT', s1_country, norm_n))
    if norm_n and len(norm_n) >= 4: keys.append(('PRE4', s1_country, norm_n[:4]))
    for tok in [t for t in norm_n.split() if len(t) >= 4][:3]:
        keys.append(('N_TOK', s1_country, tok))
    a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
    for i in range(len(a_tokens) - 1):
        keys.append(('ADDR_BI', s1_country, a_tokens[i], a_tokens[i+1]))
        if i >= 3: break
    for num in list(nums)[:3]:
        if len(num) >= 3: keys.append(('NUM', s1_country, num))
        
    scores = defaultdict(int)
    for k in keys:
        for cid in index.get(k, [])[:150]:
            scores[cid] += 1
            
    cands = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:35]
    top_set = set(cands)
    total_candidates += len(top_set)
    
    if true_set:
        retrieved_true_matches += len(top_set.intersection(true_set))
        
    cand_pairs = []
    for rank, cid in enumerate(cands):
        c_name, c_addr, _, c_norm_n, c_norm_a, c_nums = cand_dict[cid]
        feats = compute_pair_features(
            s1_name, s1_addr, c_name, c_addr,
            norm_n, norm_a, nums,
            c_norm_n, c_norm_a, c_nums,
            scores[cid], rank
        )
        cand_pairs.append((cid, feats))
        
    if cand_pairs:
        X_cand = np.array([f for _, f in cand_pairs], dtype=np.float32)
        probs = model.predict(X_cand)
        val_eval_data.append((s1_id, true_set, [(cid, float(p)) for (cid, _), p in zip(cand_pairs, probs)]))
    else:
        val_eval_data.append((s1_id, true_set, []))

elapsed = time.time() - t0
cand_recall = retrieved_true_matches / total_true_matches if total_true_matches > 0 else 0
print(f"Scored {len(val_s1):,} entities in {elapsed:.2f}s ({len(val_s1)/elapsed:.0f} ent/s)")
print(f"Candidate Recall: {cand_recall*100:.2f}% ({retrieved_true_matches:,} / {total_true_matches:,})")
print(f"Average candidates per S1: {total_candidates / len(val_s1):.1f}")

# Grid Search F_0.5
print("\n--- Macro F_0.5 Evaluation across Thresholds ---")
best_thresh = 0.5
best_f05 = 0.0

for thresh in np.arange(0.20, 0.95, 0.05):
    f_scores = []
    precs = []
    recs = []
    for s1_id, true_set, cand_probs in val_eval_data:
        pred_set = set(cid for cid, p in cand_probs if p >= thresh)
        f = compute_entity_f_beta(pred_set, true_set, beta=0.5)
        f_scores.append(f)
        if pred_set and true_set:
            tp = len(pred_set.intersection(true_set))
            precs.append(tp / len(pred_set))
            recs.append(tp / len(true_set))
            
    macro_f05 = np.mean(f_scores)
    avg_p = np.mean(precs) if precs else 0
    avg_r = np.mean(recs) if recs else 0
    print(f"  Threshold {thresh:.2f} -> Macro F_0.5: {macro_f05:.4f} | Avg Precision: {avg_p:.3f} | Avg Recall: {avg_r:.3f}")
    if macro_f05 > best_f05:
        best_f05 = macro_f05
        best_thresh = thresh

print(f"\n==========================================")
print(f"OPTIMAL THRESHOLD: {best_thresh:.2f} WITH MACRO F_0.5 = {best_f05:.4f}!")
print(f"==========================================")
