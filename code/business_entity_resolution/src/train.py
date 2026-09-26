import os
import sys
import time
import duckdb
import numpy as np
import lightgbm as lgb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address, extract_numbers
from features import compute_pair_features, FEATURE_NAMES

def compute_entity_f_beta(pred_set, true_set, beta=0.5):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set.intersection(true_set))
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    beta_sq = beta ** 2
    return (1.0 + beta_sq) * prec * rec / (beta_sq * prec + rec)

def main():
    print("Training Business Entity Resolution Matcher")
    os.makedirs("models", exist_ok=True)
    con = duckdb.connect()
    
    # 1. Load representative training sample (60,000 S1 records)
    print("Loading training S1 entities and ground truth...")
    t0 = time.time()
    s1_rows = con.execute('''
        SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, gt.matched_entity_ids
        FROM read_csv('dataset/train/train_source1.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}) s1
        JOIN read_csv('dataset/train/train_ground_truth.tsv', sep='\\t', header=true, columns={'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}) gt
          ON s1.entity_id = gt.source1_entity_id
        LIMIT 60000
    ''').fetchall()
    
    print(f"Loaded {len(s1_rows):,} S1 training entities in {time.time()-t0:.2f}s")
    
    # 50k train, 10k validation
    train_s1 = s1_rows[:50000]
    val_s1 = s1_rows[50000:55000] # 5k held-out for validation
    
    train_match_ids = set()
    train_gt_map = {}
    for s1_id, _, _, _, gt_str in train_s1:
        if gt_str and gt_str.strip():
            m_ids = [m.strip() for m in gt_str.split(',') if m.strip()]
            train_gt_map[s1_id] = set(m_ids)
            train_match_ids.update(m_ids)
        else:
            train_gt_map[s1_id] = set()
            
    val_match_ids = set()
    val_gt_map = {}
    for s1_id, _, _, _, gt_str in val_s1:
        if gt_str and gt_str.strip():
            m_ids = [m.strip() for m in gt_str.split(',') if m.strip()]
            val_gt_map[s1_id] = set(m_ids)
            val_match_ids.update(m_ids)
        else:
            val_gt_map[s1_id] = set()

    all_needed_ids = train_match_ids.union(val_match_ids)
    print(f"Total true matching IDs needed: {len(all_needed_ids):,}")
    
    # Load candidate pool records (S2 + S3)
    print("Loading candidate pool records...")
    t0 = time.time()
    s2_sample = con.execute('''
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source2.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
        LIMIT 100000
    ''').fetchall()
    
    s3_sample = con.execute('''
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source3.tsv', sep='\\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
        LIMIT 100000
    ''').fetchall()
    
    loaded_ids = {r[0] for r in s2_sample} | {r[0] for r in s3_sample}
    missing_ids = list(all_needed_ids - loaded_ids)
    print(f"Fetching {len(missing_ids):,} missing true records from dataset...")
    for i in range(0, min(len(missing_ids), 120000), 5000):
        chk = missing_ids[i:i+5000]
        id_str = "'" + "','".join(chk) + "'"
        q2 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source2.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
        q3 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source3.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
        s2_sample.extend(con.execute(q2).fetchall())
        s3_sample.extend(con.execute(q3).fetchall())
        
    cand_records = s2_sample + s3_sample
    print(f"Total candidate records loaded: {len(cand_records):,} in {time.time()-t0:.2f}s")
    
    cand_dict = {}
    for eid, name, addr, country in cand_records:
        norm_n = normalize_name(name)
        norm_a = normalize_address(addr)
        nums = extract_numbers(addr)
        cand_dict[eid] = (name, addr, country, norm_n, norm_a, nums)

    # 2. Build multi-channel blocking index
    print("Building blocking index for hard negative mining...")
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
            if i >= 2: break
        for num in list(nums)[:2]:
            if len(num) >= 3:
                index[('NUM', country, num)].append(eid)

    print(f"Index built with {len(index):,} keys in {time.time()-t0:.2f}s")

    # 3. Create Training Pairs (Positives + Mined Hard Negatives)
    print("Generating training pairs (Positives + Mined Hard Negatives)...")
    t0 = time.time()
    X_train = []
    y_train = []
    
    for s1_id, s1_name, s1_addr, s1_c, _ in train_s1:
        norm_n = normalize_name(s1_name)
        norm_a = normalize_address(s1_addr)
        nums = extract_numbers(s1_addr)
        true_set = train_gt_map.get(s1_id, set())
        
        for m_id in true_set:
            if m_id in cand_dict:
                _, _, _, c_norm_n, c_norm_a, c_nums = cand_dict[m_id]
                feats = compute_pair_features(
                    norm_n, norm_a, nums,
                    c_norm_n, c_norm_a, c_nums
                )
                X_train.append(feats)
                y_train.append(1)
                
        keys = []
        if norm_n and len(norm_n) >= 3: keys.append(('N_EXACT', s1_c, norm_n))
        if norm_n and len(norm_n) >= 4: keys.append(('PRE4', s1_c, norm_n[:4]))
        for tok in [t for t in norm_n.split() if len(t) >= 4][:2]:
            keys.append(('N_TOK', s1_c, tok))
        a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
        for i in range(len(a_tokens) - 1):
            keys.append(('ADDR_BI', s1_c, a_tokens[i], a_tokens[i+1]))
            if i >= 2: break
        for num in list(nums)[:2]:
            if len(num) >= 3: keys.append(('NUM', s1_c, num))
            
        cand_scores = defaultdict(int)
        for k in keys:
            for cid in index.get(k, [])[:40]:
                if cid not in true_set:
                    cand_scores[cid] += 1
                    
        hard_negs = sorted(cand_scores.keys(), key=lambda c: cand_scores[c], reverse=True)[:4]
        for hn_id in hard_negs:
            if hn_id in cand_dict:
                _, _, _, c_norm_n, c_norm_a, c_nums = cand_dict[hn_id]
                feats = compute_pair_features(
                    norm_n, norm_a, nums,
                    c_norm_n, c_norm_a, c_nums
                )
                X_train.append(feats)
                y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Generated {len(X_train):,} training pairs (Pos: {np.sum(y_train==1):,}, HardNeg: {np.sum(y_train==0):,}) in {time.time()-t0:.2f}s")

    # 4. Train LightGBM Model
    print("Training LightGBM Classifier...")
    t0 = time.time()
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.08,
        'num_leaves': 45,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data)
    model_path = "models/lgbm_matcher.txt"
    model.save_model(model_path)
    print(f"Model trained and saved to {model_path} in {time.time()-t0:.2f}s")

    print("\nFeature Importances:")
    importances = model.feature_importance(importance_type='gain')
    for fn, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fn:<24}: {imp:.1f}")

    # 5. Evaluate on Validation Set & Tune Macro F_0.5 Threshold
    print(f"\nEvaluating on {len(val_s1):,} Validation Entities & Tuning F_0.5 Threshold...")
    t0 = time.time()
    val_predictions = []
    
    for s1_id, s1_name, s1_addr, s1_c, _ in val_s1:
        norm_n = normalize_name(s1_name)
        norm_a = normalize_address(s1_addr)
        nums = extract_numbers(s1_addr)
        true_set = val_gt_map.get(s1_id, set())
        
        keys = []
        if norm_n and len(norm_n) >= 3: keys.append(('N_EXACT', s1_c, norm_n))
        if norm_n and len(norm_n) >= 4: keys.append(('PRE4', s1_c, norm_n[:4]))
        for tok in [t for t in norm_n.split() if len(t) >= 4][:3]:
            keys.append(('N_TOK', s1_c, tok))
        a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
        for i in range(len(a_tokens) - 1):
            keys.append(('ADDR_BI', s1_c, a_tokens[i], a_tokens[i+1]))
            if i >= 3: break
        for num in list(nums)[:3]:
            if len(num) >= 3: keys.append(('NUM', s1_c, num))
            
        scores = defaultdict(int)
        for k in keys:
            for cid in index.get(k, [])[:80]:
                scores[cid] += 1
                
        cands = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:35]
        
        cand_pairs = []
        for rank, cid in enumerate(cands):
            if cid in cand_dict:
                _, _, _, c_norm_n, c_norm_a, c_nums = cand_dict[cid]
                feats = compute_pair_features(
                    norm_n, norm_a, nums,
                    c_norm_n, c_norm_a, c_nums
                )
                cand_pairs.append((cid, feats))
                
        if cand_pairs:
            X_cand = np.array([f for _, f in cand_pairs], dtype=np.float32)
            probs = model.predict(X_cand)
            val_predictions.append((s1_id, true_set, [(cid, float(p)) for (cid, _), p in zip(cand_pairs, probs)]))
        else:
            val_predictions.append((s1_id, true_set, []))

    print(f"Validation inference completed in {time.time()-t0:.2f}s")
    
    # Grid search threshold to maximize Macro F_0.5
    best_thresh = 0.5
    best_f05 = 0.0
    
    for thresh in np.arange(0.20, 0.90, 0.05):
        scores = []
        precs = []
        recs = []
        for s1_id, true_set, cand_probs in val_predictions:
            pred_set = set(cid for cid, p in cand_probs if p >= thresh)
            f_val = compute_entity_f_beta(pred_set, true_set, beta=0.5)
            scores.append(f_val)
            if pred_set and true_set:
                tp = len(pred_set.intersection(true_set))
                precs.append(tp / len(pred_set))
                recs.append(tp / len(true_set))
                
        macro_f05 = np.mean(scores)
        avg_p = np.mean(precs) if precs else 0
        avg_r = np.mean(recs) if recs else 0
        print(f"  Threshold {thresh:.2f} -> Macro F_0.5: {macro_f05:.4f} | Avg Precision: {avg_p:.3f} | Avg Recall: {avg_r:.3f}")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh
            
    print(f"\n>>> BEST THRESHOLD: {best_thresh:.2f} with Macro F_0.5 = {best_f05:.4f} <<<")
    
    with open("models/optimal_threshold.txt", "w") as f:
        f.write(f"{best_thresh:.4f}\n")

if __name__ == "__main__":
    main()
