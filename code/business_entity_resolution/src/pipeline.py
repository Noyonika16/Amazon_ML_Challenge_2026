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
from set_decision import decide_match_set

STOPWORDS = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'opp', 'floor', 'block', 'sector', 'city', 'state', 'india', 'null'}

def extract_blocking_keys(name, addr, country):
    keys = []
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    
    if norm_n and len(norm_n) >= 3:
        keys.append(('N_EXACT', country, norm_n))
        
    if norm_n and len(norm_n) >= 4:
        keys.append(('PRE4', country, norm_n[:4]))
        
    n_tokens = [t for t in norm_n.split() if len(t) >= 4]
    for tok in n_tokens[:3]:
        keys.append(('N_TOK', country, tok))
        
    a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
    for i in range(len(a_tokens) - 1):
        keys.append(('ADDR_BI', country, a_tokens[i], a_tokens[i+1]))
        if i >= 3: break
        
    nums = extract_numbers(addr)
    for num in list(nums)[:3]:
        if len(num) >= 3:
            keys.append(('NUM', country, num))
            
    return keys, norm_n, norm_a, nums

def process_country(country, test_dir, model, threshold, cand_file, match_file, top_k=30):
      print(f"Processing Country: {country}")
    con = duckdb.connect()
    
    t0 = time.time()
    print(f"Loading Source 2 records for {country}...")
    s2_rows = con.execute(f'''
        SELECT entity_id, business_name, business_address
        FROM read_csv('{test_dir}/test_source2.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}})
        WHERE country = '{country}'
    ''').fetchall()
    
    print(f"Loading Source 3 records for {country}...")
    s3_rows = con.execute(f'''
        SELECT entity_id, business_name, business_address
        FROM read_csv('{test_dir}/test_source3.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}})
        WHERE country = '{country}'
    ''').fetchall()
    
    s23_rows = s2_rows + s3_rows
    print(f"Total candidate pool for {country}: {len(s23_rows):,} records loaded in {time.time()-t0:.2f}s")
    
    print(f"Building blocking index for {country}...")
    t0 = time.time()
    index = defaultdict(list)
    cand_dict = {}
    
    for eid, name, addr in s23_rows:
        keys, norm_n, norm_a, nums = extract_blocking_keys(name, addr, country)
        cand_dict[eid] = (norm_n, norm_a, nums)
        for k in keys:
            index[k].append(eid)
            
    print(f"Index built with {len(index):,} distinct keys in {time.time()-t0:.2f}s")
    del s2_rows, s3_rows, s23_rows
    
    print(f"Loading Source 1 records for {country}...")
    t0 = time.time()
    s1_rows = con.execute(f'''
        SELECT entity_id, business_name, business_address
        FROM read_csv('{test_dir}/test_source1.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}})
        WHERE country = '{country}'
    ''').fetchall()
    print(f"Loaded {len(s1_rows):,} Source 1 records for {country} in {time.time()-t0:.2f}s")
    
    print(f"Running Vectorized Candidate Retrieval & Scoring for {len(s1_rows):,} records...")
    t_start = time.time()
    BATCH_SIZE = 10000
    total_processed = 0
    total_matches = 0
    total_candidates = 0
    
    for b_idx in range(0, len(s1_rows), BATCH_SIZE):
        batch = s1_rows[b_idx:b_idx + BATCH_SIZE]
        
        all_feats = []
        entity_cand_slices = []
        curr_idx = 0
        
        for s1_id, s1_name, s1_addr in batch:
            keys, norm_n, norm_a, nums = extract_blocking_keys(s1_name, s1_addr, country)
            
            # Candidate retrieval
            scores = defaultdict(int)
            for k in keys:
                for cid in index.get(k, [])[:80]:
                    scores[cid] += 1
                    
            cands = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]
            total_candidates += len(cands)
            
            start_idx = curr_idx
            for cid in cands:
                c_norm_n, c_norm_a, c_nums = cand_dict[cid]
                feats = compute_pair_features(
                    norm_n, norm_a, nums,
                    c_norm_n, c_norm_a, c_nums
                )
                all_feats.append(feats)
                curr_idx += 1
            end_idx = curr_idx
            entity_cand_slices.append((s1_id, start_idx, end_idx, cands))
            
        if all_feats:
            X_batch = np.array(all_feats, dtype=np.float32)
            batch_probs = model.predict(X_batch)
        else:
            batch_probs = np.array([], dtype=np.float32)
            
        batch_cand_outputs = []
        batch_match_outputs = []
        
        for s1_id, start_idx, end_idx, cands in entity_cand_slices:
            cand_str = ",".join(cands)
            batch_cand_outputs.append(f"{s1_id}\t{cand_str}\n")
            
            if start_idx == end_idx:
                batch_match_outputs.append(f"{s1_id}\t\n")
            else:
                probs = batch_probs[start_idx:end_idx]
                cand_probs = list(zip(cands, probs))
                cand_probs.sort(key=lambda x: x[1], reverse=True)
                
                matches = decide_match_set(cand_probs, threshold=threshold)
                total_matches += len(matches)
                match_str = ",".join(matches)
                batch_match_outputs.append(f"{s1_id}\t{match_str}\n")
                
        cand_file.writelines(batch_cand_outputs)
        match_file.writelines(batch_match_outputs)
        total_processed += len(batch)
        
        if total_processed % 50000 == 0 or total_processed == len(s1_rows):
            elapsed = time.time() - t_start
            speed = total_processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {total_processed:,} / {len(s1_rows):,} ({total_processed/len(s1_rows)*100:.1f}%) | Speed: {speed:.0f} ent/s | Matches found: {total_matches:,}")

    print(f"Completed {country}: {total_processed:,} entities | {total_candidates:,} candidates | {total_matches:,} matches in {time.time()-t_start:.2f}s")

def main():
    test_dir = "dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    
    model_path = "models/lgbm_matcher.txt"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Train the model first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading trained LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    
    threshold = 0.80
    thresh_file = "models/optimal_threshold.txt"
    if os.path.exists(thresh_file):
        with open(thresh_file, "r") as f:
            try:
                threshold = float(f.read().strip())
            except Exception:
                pass
    print(f"Using Macro F_0.5 optimal decision threshold: {threshold:.4f}")
    
    con = duckdb.connect()
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM read_csv('{test_dir}/test_source1.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) ORDER BY 1").fetchall()]
    print(f"Discovered test countries: {countries}")
    
    t_global = time.time()
    with open(cand_path, "w", encoding="utf-8") as f_cand, open(match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        for country in countries:
            process_country(country, test_dir, model, threshold, f_cand, f_match, top_k=30)
            
    print(f"END-TO-END INFERENCE COMPLETED IN {time.time()-t_global:.2f}s!")
    print(f"Generated:")
    print(f"  {cand_path} ({os.path.getsize(cand_path)/1024/1024:.1f} MB)")
    print(f"  {match_path} ({os.path.getsize(match_path)/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
