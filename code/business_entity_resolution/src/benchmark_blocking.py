import sys
import time
import duckdb
from collections import defaultdict

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_name, normalize_address, extract_numbers, get_core_tokens

print("Connecting to DuckDB and loading sample train data...")
con = duckdb.connect()

# Load 10,000 S1 records with their ground truth matches
t0 = time.time()
val_s1_records = con.execute('''
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, gt.matched_entity_ids
    FROM read_csv('dataset/train/train_source1.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}) s1
    JOIN read_csv('dataset/train/train_ground_truth.tsv', sep='\t', header=true, columns={'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}) gt
      ON s1.entity_id = gt.source1_entity_id
    LIMIT 20000
''').fetchall()

print(f"Loaded {len(val_s1_records):,} validation S1 records in {time.time()-t0:.2f}s")

# Extract all true match IDs needed for these S1 records
target_match_ids = set()
gt_map = {}
for s1_id, s1_name, s1_addr, s1_c, gt_str in val_s1_records:
    if gt_str and gt_str != 'None' and gt_str.strip():
        m_ids = [m.strip() for m in gt_str.split(',') if m.strip()]
        gt_map[s1_id] = set(m_ids)
        target_match_ids.update(m_ids)
    else:
        gt_map[s1_id] = set()

print(f"Total S1 entities: {len(val_s1_records):,}, Entities with >=1 match: {len([k for k, v in gt_map.items() if v]):,}")
print(f"Total true matching IDs across S2/S3: {len(target_match_ids):,}")

print("Loading S2 and S3 candidate pool (including all target match IDs + 100,000 background records)...")
t0 = time.time()
s2_records = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source2.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    LIMIT 150000
''').fetchall()

s3_records = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source3.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    LIMIT 150000
''').fetchall()

# Also fetch any target_match_ids that weren't in the top 150k
s2_ids_loaded = {r[0] for r in s2_records}
s3_ids_loaded = {r[0] for r in s3_records}
missing_matches = target_match_ids - s2_ids_loaded - s3_ids_loaded
print(f"Target match IDs not in first 150k: {len(missing_matches):,}. Fetching missing true records...")

if missing_matches:
    missing_list = list(missing_matches)
    # Batch query in chunks
    for i in range(0, min(len(missing_list), 50000), 5000):
        chunk = missing_list[i:i+5000]
        id_str = "'" + "','".join(chunk) + "'"
        m_s2 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source2.tsv', sep='\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})").fetchall()
        m_s3 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source3.tsv', sep='\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})").fetchall()
        s2_records.extend(m_s2)
        s3_records.extend(m_s3)

pool_records = s2_records + s3_records
print(f"Total candidate pool size: {len(pool_records):,} records. Loaded in {time.time()-t0:.2f}s")

print("Building Multi-Channel Blocking Indices...")
t0 = time.time()
name_exact_index = defaultdict(list)
name_token_index = defaultdict(list)
addr_num_index = defaultdict(list)
postal_index = defaultdict(list)

GENERIC_WORDS = {'ltd', 'pvt', 'inc', 'llc', 'the', 'and', 'corp', 'company', 'services', 'solutions', 'enterprises'}

for r in pool_records:
    eid, name, addr, country = r
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    nums = extract_numbers(addr)
    
    # 1. Exact core name
    if norm_n and len(norm_n) >= 3:
        name_exact_index[(country, norm_n)].append(eid)
        
    # 2. Name tokens
    tokens = get_core_tokens(norm_n, min_len=4, stopwords=GENERIC_WORDS)
    for tok in tokens:
        name_token_index[(country, tok)].append(eid)
        
    # 3. Address numbers
    for num in nums:
        if len(num) >= 3:
            addr_num_index[(country, num)].append(eid)
            
    # 4. Postal / Phone numbers (5 to 10 digits)
    for num in nums:
        if 5 <= len(num) <= 10:
            postal_index[(country, num)].append(eid)

print(f"Index built in {time.time()-t0:.2f}s:")
print(f"  Exact name keys: {len(name_exact_index):,}")
print(f"  Token keys: {len(name_token_index):,}")
print(f"  Address number keys: {len(addr_num_index):,}")
print(f"  Postal/Phone keys: {len(postal_index):,}")

print("\nRetrieving candidates for validation S1 records...")
t0 = time.time()
total_true_matches = sum(len(v) for v in gt_map.values())
retrieved_true_matches = 0
total_candidates = 0

for s1_id, s1_name, s1_addr, s1_country, _ in val_s1_records:
    norm_n = normalize_name(s1_name)
    norm_a = normalize_address(s1_addr)
    nums = extract_numbers(s1_addr)
    
    candidate_scores = defaultdict(int)
    
    # Channel 1: Exact Name (weight 10)
    if norm_n and (s1_country, norm_n) in name_exact_index:
        for cid in name_exact_index[(s1_country, norm_n)]:
            candidate_scores[cid] += 10
            
    # Channel 2: Name Tokens (weight 4, max 5 tokens)
    tokens = list(get_core_tokens(norm_n, min_len=4, stopwords=GENERIC_WORDS))[:5]
    for tok in tokens:
        cids = name_token_index.get((s1_country, tok), [])
        if len(cids) <= 500: # Filter overly frequent tokens
            for cid in cids:
                candidate_scores[cid] += 4
                
    # Channel 3: Address numbers (weight 3)
    for num in nums:
        if len(num) >= 3:
            cids = addr_num_index.get((s1_country, num), [])
            if len(cids) <= 300:
                for cid in cids:
                    candidate_scores[cid] += 3
                    
    # Channel 4: Postal / Phone (weight 5)
    for num in nums:
        if 5 <= len(num) <= 10:
            cids = postal_index.get((s1_country, num), [])
            if len(cids) <= 100:
                for cid in cids:
                    candidate_scores[cid] += 5
                    
    # Sort and take top 25 candidates
    sorted_cands = sorted(candidate_scores.keys(), key=lambda c: candidate_scores[c], reverse=True)[:25]
    cand_set = set(sorted_cands)
    total_candidates += len(cand_set)
    
    # Evaluate recall against ground truth
    true_set = gt_map.get(s1_id, set())
    if true_set:
        retrieved_true_matches += len(cand_set.intersection(true_set))

elapsed = time.time() - t0
avg_cands = total_candidates / len(val_s1_records)
recall = retrieved_true_matches / total_true_matches if total_true_matches > 0 else 0

print(f"\nBENCHMARK RESULTS ({len(val_s1_records):,} S1 Entities) ===")
print(f"Time taken: {elapsed:.2f}s ({len(val_s1_records)/elapsed:.0f} entities/sec)")
print(f"Candidate Recall: {recall*100:.2f}% ({retrieved_true_matches:,} / {total_true_matches:,})")
print(f"Average candidates per S1: {avg_cands:.1f}")
