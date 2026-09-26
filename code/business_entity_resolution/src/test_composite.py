import sys, time, duckdb
from collections import defaultdict

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_name, normalize_address, extract_numbers, get_core_tokens

con = duckdb.connect()

print('Loading validation data...')
val_s1 = con.execute('''
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, gt.matched_entity_ids
    FROM read_csv('dataset/train/train_source1.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}) s1
    JOIN read_csv('dataset/train/train_ground_truth.tsv', sep='\t', header=true, columns={'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}) gt
      ON s1.entity_id = gt.source1_entity_id
    LIMIT 20000
''').fetchall()

gt_map = {}
target_ids = set()
for s1_id, _, _, _, gt_str in val_s1:
    if gt_str and gt_str != 'None' and gt_str.strip():
        m_ids = [m.strip() for m in gt_str.split(',') if m.strip()]
        gt_map[s1_id] = set(m_ids)
        target_ids.update(m_ids)
    else:
        gt_map[s1_id] = set()

# Load candidate pool
s2 = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source2.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    LIMIT 150000
''').fetchall()
s3 = con.execute('''
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('dataset/train/train_source3.tsv', sep='\t', header=true, columns={'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'})
    LIMIT 150000
''').fetchall()

loaded_ids = {r[0] for r in s2} | {r[0] for r in s3}
missing = list(target_ids - loaded_ids)
print(f'Fetching {len(missing):,} missing true candidate records...')
for i in range(0, min(len(missing), 50000), 5000):
    chk = missing[i:i+5000]
    id_str = "'" + "','".join(chk) + "'"
    q2 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source2.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
    q3 = f"SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source3.tsv', sep='\\t', header=true, columns={{'entity_id': 'VARCHAR', 'business_name': 'VARCHAR', 'business_address': 'VARCHAR', 'country': 'VARCHAR'}}) WHERE entity_id IN ({id_str})"
    s2.extend(con.execute(q2).fetchall())
    s3.extend(con.execute(q3).fetchall())

pool = s2 + s3
print(f'Candidate pool size: {len(pool):,}')

STOPWORDS = {'road', 'street', 'avenue', 'lane', 'drive', 'near', 'opp', 'floor', 'block', 'sector', 'city', 'state', 'india', 'null'}

def extract_keys(name, addr, country):
    keys = []
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    
    # Key 1: Exact core name
    if norm_n and len(norm_n) >= 3:
        keys.append(('N_EXACT', country, norm_n))
        
    n_tokens = [t for t in norm_n.split() if len(t) >= 3]
    a_tokens = [t for t in norm_a.split() if len(t) >= 3 and t not in STOPWORDS]
    nums = extract_numbers(addr)
    
    # Key 2: Prefix-4 of name
    if norm_n and len(norm_n) >= 4:
        keys.append(('PRE4', country, norm_n[:4]))
        
    # Key 3: Individual significant name tokens (max 4)
    for tok in n_tokens[:4]:
        if len(tok) >= 4:
            keys.append(('N_TOK', country, tok))
            
    # Key 4: Distinctive address word pairs (bigrams)
    for i in range(len(a_tokens) - 1):
        t1, t2 = a_tokens[i], a_tokens[i+1]
        keys.append(('ADDR_BI', country, t1, t2))
        if i >= 4: break
        
    # Key 5: Number + street token
    if nums and a_tokens:
        for num in list(nums)[:3]:
            for tok in a_tokens[:4]:
                if tok != num and not tok.isdigit():
                    keys.append(('NUM_ST', country, num, tok))
                    
    # Key 6: Distinctive address numbers
    for num in list(nums)[:3]:
        if len(num) >= 3:
            keys.append(('NUM', country, num))
            
    return keys

print('Building multi-channel index...')
t0 = time.time()
index = defaultdict(list)
for r in pool:
    eid, name, addr, country = r
    keys = extract_keys(name, addr, country)
    for k in keys:
        index[k].append(eid)
print(f'Index built with {len(index):,} distinct keys in {time.time()-t0:.2f}s')

WEIGHTS = {
    'N_EXACT': 50,
    'NUM_ST': 40,
    'ADDR_BI': 30,
    'N_TOK': 20,
    'NUM': 15,
    'PRE4': 10,
}

MAX_KEY_LIST = {
    'N_EXACT': 300,
    'NUM_ST': 200,
    'ADDR_BI': 200,
    'N_TOK': 200,
    'NUM': 100,
    'PRE4': 100,
}

for top_k in [30, 50, 75]:
    total_true = sum(len(v) for v in gt_map.values())
    retrieved_true = 0
    total_cand_count = 0
    t0 = time.time()

    for s1_id, s1_name, s1_addr, s1_country, _ in val_s1:
        keys = extract_keys(s1_name, s1_addr, s1_country)
        scores = defaultdict(int)
        for k in keys:
            cids = index.get(k, [])
            k_type = k[0]
            max_allowed = MAX_KEY_LIST.get(k_type, 100)
            if len(cids) <= max_allowed:
                w = WEIGHTS.get(k_type, 5)
                for cid in cids:
                    scores[cid] += w
                    
        # Select top_k candidates
        top = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]
        top_set = set(top)
        total_cand_count += len(top_set)
        true_set = gt_map.get(s1_id, set())
        if true_set:
            retrieved_true += len(top_set.intersection(true_set))

    elapsed = time.time() - t0
    print(f'\nTop-{top_k} Candidates:')
    print(f'  Recall: {retrieved_true/total_true*100:.2f}% ({retrieved_true:,} / {total_true:,})')
    print(f'  Avg Cands: {total_cand_count/len(val_s1):.1f}')
    print(f'  Time: {elapsed:.2f}s ({len(val_s1)/elapsed:.0f} ent/s)')
