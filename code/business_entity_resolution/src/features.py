from rapidfuzz import fuzz, distance

def compute_pair_features(s1_norm_n, s1_norm_a, s1_nums, cand_norm_n, cand_norm_a, cand_nums):
   
    if s1_norm_n and cand_norm_n:
        n_exact = 1.0 if s1_norm_n == cand_norm_n else 0.0
        n_ratio = fuzz.ratio(s1_norm_n, cand_norm_n) / 100.0
        n_tsort = fuzz.token_sort_ratio(s1_norm_n, cand_norm_n) / 100.0
        n_tset = fuzz.token_set_ratio(s1_norm_n, cand_norm_n) / 100.0
        n_partial = fuzz.partial_ratio(s1_norm_n, cand_norm_n) / 100.0
        n_jw = distance.JaroWinkler.similarity(s1_norm_n, cand_norm_n)
        len_diff = abs(len(s1_norm_n) - len(cand_norm_n)) / max(len(s1_norm_n), len(cand_norm_n), 1)
    else:
        n_exact, n_ratio, n_tsort, n_tset, n_partial, n_jw, len_diff = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

    addr_missing = 1.0 if (not s1_norm_a or not cand_norm_a) else 0.0
    if not addr_missing:
        a_ratio = fuzz.ratio(s1_norm_a, cand_norm_a) / 100.0
        a_tsort = fuzz.token_sort_ratio(s1_norm_a, cand_norm_a) / 100.0
        a_tset = fuzz.token_set_ratio(s1_norm_a, cand_norm_a) / 100.0
        a_partial = fuzz.partial_ratio(s1_norm_a, cand_norm_a) / 100.0
        a_jw = distance.JaroWinkler.similarity(s1_norm_a, cand_norm_a)
    else:
        a_ratio, a_tsort, a_tset, a_partial, a_jw = 0.0, 0.0, 0.0, 0.0, 0.0

    has_s1_nums = bool(s1_nums)
    has_cand_nums = bool(cand_nums)
    
    if has_s1_nums and has_cand_nums:
        common_nums = s1_nums.intersection(cand_nums)
        num_jaccard = len(common_nums) / len(s1_nums.union(cand_nums))
        exact_num_match = 1.0 if common_nums else 0.0
        num_conflict = 1.0 if not common_nums else 0.0
    else:
        num_jaccard = 0.5 if (not has_s1_nums and not has_cand_nums) else 0.0
        exact_num_match = 0.0
        num_conflict = 0.0

    name_x_addr = n_tsort * a_tsort
    max_sim = max(n_tsort, a_tsort)
    
    is_name_addr_match = 1.0 if (n_tsort >= 0.75 and a_tsort >= 0.65) else 0.0
    is_transliterated_match = 1.0 if (exact_num_match == 1.0 and a_tsort >= 0.65) else 0.0
    is_name_match_missing_addr = 1.0 if (addr_missing == 1.0 and n_tsort >= 0.80) else 0.0
    
    features = [
        n_exact, n_ratio, n_tsort, n_tset, n_partial, n_jw, len_diff,
        addr_missing, a_ratio, a_tsort, a_tset, a_partial, a_jw,
        num_jaccard, exact_num_match, num_conflict,
        name_x_addr, max_sim, is_name_addr_match, is_transliterated_match,
        is_name_match_missing_addr
    ]
    return features

FEATURE_NAMES = [
    'n_exact', 'n_ratio', 'n_tsort', 'n_tset', 'n_partial', 'n_jw', 'len_diff',
    'addr_missing', 'a_ratio', 'a_tsort', 'a_tset', 'a_partial', 'a_jw',
    'num_jaccard', 'exact_num_match', 'num_conflict',
    'name_x_addr', 'max_sim', 'is_name_addr_match', 'is_transliterated_match',
    'is_name_match_missing_addr'
]
