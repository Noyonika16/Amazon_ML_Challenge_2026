"""
Set Decision Engine for Business Entity Resolution.
Optimized specifically for Macro F_0.5 evaluation metric:
1. Penalizes false merges 2x more than missed matches.
2. Singleton protection: Entities with no match score 1.0 when empty, 0.0 on false merge.
3. Strict subset constraint: matched IDs are always a subset of candidate IDs.
"""

def decide_match_set(cand_scores, threshold=0.80, max_matches=10):
    """
    Given a list of (candidate_id, prob) pairs sorted by prob descending:
    Decides the final predicted set of matched IDs for an S1 entity.
    """
    if not cand_scores:
        return []
        
    top_id, top_p = cand_scores[0]
    
    # Singleton Gate: If highest confidence candidate is below threshold, predict no match
    if top_p < threshold:
        return []
        
    matches = []
    for cid, p in cand_scores[:max_matches]:
        if p >= threshold:
            matches.append(cid)
            
    return matches
