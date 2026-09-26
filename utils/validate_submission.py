#!/usr/bin/env python3
"""
Submission Validator for Business Entity Resolution Challenge.
Checks both matching_results.tsv and candidate_pairs.tsv against all competition rules:
1. Correct file format (tab-separated, exact column headers).
2. Exactly one row per test Source 1 entity.
3. No duplicate rows.
4. Matched entity IDs must only reference valid entities from test Source 2 or Source 3.
5. No duplicate IDs within any ID list.
6. Every matched ID in matching_results.tsv must appear in candidate_pairs.tsv for that entity.
"""

import argparse
import os
import sys

def validate(matching_path, candidate_path, test_dir):
    issues = []
    
    # Check paths
    if not os.path.exists(matching_path):
        print(f"ERROR: Matching file not found: {matching_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(candidate_path):
        print(f"ERROR: Candidate file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)
        
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    for p in [test_s1_path, test_s2_path, test_s3_path]:
        if not os.path.exists(p):
            print(f"ERROR: Test source file not found: {p}", file=sys.stderr)
            sys.exit(1)
            
    print("Loading test entity IDs...")
    valid_s1_ids = set()
    with open(test_s1_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[id_idx]:
                valid_s1_ids.add(parts[id_idx])
                
    valid_s2_s3_ids = set()
    for s_path in [test_s2_path, test_s3_path]:
        with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[id_idx]:
                    valid_s2_s3_ids.add(parts[id_idx])
                    
    print(f"Loaded {len(valid_s1_ids):,} valid S1 IDs and {len(valid_s2_s3_ids):,} valid S2/S3 IDs.")

    # 1. Validate Candidate Pairs
    print(f"Validating candidate file: {candidate_path} ...")
    candidate_map = {}
    seen_s1_candidates = set()
    with open(candidate_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().rstrip("\r\n")
        if first_line != "source1_entity_id\tcandidate_entity_ids":
            issues.append(f"Candidate file header mismatch. Expected 'source1_entity_id\\tcandidate_entity_ids', got '{first_line}'")
            
        for line_no, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                cand_str = ""
            elif len(parts) == 2:
                s1_id, cand_str = parts
            else:
                issues.append(f"Candidate file line {line_no} has invalid number of tab columns ({len(parts)}).")
                continue
                
            if s1_id in seen_s1_candidates:
                issues.append(f"Candidate file duplicate source1_entity_id: {s1_id} at line {line_no}")
            seen_s1_candidates.add(s1_id)
            
            cands = [c.strip() for c in cand_str.split(",") if c.strip()]
            if len(cands) != len(set(cands)):
                issues.append(f"Candidate file line {line_no} ({s1_id}) contains duplicate candidate IDs.")
            
            cand_set = set(cands)
            for c_id in cands:
                if c_id.startswith("S1-"):
                    issues.append(f"Candidate file line {line_no} ({s1_id}) contains self-match to Source 1 ID: {c_id}")
                    break
                if c_id not in valid_s2_s3_ids:
                    issues.append(f"Candidate file line {line_no} ({s1_id}) contains candidate ID not in test S2/S3: {c_id}")
                    break
            candidate_map[s1_id] = cand_set
            
    if seen_s1_candidates != valid_s1_ids:
        missing = valid_s1_ids - seen_s1_candidates
        extra = seen_s1_candidates - valid_s1_ids
        if missing:
            issues.append(f"Candidate file is missing {len(missing):,} Source 1 entities from the test set.")
        if extra:
            issues.append(f"Candidate file has {len(extra):,} Source 1 entities not in the test set.")

    # 2. Validate Matching Results
    print(f"Validating matching file: {matching_path} ...")
    seen_s1_matching = set()
    with open(matching_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().rstrip("\r\n")
        if first_line != "source1_entity_id\tmatched_entity_ids":
            issues.append(f"Matching file header mismatch. Expected 'source1_entity_id\\tmatched_entity_ids', got '{first_line}'")
            
        for line_no, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                match_str = ""
            elif len(parts) == 2:
                s1_id, match_str = parts
            else:
                issues.append(f"Matching file line {line_no} has invalid number of tab columns ({len(parts)}).")
                continue
                
            if s1_id in seen_s1_matching:
                issues.append(f"Matching file duplicate source1_entity_id: {s1_id} at line {line_no}")
            seen_s1_matching.add(s1_id)
            
            matches = [m.strip() for m in match_str.split(",") if m.strip()]
            if len(matches) != len(set(matches)):
                issues.append(f"Matching file line {line_no} ({s1_id}) contains duplicate matched IDs.")
                
            match_set = set(matches)
            for m_id in matches:
                if m_id.startswith("S1-"):
                    issues.append(f"Matching file line {line_no} ({s1_id}) contains self-match to Source 1 ID: {m_id}")
                    break
                if m_id not in valid_s2_s3_ids:
                    issues.append(f"Matching file line {line_no} ({s1_id}) contains matched ID not in test S2/S3: {m_id}")
                    break
                    
            # Check subset constraint
            if s1_id in candidate_map:
                diff = match_set - candidate_map[s1_id]
                if diff:
                    issues.append(f"Matching file line {line_no} ({s1_id}) contains matched IDs that were not in candidate_pairs: {list(diff)[:3]}")

    if seen_s1_matching != valid_s1_ids:
        missing = valid_s1_ids - seen_s1_matching
        extra = seen_s1_matching - valid_s1_ids
        if missing:
            issues.append(f"Matching file is missing {len(missing):,} Source 1 entities from the test set.")
        if extra:
            issues.append(f"Matching file has {len(extra):,} Source 1 entities not in the test set.")

    if issues:
        print(f"\nFAILED: Found {len(issues)} validation issue(s):")
        for i, issue in enumerate(issues[:25], start=1):
            print(f"{i}. {issue}")
        if len(issues) > 25:
            print(f"... and {len(issues) - 25} more issues.")
        sys.exit(1)
    else:
        print("\nPASS")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()
    
    validate(args.matching, args.candidate, args.test_dir)
