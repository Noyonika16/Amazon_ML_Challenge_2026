#!/usr/bin/env python3
"""
Packaging script for Amazon ML Challenge 2026: Business Entity Resolution.
1. Validates output/matching_results.tsv and output/candidate_pairs.tsv using utils/validate_submission.py.
2. Packages the entire submission into <team_name>_submission.zip adhering strictly to competition specifications:
   <team_name>_submission.zip
   ├── output/
   │   ├── matching_results.tsv
   │   └── candidate_pairs.tsv
   ├── code/
   │   └── business_entity_resolution/
   │       ├── src/
   │       ├── README.md
   │       └── requirements.txt
   └── Documentation_template.md
"""

import os
import sys
import zipfile
import subprocess
import argparse

def package(team_name="Team_EntityResolvers"):
    print(f"=== Packaging Submission for {team_name} ===")
    
    # 1. Run local validation check
    val_script = os.path.join("utils", "validate_submission.py")
    match_tsv = os.path.join("output", "matching_results.tsv")
    cand_tsv = os.path.join("output", "candidate_pairs.tsv")
    test_dir = os.path.join("dataset", "test")
    
    print("\nRunning official submission validator...")
    cmd = [
        sys.executable, val_script,
        "--matching", match_tsv,
        "--candidate", cand_tsv,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("ERROR: Validation failed! Please fix issues before packaging.", file=sys.stderr)
        sys.exit(1)
        
    print("\nValidation PASSED successfully!")
    
    # 2. Create Zip Package
    zip_filename = f"{team_name}_submission.zip"
    print(f"\nCreating submission zip: {zip_filename} ...")
    
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add output files
        zipf.write(match_tsv, arcname=os.path.join("output", "matching_results.tsv"))
        zipf.write(cand_tsv, arcname=os.path.join("output", "candidate_pairs.tsv"))
        print("  Added output/matching_results.tsv")
        print("  Added output/candidate_pairs.tsv")
        
        # Add documentation template
        doc_path = "Documentation_template.md"
        if os.path.exists(doc_path):
            zipf.write(doc_path, arcname="Documentation_template.md")
            print("  Added Documentation_template.md")
        else:
            print("WARNING: Documentation_template.md not found in root.", file=sys.stderr)
            
        # Add code/business_entity_resolution/
        code_root = os.path.join("code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_root):
            # Skip caches
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=".")
                zipf.write(file_path, arcname=rel_path)
                print(f"  Added {rel_path}")
                
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\nSUCCESS: Submission archive created: {zip_filename} ({zip_size_mb:.2f} MB)")
    print(f"Ready for leaderboard upload and final package submission!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission archive.")
    parser.add_argument("--team-name", default="Team_EntityResolvers", help="Your team name")
    args = parser.parse_args()
    package(args.team_name)
