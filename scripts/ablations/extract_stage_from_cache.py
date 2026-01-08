# extract_stage_from_cache_final.py
import json
import pandas as pd
from pathlib import Path
import argparse
import re

def main(args):
    cache_dir = Path(args.cache_dir)
    stage_name = args.stage
    extract_key = args.extract_key
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning cache for stage: {stage_name} in {cache_dir}...")

    # New pattern: stage_id<interview_id>_q<question_id>_<hash>.json
    # Example: synthesis_id76071229_qQ111_f3f0401a5717.json
    pattern = re.compile(rf'^{stage_name}_id(\d+)_q(Q\d+)_*', re.IGNORECASE)


    cache_files = [f for f in cache_dir.glob("*.json") if pattern.match(f.name)]

    if not cache_files:
        print("No files found for this stage! Check stage name and cache dir.")
        return

    data = []
    for file in cache_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            response = content.get(extract_key, f'ERROR: No {extract_key} key')

        # Parse filename
        filename = file.stem
        match = pattern.match(filename)
        if match:
            interview_id = match.group(1)
            question_id = match.group(2)  # Includes the 'Q', e.g., Q111

            data.append({
                'interview_id': interview_id,
                'question_id': question_id,
                'simulated_answer': response,
            })

    # Save to Excel
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(['interview_id', 'question_id'])
        df.to_csv(output_file, index=False)
        print(f"\nExtracted {len(df)} responses for stage '{stage_name}' → {output_file}")
        print("Columns: interview_id, question_id, simulated_answer")
    else:
        print("No data extracted! Check regex or files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific stage responses from cache (format: stage_idXXXX_qQXX_hash.json)")
    parser.add_argument('--cache_dir', type=str, default="cache_batch", help="Path to cache directory")
    parser.add_argument('--stage', type=str, required=True, help="Stage name (e.g., synthesis)")
    parser.add_argument('--extract_key', type=str, required=True, help="Target extraction key")
    parser.add_argument('--output_file', type=str, default="extracted_stage.xlsx", help="Output Excel file")
    args = parser.parse_args()
    main(args)