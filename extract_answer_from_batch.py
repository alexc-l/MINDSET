import json
import pandas as pd
from pathlib import Path
import argparse


def main(matched_jsonl_path: str, output_dir: str, theory: str = "bigfive"):
    """
    Extract simulated answers from matched_outputs.jsonl and generate
    one cleaned CSV per country with columns: country, interview_id, question_id, simulated_answer.
    Assumes 'answer' in each JSONL entry is a string (the simulated answer).
    If 'answer' is a dict, adjust to extract the relevant key (e.g., answer['simulated_answer']).
    """
    # Load all entries from JSONL
    entries = []
    with open(matched_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                meta = entry['metadata']
                answer = entry['answer']["conclusion"] if isinstance(entry['answer'], dict) else entry['answer'] # Assume string; if dict, use entry['answer']['simulated_answer']
                entries.append({
                    'country': meta['country'],
                    'interview_id': meta['interview_id'],
                    'question_id': meta['question_id'],
                    'simulated_answer': answer
                })

    if not entries:
        print("No entries found in JSONL.")
        return

    # Create DataFrame
    df = pd.DataFrame(entries)

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by country and save one CSV per country
    for country, group_df in df.groupby('country'):
        # Clean country name for filename (replace spaces with underscores)
        country_clean = country.replace(' ', '_')
        csv_filename = f"{theory}-{country_clean}_cleaned.csv"
        csv_path = output_path / csv_filename

        # Sort by interview_id and question_id for consistency
        group_df = group_df.sort_values(['interview_id', 'question_id'])

        # Save to CSV without index
        group_df.to_csv(csv_path, index=False)
        print(f"Saved {len(group_df)} rows to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract simulated answers from matched_outputs.jsonl and generate per-country CSVs."
    )
    parser.add_argument('--matched_jsonl_path', type=str, required=True,
                        help="Path to matched_outputs.jsonl")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save per-country CSVs")
    parser.add_argument('--theory', type=str, default="bigfive",
                        help="Theory prefix for CSV filenames (e.g., bigfive, MBTI)")
    args = parser.parse_args()

    main(args.matched_jsonl_path, args.output_dir, args.theory)