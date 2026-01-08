# refine_vllm_cache.py
import json
import re
from json import JSONDecodeError
from pathlib import Path
import argparse
from typing import List, Optional, Set

from register_dataset import register_dataset
from scripts.evaluate_cluster_question_group import extract_option_mapping, map_answer_to_letter


def extract_answer(predict: str, baseline_parse=False, mapping=None) -> Optional[dict]:
    """Extract structured JSON answer from model prediction."""
    cleaned = re.sub(r'<think>.*?</think>', '', predict, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    if not baseline_parse:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    else:
        return map_answer_to_letter(cleaned, mapping)



def load_lines(path: Path) -> List[str]:
    """Load all non-empty lines from a JSONL file."""
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file as list of dictionaries."""
    try:
        lines = load_lines(path)
        return [json.loads(line) for line in lines]
    except JSONDecodeError:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_jsonl(path: Path, entries: List[dict]):
    """Save list of dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def load_retry_indices(path: Path) -> Set[int]:
    """Load retry indices from txt file (0-based)."""
    if not path.exists():
        return set()
    with open(path, 'r') as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def save_retry_indices(path: Path, indices: Set[int]):
    """Save retry indices to txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for idx in sorted(indices):
            f.write(f"{idx}\n")


def refine_cache_logic(
    vllm_output_dir: str,
    dataset_path: str,
    stage_name: str,
    main_output_path: str = "vllm_outputs.jsonl",
    retry_output_path: str = "retry_outputs.jsonl",
    matched_output_path: str = "matched_outputs.jsonl",
    retry_dataset_path: str = "retry_dataset.jsonl",
    retry_indices_path: str = "retry_indices.txt",
    theory_name: str = "MBTI"
):
    output_dir = Path(vllm_output_dir)
    main_file = output_dir / main_output_path
    retry_file = output_dir / retry_output_path
    retry_indices_file = output_dir / retry_indices_path

    # Load dataset
    dataset_entries = load_jsonl(Path(dataset_path))
    total_dataset = len(dataset_entries)
    print(f"Loaded {total_dataset} entries from dataset")

    # Load main outputs
    main_lines = load_lines(main_file)
    print(f"Loaded {len(main_lines)} entries from main")
    total_main = len(main_lines)

    # Load retry indices (lines to potentially replace)
    retry_indices = load_retry_indices(retry_indices_file)
    print(f"Loaded {len(retry_indices)} retry indices")

    # Load retry outputs if present
    retry_lines = load_lines(retry_file)
    total_retry = len(retry_lines)

    # If retry outputs exist and match the number of retry indices, update main
    if total_retry > 0:
        if total_retry != len(retry_indices):
            print(f"Warning: Retry outputs ({total_retry}) don't match retry indices ({len(retry_indices)})")
            print("→ Skipping update. Check files.")
        else:
            # Sort indices to replace in order
            sorted_indices = sorted(retry_indices)
            retry_idx = 0
            new_main_lines = []
            for i, line in enumerate(main_lines):
                if i in retry_indices:
                    # Replace with retry line
                    new_main_lines.append(retry_lines[retry_idx])
                    retry_idx += 1
                else:
                    new_main_lines.append(line)

            # Update main file
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_main_lines) + '\n')
            print(f"Updated main file with {total_retry} retry lines")

            # Clear retry file and indices after successful update
            open(retry_file, 'w').close()
            save_retry_indices(retry_indices_file, set())
            retry_indices = set()
            print("Cleared retry file and indices")

            # Reload main lines after update
            main_lines = load_lines(main_file)
            total_main = len(main_lines)

    # Now process with updated main
    main_outputs = []
    for line in main_lines:
        try:
            data = json.loads(line)
            main_outputs.append(data.get('predict', ''))
        except json.JSONDecodeError:
            main_outputs.append('')

    total_final = len(main_outputs)

    # Check for length mismatch
    if total_final != total_dataset:
        print(f"Length mismatch! Dataset: {total_dataset} | Outputs: {total_final}")
        print("→ Forcing FULL retry.")

        save_jsonl(Path(retry_dataset_path), dataset_entries)
        # Save all indices for full retry
        full_indices = set(range(total_dataset))
        save_retry_indices(retry_indices_file, full_indices)
        print(f"Full retry dataset and indices saved")
        return

    # Lengths match → perform matching
    print("Lengths match! Proceeding with matching...")

    matched = []
    new_retry_indices = set()
    retry_dataset = []

    for idx, (ds_entry, predict) in enumerate(zip(dataset_entries, main_outputs)):
        if args.baseline_parse:
            mapping = extract_option_mapping(ds_entry["instruction"])

        answer = extract_answer(predict, baseline_parse=args.baseline_parse, mapping=mapping)

        if answer is not None:
            matched.append({
                "answer": answer,
                "metadata": ds_entry["metadata"] if "metadata" in ds_entry.keys() else
                {"interview_id": ds_entry["interview_id"], "question_id": ds_entry["question_id"], "country": ds_entry["country"], "baseline": ds_entry["baseline"]},
            })
        else:
            new_retry_indices.add(idx)
            retry_dataset.append(ds_entry)

            if predict.strip():
                print(f"Failed parse at index {idx}: {predict[:100]}...")

    # Save matched
    save_jsonl(Path(matched_output_path), matched)

    # Save retry dataset and update indices
    save_jsonl(Path(retry_dataset_path), retry_dataset)
    save_retry_indices(retry_indices_file, new_retry_indices)

    print(f"\nMatched: {len(matched)} / {total_dataset}")
    print(f"  → Saved to {matched_output_path}")
    print(f"Need retry: {len(retry_dataset)}")
    print(f"  → Dataset to {retry_dataset_path}")
    print(f"  → Indices to {retry_indices_file}")

    if len(retry_dataset) == 0:
        print("🎉 All entries matched!")

        # Create stage-specific matched output
        stage_matched_path = output_dir / f"{stage_name}_matched_outputs.jsonl"
        save_jsonl(stage_matched_path, matched)
        print(f"Created stage matched file: {stage_matched_path}")
    else:
        register_dataset(
            dataset_name=f"{stage_name}_{theory_name}_retries",
            dataset_file=str(retry_dataset_path).replace("data/", ""),
            description=f"Batch dataset for {args.stage_name}"
        )
        print("Run inference on retry_dataset.jsonl → save to retry_outputs.jsonl")
        print("Then re-run this script to update main file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLLM cache refiner with retry index tracking"
    )
    parser.add_argument('--vllm_output_dir', type=str, required=True,
                        help="Directory with vllm_outputs.jsonl, retry_outputs.jsonl, etc.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Original dataset JSONL")
    parser.add_argument('--stage_name', type=str, required=True,
                        help="Stage name for final matched output file")
    parser.add_argument('--main_output_path', type=str, default="vllm_outputs.jsonl")
    parser.add_argument('--theory_name', type=str, default="vllm_outputs.jsonl")
    parser.add_argument('--retry_output_path', type=str, default="retry_outputs.jsonl")
    parser.add_argument('--matched_output_path', type=str, default="matched_outputs.jsonl")
    parser.add_argument('--retry_dataset_path', type=str, default="retry.jsonl")
    parser.add_argument('--retry_indices_path', type=str, default="retry_indices.txt")
    parser.add_argument('--baseline_parse', type=bool, default=False)
    args = parser.parse_args()

    refine_cache_logic(
        args.vllm_output_dir,
        args.dataset_path,
        args.stage_name,
        args.main_output_path,
        args.retry_output_path,
        args.matched_output_path,
        args.retry_dataset_path,
        args.retry_indices_path,
        args.theory_name
    )