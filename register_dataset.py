# register_dataset.py
import json
import argparse
from pathlib import Path

def register_dataset(
    dataset_name: str,
    dataset_file: str,
    description: str = "Custom WVS-based opinion simulation dataset",
):
    """
    Automatically add a new dataset to dataset_info.json in LlamaFactory format.
    """
    info_path = Path("data") / "dataset_info.json"
    if not info_path.exists():
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps({}, indent=2))

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Prepare new entry
    new_entry = {
        "file_name": dataset_file,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        },
        "description": description,
    }

    if dataset_name in info:
        print(f"Warning: {dataset_name} already exists. Overwriting.")
    info[dataset_name] = new_entry

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully registered dataset '{dataset_name}'")
    print(f"  File: {dataset_file}")
    print(f"  Updated: {info_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a dataset to LlamaFactory's dataset_info.json")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (e.g., wvs_opinion_200)")
    parser.add_argument("--dir", type=str, default="data", help="Dataset directory (relative)")
    parser.add_argument("--file", type=str, required=True, help="Dataset file name (e.g., Brazil_opinion_200.json)")
    parser.add_argument("--desc", type=str, default="Custom WVS-based opinion simulation dataset")
    args = parser.parse_args()

    register_dataset(
        dataset_name=args.name,
        dataset_dir=args.dir,
        dataset_file=args.file,
        description=args.desc
    )
