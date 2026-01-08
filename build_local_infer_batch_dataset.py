# build_llamafactory_stage_dataset_global.py
import argparse
import importlib
import json
import pandas as pd
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any

from agents.bigfive import BigFiveTheory
from agents.mbti import MBTITheory
from register_dataset import register_dataset


def get_cache_key(stage_instance, question: str, options: str, personality_profile: dict,
                  constraints: dict, include_metadata: bool, **extra) -> str:
    key = {
        "question": question if getattr(stage_instance, 'question_dependent', True) else None,
        "options": options if getattr(stage_instance, 'question_dependent', True) else None,
        "constraints": constraints,
        "include_metadata": include_metadata,
        "demographics": extra.get("demographics", {}),
        "prev_output": extra.get("prev_output", {}) if getattr(stage_instance, 'question_dependent', True) else None,
    }
    key = {k: v for k, v in key.items() if v is not None}
    json_str = json.dumps(key, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()[:12]


def import_class_from_string(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_all_previous_outputs(prev_dir: Path, full_config: dict) -> Dict[tuple, Dict[str, Any]]:
    """
    Load all previous stage outputs from files in prev_dir.
    Returns a dict: (country, interview_id, question_id or "*") -> {stage_name: answer, ...}
    For question-independent stages, use "*" as question_id.
    """
    cumulative = {}
    if not prev_dir or not prev_dir.exists():
        return cumulative

    stages_configs = {s['name']: s for s in full_config.get('stages', [])}

    for jsonl_path in prev_dir.glob("*_outputs.jsonl"):
        stage_name = jsonl_path.stem.replace("_matched_outputs", "")
        stage_conf = stages_configs.get(stage_name, {})
        is_question_dependent = stage_conf.get("question_dependent", True)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                meta = data["metadata"]
                country = meta["country"]
                iid = meta["interview_id"]
                if is_question_dependent:
                    q_id = meta["question_id"]
                else:
                    q_id = "*"
                key = (country, iid, q_id)
                if key not in cumulative:
                    cumulative[key] = {}

                if isinstance(data["answer"], list):
                    cumulative[key]["stacks"] = data["answer"]
                else:
                    if isinstance(data["answer"], str):
                        data["answer"] = json.loads(data["answer"])
                    cumulative[key].update(data["answer"])
    return cumulative


def append_result(jsonl_path: Path, answer: dict, metadata: dict):
    """Append current stage result to its own outputs file"""
    entry = {"answer": answer, "metadata": metadata}
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')


def main(args):
    excel_dir = Path(args.excel_dir)
    config_path = Path(args.config_path)

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    global_extra = {k: v for k, v in full_config.items() if k != "stages"}
    stage_config = next((s for s in full_config["stages"] if s["name"] == args.stage_name), None)
    if not stage_config:
        raise ValueError(f"Stage '{args.stage_name}' not found")

    produce_dataset = args.produce_dataset

    # Current stage persistent results file
    results_jsonl = Path(f"{args.stage_name}_outputs.jsonl")

    # Load ALL previous stages accumulatively
    prev_all = load_all_previous_outputs(Path(args.prev_stages_dir) if args.prev_stages_dir else None, full_config)

    TheoryClass = MBTITheory if args.theory_name == "MBTI" else BigFiveTheory
    theory_obj = TheoryClass()

    StageClass = import_class_from_string(stage_config["class"])
    stage_instance = StageClass()

    stage_extra = {k: v for k, v in stage_config.items() if k not in {"name", "class"}}
    extra_kwargs = {"global_constraint_config": global_extra.get("constraint_config", {}), **stage_extra}
    if "prompt_paths" in stage_config:
        inc_meta = stage_extra.get("include_metadata", False)
        extra_kwargs["prompt_path"] = stage_extra["prompt_paths"]["default"] if not inc_meta else \
        stage_extra["prompt_paths"]["metadata"]
        if "snippet_paths" in stage_config["prompt_paths"]:
            extra_kwargs["snippet_paths"] = stage_extra["prompt_paths"]["snippet_paths"]

    excel_files = list(excel_dir.glob("*.xlsx"))
    print(f"Found {len(excel_files)} country Excel files")

    dataset_entries = [] if produce_dataset else None
    in_memory_cache = {}  # For question-independent reuse within run

    for excel_path in excel_files:
        country = excel_path.stem.split('_')[0].replace('_', ' ')
        print(f"  → Processing {country}")

        human_chara = pd.read_excel(excel_path, sheet_name='Human_chara', header=1)
        qa_pair = pd.read_excel(excel_path, sheet_name='QA_pairs', header=0)
        q_ids = qa_pair.columns[2:].tolist()
        questions = {q_id: qa_pair[q_id].iloc[0] for q_id in q_ids}

        for _, row in human_chara.iterrows():
            iid = str(row['Interview ID'])
            demographics = row.to_dict()
            personality_profile = theory_obj.predict_from_demographics(demographics)
            constraints = {}

            is_question_dependent = stage_config.get("question_dependent", True)

            if not is_question_dependent:
                rep_q = "Q1"
                key_exact = (country, iid, rep_q)
                key_wild = (country, iid, "*")
                prev_output = {}
                prev_output.update(prev_all.get(key_wild, {}))
                prev_output.update(prev_all.get(key_exact, {}))
                cache_key = (country, iid)
                if cache_key in in_memory_cache:
                    result = in_memory_cache[cache_key]
                    prompt = "REUSED_FROM_CACHE"
                else:
                    prompt, raw_output = stage_instance.execute(
                        question="DUMMY",
                        options="",
                        personality_profile=personality_profile,
                        constraints=constraints,
                        batch_size=1,
                        demographics=demographics,
                        prev_output=json.dumps(prev_output),
                        **extra_kwargs
                    )
                    result = raw_output if raw_output not in [None, ""] else getattr(stage_instance,
                                                                                     'last_python_result', {})
                    in_memory_cache[cache_key] = result

                # Save current result
                metadata = {"country": country, "interview_id": iid, "question_id": rep_q, "stage": args.stage_name}
                append_result(results_jsonl, result, metadata)

                if produce_dataset:
                    dataset_entries.append({
                        "instruction": prompt,
                        "input": "",
                        "output": "",
                        "metadata": metadata
                    })

            else:
                for q_id, q_desc in questions.items():
                    key_exact = (country, iid, q_id)
                    key_wild = (country, iid, "*")
                    prev_output = {}
                    prev_output.update(prev_all.get(key_wild, {}))
                    prev_output.update(prev_all.get(key_exact, {}))
                    question = q_desc.split("\n")[0]
                    options = q_desc.split("\n")[-1]
                    extra_kwargs["chara_summary"] = prev_output.pop("chara_summary")
                    extra_kwargs["stress_level"] = prev_output.pop("stress_level")
                    if args.theory_name == "MBTI" and args.stage_name == "reason":
                        prv_reason_output = prev_output.pop("stacks")
                        extra_kwargs={**extra_kwargs, **prev_output}
                        prev_output = prv_reason_output
                    elif args.theory_name == "big5" and args.stage_name == "reason":
                        prv_reason_output = prev_output.pop("stacks")
                        extra_kwargs = {**extra_kwargs, **prev_output}
                        prev_output = prv_reason_output
                        for item in prv_reason_output:
                            item["level"] = extra_kwargs[item["trait"].lower()]["level"]
                        # print(extra_kwargs)
                        # print(prev_output)

                    prompt, raw_output = stage_instance.execute(
                        question=question,
                        options=options,
                        personality_profile=personality_profile,
                        constraints=constraints,
                        batch_size=1,
                        demographics=demographics,
                        prev_output=json.dumps(prev_output),
                        **extra_kwargs
                    )
                    result = raw_output if raw_output not in [None, ""] else getattr(stage_instance,
                                                                                     'last_python_result', {})

                    # Save current result
                    metadata = {"country": country, "interview_id": iid, "question_id": q_id, "stage": args.stage_name}
                    append_result(results_jsonl, result, metadata)

                    if produce_dataset:
                        dataset_entries.append({
                            "instruction": prompt,
                            "input": "",
                            "output": "",
                            "metadata": metadata
                        })

    # Generate dataset only if requested
    if produce_dataset:
        output_jsonl_path = Path(args.output_jsonl)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        register_dataset(
            dataset_name=f"{args.dataset_prefix}_{args.theory_name}_{args.stage_name}",
            dataset_file=str(output_jsonl_path).replace("data/", "", 1),
            description=f"Batch dataset for {args.stage_name}"
        )
        print(f"\nDataset written: {output_jsonl_path} ({len(dataset_entries)} entries)")
    else:
        print(f"\nStage '{args.stage_name}' ran in pure Python mode → no dataset generated.")

    print(f"Current stage results saved to: {results_jsonl}")
    print(f"Put this file into your previous stages folder for the next stage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_dir', required=True)
    parser.add_argument('--theory_name', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--stage_name', required=True)
    parser.add_argument('--prev_stages_dir', default=None,
                        help="Folder containing all previous *_outputs.jsonl files (e.g. prev_stages/)")
    parser.add_argument('--output_jsonl', default="next_stage_dataset.jsonl",
                        help="Output dataset path (only used if --produce_dataset)")
    parser.add_argument('--produce_dataset', action='store_true', default=False,
                        help="Generate LlamaFactory dataset JSONL and register it (default: False)")
    parser.add_argument('--dataset_prefix', default="test")
    args = parser.parse_args()
    main(args)