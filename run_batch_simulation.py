# run_batch_simulation.py
import argparse
import os
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from vllm import LLM, SamplingParams

from agents.mbti import MBTITheory
from agents.bigfive import BigFiveTheory
from agents.llm_helper.async_batch_client import AsyncBatchClient
from agents.mindset.process_combination import ProcessCombination
from agents.utils import parse_messy_json, parse_messy_json_with_fallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Stages that are pure Python → no LLM call
PURE_PYTHON_STAGES = ("get_stack", "get_traits")

# Stages that run once per human (not per question)
PERSON_INDEPENDENT_STAGES = {
    "mbti": ["stress_chara", "mbti_select", "get_stack"],
    "bigfive": ["stress_chara", "bigfive_select", "get_traits"]
}

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--theory_name', type=str, default="mbti", required=True, help='Personality cognitive theory')
    parser.add_argument('--config_path', type=str, default="config/example_mbti.yml", help='Path to the config file')
    parser.add_argument('--llm_provider', type=str, default='zhipuai', help='LLM provider')
    parser.add_argument('--llm_model', type=str, default='glm-4-flash', help='LLM model')
    parser.add_argument('--base_url', type=str, default=None, help='API endpoint url')
    parser.add_argument('--api_key', type=str, default=None, help='API key for LLM provider')
    parser.add_argument('--include_metadata', action='store_true', help='Include metadata in the output')
    parser.add_argument('--output_path', type=str, default='output', help='Path to the output dir')
    parser.add_argument("--max_questions_per_person", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--debug_mode", action='store_true', help="Enable debugging with logging/caching")
    parser.add_argument("--debug_cache_dir", type=str, default="debug_cache", help="Directory for cached stage responses")
    parser.add_argument("--batch_size", type=int, default=40000, help="Requests per batch job (max 50000 for ZhipuAI)")
    parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument("--max_polls", type=int, default=300, help="Max poll attempts (~5 hours)")
    parser.add_argument("--chunk_size", type=int, default=40000, help="Requests per sub-batch chunk")
    return parser.parse_args()

def run_batch_simulation(arguments):
    data_path = arguments.data_path
    exp_name = arguments.exp_name
    theory_name = arguments.theory_name
    config_path = arguments.config_path
    llm_provider = arguments.llm_provider
    llm_model = arguments.llm_model
    base_url = arguments.base_url
    api_key = arguments.api_key
    include_metadata = arguments.include_metadata
    output_path = arguments.output_path
    max_questions_per_person = arguments.max_questions_per_person
    debug_mode = arguments.debug_mode
    debug_cache_dir = arguments.debug_cache_dir
    batch_size = arguments.batch_size
    poll_interval = arguments.poll_interval
    max_polls = arguments.max_polls
    chunk_size = arguments.chunk_size

    # Validate provider for batch

    start_time = datetime.now()
    log.info("=== Batch Simulation Started ===")
    log.info(f"Data: {data_path} | Theory: {theory_name} | Model: {llm_model} | Batch Size: {batch_size}")
    log.info(f"Metadata: {'ON' if include_metadata else 'OFF'} | Output: {output_path}")

    # Create output dir
    exp_name = f"{theory_name}-{exp_name}"
    full_output_path = os.path.join(output_path, exp_name)
    os.makedirs(full_output_path, exist_ok=True)
    # ================================================================
    # 1. SINGLE LLM ENGINE INITIALIZATION (outside all stages!)
    # ================================================================
    if arguments.llm_provider == "vllm":
        llm = LLM(
            model=llm_model,
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=4096
        )
    elif arguments.llm_provider in ["zhipuai", "openai", "anthropic"]:
        llm = AsyncBatchClient(llm_provider, llm_model, api_key=api_key, base_url=base_url)
        log.info(f"{arguments.llm_provider} async batch client ready")

    else:
        raise ValueError(f"Unsupported provider: {arguments.llm_provider}")

    # Load data
    human_chara = pd.read_excel(data_path, sheet_name='Human_chara', header=0).iloc[1:]
    qa_pair = pd.read_excel(data_path, sheet_name='QA_pair', header=0)
    q_ids = qa_pair.columns[2:].tolist()
    questions = {q_id: qa_pair[q_id].iloc[0] for q_id in q_ids}

    # Init theory
    TheoryClass = MBTITheory if theory_name == "mbti" else BigFiveTheory
    theory_obj = TheoryClass()

    # Init pipeline
    combo = ProcessCombination.build_from_config(config_path=config_path)

    # Get ordered stages
    stage_names = [s.name for s in combo.stages]
    indep_stages = PERSON_INDEPENDENT_STAGES[arguments.theory_name]
    log.info(f"Stages: {stage_names}")

    simulated = []

    # Main loop: one full batch per stage
    for stage_idx, stage_name in enumerate(stage_names):
        log.info(f"\n=== STAGE {stage_idx+1}/{len(stage_names)}: {stage_name} ===")

        batch_prompts = []
        batch_metadata = []
        parsed_results = []
        is_independent = stage_name in indep_stages

        for _, row in human_chara.iterrows():
            interview_id = str(row["D_INTERVIEW"])
            cluster = row["cluster"]
            demographics = row.to_dict()
            profile = theory_obj.predict_from_demographics(demographics)

            if is_independent:
                # Run ONCE per human, reuse for all questions
                cache_key = f"{interview_id}_{stage_name}"
                cache_path = os.path.join(debug_cache_dir, f"{cache_key}.json")
                if os.path.exists(cache_path):
                    continue

                state = {
                    "demographics": demographics,
                    "batch_size": arguments.batch_size
                }

                # Build state from previous independent stages
                for prev_stage in stage_names[:stage_idx]:
                    prev_key = f"{interview_id}_{prev_stage}"
                    prev_path = os.path.join(debug_cache_dir, f"{prev_key}.json")
                    if os.path.exists(prev_path):
                        with open(prev_path) as f:
                            state["prev_output"] = json.load(f)["output"]

                # Get prompt (no question)
                stage_wrapper = combo.stages[stage_idx]

                if stage_name in PURE_PYTHON_STAGES:
                    _, parsed_result = stage_wrapper.execute(
                        question="", options="", personality_profile=profile.copy(),
                        constraints={}, include_metadata=include_metadata, **state
                    )
                    parsed_results.append(parsed_result)
                else:
                    prompt, _ = stage_wrapper.execute(
                        question="", options="", personality_profile=profile.copy(),
                        constraints={}, include_metadata=include_metadata, **state
                    )
                    batch_prompts.append(prompt)
                    custom_id = f"{int(interview_id):04d}-{stage_name}"
                    batch_metadata.append({
                        "cluster": cluster,
                        "interview_id": interview_id,
                        "stage": stage_name,
                        "custom_id": custom_id,
                        "cache_path": cache_path
                    })
            else:
                # Question-dependent: per (human, question)
                for q_id in q_ids:
                    if max_questions_per_person and len(batch_metadata) >= max_questions_per_person:
                        break

                    q_text = questions[q_id]
                    question = q_text.split("?")[0] + "?" if "?" in q_text else q_text
                    options = "?".join(q_text.split("?")[1:]).strip() if "?" in q_text else ""

                    cache_key = f"{interview_id}_{q_id}_{stage_name}"
                    cache_path = os.path.join(debug_cache_dir if debug_mode else "cache_batch", f"{cache_key}.json")
                    if os.path.exists(cache_path):
                        continue

                    state = {
                        "demographics": demographics,
                        "interview_id": interview_id,
                        "batch_size": arguments.batch_size,
                        "q_id": q_id,
                    }

                    # Load previous (independent or dependent)
                    for prev_stage in stage_names[:stage_idx]:
                        if prev_stage in indep_stages:
                            prev_key = f"{interview_id}_{prev_stage}"
                        else:
                            prev_key = f"{interview_id}_{q_id}_{prev_stage}"
                        prev_path = os.path.join(debug_cache_dir if debug_mode else "cache_batch", f"{prev_key}.json")
                        if os.path.exists(prev_path):
                            with open(prev_path) as f:
                                state["prev_output"] = json.load(f)["output"]

                    # Get prompt
                    stage_wrapper = combo.stages[stage_idx]
                    prompt, _ = stage_wrapper.execute(question, options, profile.copy(), constraints={}, include_metadata=include_metadata, **state)

                    batch_prompts.append(prompt)
                    batch_metadata.append({
                        "cluster": cluster,
                        "interview_id": interview_id,
                        "q_id": q_id,
                        "stage": stage_name,
                        "custom_id": f"{int(interview_id):04d}-{q_id}-{stage_name}",
                        "cache_path": cache_path
                    })

        if not batch_prompts:
            log.info(f"Stage {stage_name}: all cached, skipping")
            continue

        log.info(f"Running {len(batch_prompts)} requests for {stage_name}...")
        if arguments.llm_provider == "vllm":
            # vLLM real batch inference
            for i in range(0, len(batch_prompts), arguments.batch_size):
                batch = batch_prompts[i:i + arguments.batch_size]
                outputs = llm.generate(batch, sampling_params)
                raw_texts = [out.outputs[0].text.strip() for out in outputs]
                parsed_batch = [
                    json.dumps(parse_messy_json_with_fallback(text, prompt), ensure_ascii=False)
                    for text, prompt in zip(raw_texts, batch)
                ]
                parsed_results.extend(parsed_batch)
        else:
            # ZhipuAI async batch (unchanged)
            batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]  # Convert prompts to messages
            custom_ids = [m["custom_id"] for m in batch_metadata]
            results = llm.submit_and_wait(
                batch_messages,
                custom_ids=custom_ids,
                temperature=0.7,
                max_tokens=4096
            )
            parsed_results = [json.dumps(parse_messy_json_with_fallback(result, prompt)) for result, prompt in zip(results, batch_prompts)]

        # Save results
        for meta, parsed_result in zip(batch_metadata, parsed_results):
            with open(meta["cache_path"], "w", encoding="utf-8") as f:
                json.dump({"output": parsed_result}, f, ensure_ascii=False, indent=2)

            if stage_name == "synthesis":
                simulated.append({
                    'cluster': meta["cluster"],
                    'interview_id': meta["interview_id"],
                    'question_id': meta["q_id"],
                    'simulated_answer': parsed_result
                })

    # Final save
    out_file = os.path.join(full_output_path, "simulated.csv")
    pd.DataFrame(simulated).to_csv(out_file, index=False)
    log.info(f"Completed in {datetime.now() - start_time}")

if __name__ == "__main__":
    arguments = args()
    run_batch_simulation(arguments)