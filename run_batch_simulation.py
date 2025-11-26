# run_batch_simulation.py
import pandas as pd
import json
import os
import time
from pathlib import Path
from agents.mbti import MBTITheory
from agents.bigfive import BigFiveTheory
from agents.llm_helper.async_batch_client import AsyncBatchClient
import logging

log = logging.getLogger(__name__)

def run_batch_simulation(
    data_path: str,
    theory: str = "mbti",
    model: str = "glm-4-flash",
    cache_dir: str = "cache_batch",
    api_key: str = None,
):
    os.makedirs(cache_dir, exist_ok=True)

    # Load data
    human_df = pd.read_excel(data_path, sheet_name='Human_chara', header=0).iloc[1:]
    qa_df = pd.read_excel(data_path, sheet_name='QA_pair', header=0)
    questions = {col: qa_df[col].iloc[0] for col in qa_df.columns[2:]}

    # Init theory
    TheoryClass = MBTITheory if theory == "mbti" else BigFiveTheory
    theory_obj = TheoryClass()
    combo = theory_obj.build_process_combination()

    # LLM
    llm = AsyncBatchClient("zhipuai", model, api_key=api_key)

    # Get ordered stages
    stage_names = [s.name for s in combo.stages]
    log.info(f"Stages: {stage_names}")

    # Main loop: one full batch per stage
    for stage_idx, stage_name in enumerate(stage_names):
        log.info(f"\n=== STAGE {stage_idx+1}/{len(stage_names)}: {stage_name} ===")

        requests = []
        metadata = []

        for _, row in human_df.iterrows():
            interview_id = str(row["D_INTERVIEW"])
            demographics = row.to_dict()
            profile = theory_obj.predict_from_demographics(demographics)

            for q_id, q_text in questions.items():
                question = q_text.split("?")[0] + "?"
                options = "?".join(q_text.split("?")[1:]).strip()

                cache_key = f"{interview_id}_{q_id}_{stage_name}"
                cache_path = Path(cache_dir) / f"{cache_key}.json"
                if cache_path.exists():
                    continue  # Skip if already done

                # Use combo to get prompt for this stage only
                state = {
                    "demographics": demographics,
                    "personality_profile": profile.copy(),
                    "question": question,
                    "options": options,
                    "interview_id": interview_id,
                    "q_id": q_id,
                    "cache_dir": cache_dir,
                    "api_batch": True
                }

                # Run all previous stages from cache or dummy
                for prev_stage in stage_names[:stage_idx]:
                    prev_key = f"{interview_id}_{q_id}_{prev_stage}"
                    prev_path = Path(cache_dir) / f"{prev_key}.json"
                    if prev_path.exists():
                        with open(prev_path) as f:
                            state["prev_output"] = json.load(f)["output"]

                # Now get current stage prompt
                stage_wrapper = combo.stages[stage_idx]
                prompt = stage_wrapper.execute(
                    question=question,
                    options=options,
                    personality_profile=state["personality_profile"],
                    constraints={},
                    include_metadata=False,
                    **state
                )

                messages = [{"role": "user", "content": prompt}]
                custom_id = f"{int(interview_id):04d}-{q_id}-{stage_name}"

                requests.append(messages)
                metadata.append({
                    "interview_id": interview_id,
                    "q_id": q_id,
                    "stage": stage_name,
                    "custom_id": custom_id,
                    "cache_path": str(cache_path)
                })

        if not requests:
            log.info(f"Stage {stage_name}: all cached, skipping")
            continue

        log.info(f"Sending {len(requests)} requests for {stage_name}...")
        results = llm.submit_and_wait(
            requests,
            subject_id="0000",
            question_ids=[m["custom_id"] for m in metadata],
            temperature=0.7,
            max_tokens=4096
        )

        # Save all results
        for meta, result in zip(metadata, results):
            os.makedirs(os.path.dirname(meta["cache_path"]), exist_ok=True)
            with open(meta["cache_path"], "w", encoding="utf-8") as f:
                json.dump({"output": result}, f, ensure_ascii=False, indent=2)

        log.info(f"Stage {stage_name} completed!")

    log.info("Full batch simulation completed!")

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--theory", choices=["mbti", "bigfive"], default="mbti")
    parser.add_argument("--model", default="glm-4-flash")
    args = parser.parse_args()

    run_batch_simulation(
        data_path=args.data,
        theory=args.theory,
        model=args.model
    )