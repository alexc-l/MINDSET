# run_simulation.py
import argparse
import os

import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any
from agents.mbti import MBTITheory
from agents.bigfive import BigFiveTheory
from agents.llm_helper.llm_client import LLMClient
from agents.mindset.process_combination import ProcessCombination

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Optional: Save logs to file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s | %(levelname)-8s | %(message)s',
#     handlers=[
#         logging.FileHandler(f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#         logging.StreamHandler()
#     ]
# )

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--theory_name', type=str, default="mbti", required=True, help='Personalty cogntive theory')
    parser.add_argument('--config_path', type=str, default="config/example_mbti.yml", help='Path to the config file')
    parser.add_argument('--llm_provider', type=str, default='openai', help='LLM provider')
    parser.add_argument('--llm_model', type=str, default='gpt-4o', help='LLM model')
    parser.add_argument('--base_url', type=str, default=None, help='API endpoint url')
    parser.add_argument('--api_key', type=str, default=None, help='API key for LLM provider')
    parser.add_argument('--include_metadata', action='store_true', help='Include metadata in the output')
    parser.add_argument('--output_path', type=str, default='output', help='Path to the output dir')
    parser.add_argument("--max_questions_per_person", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--debug_mode", action='store_true', help="Enable debugging with logging/caching")
    parser.add_argument("--debug_cache_dir", type=str, default="debug_cache", help="Directory for cached stage responses")
    return parser.parse_args()


def run_simulation(
    data_path: str,
    exp_name: str,
    theory_name: str,
    config_path: str = "config/example_mbti.yml",
    llm_provider: str = 'openai',
    llm_model: str = 'gpt-4o',
    base_url: str = None,
    api_key: str = None,
    include_metadata: bool = False,
    output_path: str = 'output',
    batch_size: int = 0,  # Future: enable batching
    max_questions_per_person: int = None,  # Optional limit for testing
    debug_mode: bool = False,  # NEW: Enable debugging with logging/caching
    debug_cache_dir: str = 'debug_cache',  # NEW: Directory for cached stage responses
    stage_overrides=None
):
    """
    Run MBTI-based personality simulation on survey data.

    Args:
        data_path: Path to Excel file with 'Human_chara' and 'QA_pair' sheets.
        llm_provider: LLM provider ('openai', 'anthropic', etc.)
        llm_model: Model name
        base_url: api url
        api_key: API key (or set via env)
        include_metadata: Include detailed reasoning metadata
        output_path: CSV output path
        batch_size: Future batching support
        max_questions_per_person: Limit questions per person (for debugging)
        debug_mode: Log inputs/outputs per stage; cache responses for token efficiency.
        debug_cache_dir: Folder to store/load JSON caches (e.g., 'debug_cache/stage1_id_1001_q_Q1.json')
    """
    start_time = datetime.now()
    log.info("=== MBTI Simulation Started ===")
    log.info(f"Data: {data_path}")
    log.info(f"Personality Theories: {theory_name}")
    log.info(f"LLM: {llm_provider}/{llm_model}:{base_url}")
    log.info(f"Metadata: {'ON' if include_metadata else 'OFF'}")
    log.info(f"Output: {output_path}")

    # Debugging: Create cache dir if needed
    if debug_mode:
        os.makedirs(debug_cache_dir, exist_ok=True)
        log.info(f"Debug cache dir ready: {debug_cache_dir}")
    # End debugging

    try:
        # Load data
        log.info("Loading Excel data...")
        human_chara = pd.read_excel(data_path, sheet_name='Human_chara', header=1)
        qa_pair = pd.read_excel(data_path, sheet_name='QA_pair', header=0)
        log.info(f"Loaded {len(human_chara)} human profiles, {len(qa_pair.columns)-2} questions")

        # Extract questions
        q_ids = qa_pair.columns[3:].tolist()  # Skip cluster, D_INTERVIEW
        questions = {}
        for q_id in q_ids:
            q_text = qa_pair[q_id].iloc[0]  # Definition row
            questions[q_id] = q_text
        log.info(f"Parsed {len(questions)} questions")

        # Human data (skip definition row)
        human_data = human_chara.copy()
        total_persons = len(human_data)
        log.info(f"Processing {total_persons} individuals")

        # Initialize LLM
        global llm_client
        llm_client = LLMClient(llm_provider, llm_model, api_key, base_url)
        log.info("LLM client initialized")

        # Initialize theory
        if theory_name.lower() == "mbti":
            theory = MBTITheory()
        elif theory_name .lower() == "bigfive":
            theory = BigFiveTheory()
        else:
            log.error(f"Theory not implemented: {theory_name}")
            raise NotImplementedError()
        combo = theory.build_process_combination()
        simulated = []
        person_counter = 0

        for idx, row in human_data.iterrows():
            person_counter += 1
            interview_id = row['D_INTERVIEW']
            cluster = row['cluster']
            demographics = row.to_dict()

            log.info(f"[{person_counter}/{total_persons}] Processing ID={interview_id}, Cluster={cluster}")

            # Predict rule-based profile
            profile = theory.predict_from_demographics(demographics)
            combo = ProcessCombination.build_from_config(
                config_path=config_path,
                global_include_metadata=include_metadata,
                stage_overrides=stage_overrides or {},
                cache_dir=debug_cache_dir
            )
            constraints = {}

            q_counter = 0
            for q_id, q_text in questions.items():
                q_counter += 1
                if max_questions_per_person and q_counter > max_questions_per_person:
                    log.debug(f"  Skipping remaining questions for ID={interview_id} (limit reached)")
                    break

                # Parse question and options

                if '\n' in q_text:
                    parts = q_text.split('\n')
                    question = parts[0].strip()
                    options = ', '.join(parts[1:]).strip()
                else:
                    question = q_text.strip()
                    options = ''

                log.debug(f"  → Q{q_id}: {question[:60]}{'...' if len(question)>60 else ''}")

                try:
                    # Debugging: Add debug_cache_dir to extra for access in combine/execute
                    extra_with_debug = {'demographics': demographics,  'llm_client': llm_client, 'cache_dir': debug_cache_dir,
                                        'debug_mode': debug_mode, 'interview_id': interview_id, 'q_id': q_id}
                    # End debugging

                    answer_json = combo.combine(question, options, profile, constraints,
                                                include_metadata=include_metadata, **extra_with_debug)

                    try:
                        answer_data = json.loads(answer_json)
                        final_answer = answer_data.get('conclusion', answer_json)
                    except json.JSONDecodeError:
                        final_answer = answer_json.strip()

                    simulated.append({
                        'cluster': cluster,
                        'interview_id': interview_id,
                        'question_id': q_id,
                        'simulated_answer': final_answer
                    })
                    log.debug(f"    Simulated: {final_answer}")

                except Exception as e:
                    log.error(f"    Failed Q{q_id} for ID={interview_id}: {e}")
                    simulated.append({
                        'cluster': cluster,
                        'interview_id': interview_id,
                        'question_id': q_id,
                        'simulated_answer': f"ERROR: {str(e)[:100]}"
                    })

        # Save results
        result_df = pd.DataFrame(simulated)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        exp_name = f"{theory_name}-{exp_name}"
        result_df.to_csv(os.path.join(output_path, exp_name), index=False, mode='w+')
        total_responses = len(result_df)

        elapsed = datetime.now() - start_time
        log.info(f"Simulation completed in {elapsed}")
        log.info(f"Generated {total_responses} responses → {output_path}")
        log.info("=== MBTI Simulation Finished ===")

    except Exception as e:
        log.error(f"Simulation failed: {e}")
        raise


# Example usage
if __name__ == "__main__":
    arguments = args()

    run_simulation(
        data_path=arguments.data_path,
        exp_name=arguments.exp_name,
        theory_name=arguments.theory_name,
        config_path=arguments.config_path,
        llm_provider=arguments.llm_provider,
        llm_model=arguments.llm_model,
        base_url=arguments.base_url,
        api_key=arguments.api_key,
        include_metadata=False,
        output_path=arguments.output_path,
        max_questions_per_person=arguments.max_questions_per_person,
        debug_mode=arguments.debug_mode,
        debug_cache_dir=arguments.debug_cache_dir
    )