# run_baselines_batch_vllm.py
import argparse
import pandas as pd
import random
from pathlib import Path
from vllm import LLM, SamplingParams

from run_baseline_zs import OpinionRetriever

# Unrelated theories for baseline 6
UNRELATED_THEORIES = [
    "quantum physics", "evolutionary biology", "classical mechanics", "organic chemistry",
    "thermodynamics", "genetics", "electromagnetism", "cell biology",
    "relativity theory", "ecology", "atomic structure", "neurobiology"
]

THEORY_CONCEPTS = {
    "quantum physics": ["a particle in superposition", "an entangled system", "wave function collapse"],
    "evolutionary biology": ["natural selection", "genetic mutation", "adaptive trait"],
    "classical mechanics": ["Newtonian motion", "inertia", "momentum conservation"],
    "organic chemistry": ["carbon bonding", "functional group reaction", "stereochemistry"],
    "thermodynamics": ["entropy increase", "heat transfer", "energy conservation"],
    "genetics": ["DNA replication", "gene expression", "Mendelian inheritance"],
    "electromagnetism": ["electric field", "magnetic induction", "electromagnetic wave"],
    "cell biology": ["mitosis division", "organelle function", "membrane transport"],
    "relativity theory": ["space-time curvature", "time dilation", "mass-energy equivalence"],
    "ecology": ["food web dynamics", "biodiversity", "ecosystem balance"],
    "atomic structure": ["electron orbital", "nuclear stability", "quantum energy level"],
    "neurobiology": ["neural firing", "synaptic transmission", "action potential"]
}

def main(args):
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    country = data_path.stem.split('_')[0].replace('_', ' ')
    print(f"Processing {country} from {data_path.name}")

    if args.baseline == "hwang_2023":
        retriever = OpinionRetriever()
        retriever.build_index(data_path)

    # Load Total sheet for questions
    qa_desc_sheet = pd.read_excel(data_path, sheet_name="QA_pairs", header=0)

    # Load Human_chara for demographics
    human_df = pd.read_excel(data_path, sheet_name="Human_chara", header=0)

    # Align subjects
    subjects = human_df[['D_INTERVIEW']].copy()[1:]
    subjects = subjects.merge(human_df, on='D_INTERVIEW', how='left')
    subjects['D_INTERVIEW'] = subjects['D_INTERVIEW'].astype(str)

    # QA columns
    qa_cols = [col for col in qa_desc_sheet.columns if col.startswith("GOQ") or col.startswith("Q")]
    questions = []

    for col in qa_cols:
        desc = qa_desc_sheet[col].iloc[0]
        questions.append({"code": col, "desc": str(desc)})

    print(f"  {len(subjects)} subjects × {len(questions)} questions")

    # Initialize vLLM with configurable parameters
    print("Initializing vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        quantization="awq"
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=args.stop or None,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )
    # Collect all prompts in batch
    prompts = []
    results_info = []  # To map back

    for _, subject in subjects.iterrows():
        iid = subject['D_INTERVIEW']

        for q in questions:
            q_code = q["code"]
            q_desc = q["desc"]

            # Build prompt per baseline
            if args.baseline == "durmus_2023":
                prompt = f"How would someone from {country} answer the following question:\n{q_desc}\nAssistant: If had to select one of the options, my answer would be ("

            elif args.baseline == "liu_2025":
                prompt = f"I want you to act as a citizen from {country}.\nYou should obtain an answer from the choices.\nYou should tell me the answer in the format Answer:\"\nRequest: {q_desc}"

            elif args.baseline == "hwang_2023":
                prior_opinions = retriever.retrieve_top_k(q_desc, k=3)
                prompt = f"A person can be described as follows: Age: unknown Income: unknown Political party: unknown Religion: unknown\nThe person has the following opinions. Opinions: {prior_opinions}\nBased on the above list of opinions and the demographic information, which answer choice will this person select for the question:\nQuestion: {q_desc}\nAnswer:"

            elif args.baseline == "zhao_2024":
                prompt = f"Person X provided the following demographic information in an interview: Region country specific, Urban-Rural Settlement type, Highest educational level: Respondent.\nWhat would Person X answer to the following question? Question: {q_desc}"

            elif args.baseline == "only_theory":
                theory = "MBTI" if "MBTI" in args.theory_name.lower() else "Big Five"
                prompt = f"Based on {theory} personality theory and typical patterns in {country}, how would a representative person answer:\n{q_desc}\nAnswer with only the letter."

            elif args.baseline == "unrelated_theory":
                theory = random.choice(UNRELATED_THEORIES)
                concept = random.choice(THEORY_CONCEPTS[theory])
                prompt = f"Based on {theory} ({concept}), metaphorically applied to human behavior in {country}, how would such a system respond to:\n{q_desc}\nAnswer with only the single letter."

            prompts.append(prompt)
            results_info.append((country, iid, q_code))

    # Batch inference with vLLM
    print(f"Running batch inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    answers = [out.outputs[0].text.strip() for out in outputs]

    # Map back
    results = []
    for (country, iid, q_code), answer in zip(results_info, answers):
        results.append({
            "country": country,
            "interview_id": iid,
            "question_id": q_code,
            "simulated_answer": answer
        })

    # Save
    df_out = pd.DataFrame(results)
    out_file = output_dir / f"{args.baseline}_{data_path.stem}.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved {len(df_out)} responses → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot baselines with vLLM batch inference")
    parser.add_argument('--data_path', type=str, required=True, help="Path to one country opinion Excel")
    parser.add_argument('--baseline', type=str, required=True,
                        choices=["durmus_2023", "liu_2025", "hwang_2023", "zhao_2024", "only_theory",
                                 "unrelated_theory"])
    parser.add_argument('--theory_name', type=str, default="MBTI")

    # vLLM model config
    parser.add_argument('--model', type=str, required=True, help="Model path or name")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--dtype', type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--download_dir', type=str, default=None)
    parser.add_argument('--enforce_eager', action='store_true')
    parser.add_argument('--max_num_batched_tokens', type=int, default=None)
    parser.add_argument('--max_num_seqs', type=int, default=32768)

    # Sampling
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--stop', type=str, default=None, help="Stop tokens (comma-separated)")
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)

    parser.add_argument('--output_dir', type=str, default="baseline_outputs_vllm")

    args = parser.parse_args()
    if args.stop:
        args.stop = args.stop.split(",")
    main(args)