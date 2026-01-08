#!/usr/bin/env python

import os
import json
import random
import math
import argparse
import re
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Regex for <think>...</think> (handles both raw and HTML-escaped)
THINK_PATTERN = re.compile(r'(&lt;|<)think(&gt;|>)(.*?)((&lt;|<)/think(&gt;|>))', re.DOTALL | re.IGNORECASE)

def extract_visible_reasoning(text: str) -> str:
    """Remove all <think>...</think> blocks and normalize whitespace."""
    cleaned = THINK_PATTERN.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_final_answer(text: str) -> str:
    """
    Extract the model's final answer from the full response.
    Handles common patterns like: "A", "**A**", "The answer is A", "Final answer: B", etc.
    """
    text_upper = text.upper()

    patterns = [
        r'THE ANSWER IS\s*([A-Z])',
        r'FINAL ANSWER[:\s]*([A-Z])',
        r'\*\*([A-Z])\*\*',
        r'\[([A-Z])\]',
        r'\(([A-Z])\)',
        r'\b([A-Z])\b(?=\s*$)',  # Single letter near end
        r'OPTION\s*([A-Z])',
        r'CHOICE[:\s]*([A-Z])',
    ]

    for pat in patterns:
        match = re.search(pat, text_upper)
        if match:
            return match.group(1)

    # Fallback: any uppercase letter in last 100 chars
    last_part = text_upper[-100:]
    singles = re.findall(r'\b([A-Z])\b', last_part)
    if singles:
        return singles[-1]

    return "A"  # Safe default if nothing found

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning quality on SFT model outputs (visible reasoning only). "
                    "Uses single-answer Consistency & Informativeness toward model's own prediction."
    )
    parser.add_argument("directory", type=str, help="Directory containing JSON/JSONL model outputs.")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of samples to evaluate (default: 200).")
    parser.add_argument(
        "--model",
        type=str,
        default="neuralmagic/Phi-3-medium-128k-instruct-quantized.w4a16",
        help="Evaluator model (default: 4-bit quantized Phi-3-medium, ~10-14GB VRAM)."
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for vLLM inference (default: 32).")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max context length (default: 128K).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.95).")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file (default: print to stdout).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()

def compute_logprobs_batch(llm: LLM, sequences: List[str]) -> List[float]:
    """Compute total logprob of each full sequence (prompt + continuation)."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=1,
        prompt_logprobs=True
    )
    outputs = llm.generate(sequences, sampling_params, use_tqdm=False)

    batch_logprobs = []
    for output in outputs:
        if output.prompt_logprobs:
            total = sum(list(d.values())[0].logprob for d in output.prompt_logprobs if d)
            batch_logprobs.append(total)
        else:
            batch_logprobs.append(0.0)
    return batch_logprobs

def compute_consistency_scores_batch(llm: LLM, samples: List[Dict], batch_size: int) -> List[float]:
    """P(evaluator predicts model's own answer | full visible reasoning)"""
    sequences = []
    for sample in samples:
        prompt = f"Given this reasoning: {sample['r']}\nQuestion: {sample['x']}\nFinal answer:"
        continuation = f" {sample['a']}"
        sequences.append(prompt + continuation)

    all_logprobs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Consistency"):
        all_logprobs.extend(compute_logprobs_batch(llm, sequences[i:i + batch_size]))

    # Convert logprob → probability
    scores = [math.exp(lp) for lp in all_logprobs]
    return scores

def compute_informativeness_batch(llm: LLM, samples: List[Dict], batch_size: int) -> List[List[float]]:
    """Step-by-step information gain toward model's own final answer"""
    sequences = []
    step_counts = []

    for sample in samples:
        steps = [s.strip() for s in sample['r'].split(".") if s.strip()]
        cumulative = ""
        count = 0
        base_prompt = f"Question: {sample['x']}\nFinal answer:"
        for step in steps:
            cumulative += step + ". "
            prompt = f"Given this reasoning: {cumulative.strip()}\n{base_prompt}"
            sequences.append(prompt + f" {sample['a']}")
            count += 1
        step_counts.append(count)

    all_logprobs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Informativeness"):
        all_logprobs.extend(compute_logprobs_batch(llm, sequences[i:i + batch_size]))

    gains_per_sample = []
    idx = 0
    for count in step_counts:
        gains = []
        prev_p = 0.0
        for _ in range(count):
            p = math.exp(all_logprobs[idx])
            gain = p - prev_p
            gains.append(max(gain, 0.0))  # Clamp negative gains (rare due to noise)
            prev_p = p
            idx += 1
        gains_per_sample.append(gains)
    return gains_per_sample

def compute_intra_step_correctness_batch(llm: LLM, steps: List[str], batch_size: int) -> List[int]:
    prompts = [
        "Score this reasoning step on rationality from 1 to 5 (1=poor, 5=excellent).\n"
        "Focus on: logical soundness, minimal self-doubt/repetition/hedging.\n"
        "Output ONLY the integer.\n\nStep: " + step
        for step in steps if step.strip()
    ]

    if not prompts:
        return []

    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Intra-step scoring"):
        outputs.extend(llm.generate(prompts[i:i + batch_size], sampling_params, use_tqdm=False))

    scores = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            score = int(text)
            scores.append(max(1, min(5, score)))
        except ValueError:
            scores.append(1)
    return scores

def load_outputs(directory: str) -> List[Dict]:
    outputs = []
    print(f"Loading outputs from: {directory}")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.jsonl'):
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        outputs.append(json.loads(line))
        elif filename.endswith('.json'):
            with open(filepath) as f:
                data = json.load(f)
                outputs.extend(data if isinstance(data, list) else [data])
    print(f"Loaded {len(outputs)} samples.")
    return outputs

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading evaluator model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=4,
        enforce_eager=True,  # Helps stability on some setups
    )

    outputs = load_outputs(args.directory)
    if not outputs:
        raise ValueError("No data found in directory.")

    raw_samples = random.sample(outputs, min(args.sample_size, len(outputs)))
    samples = []
    all_steps = []
    step_indices = []

    print(f"Processing {len(raw_samples)} samples...")
    valid_count = 0
    for raw in raw_samples:
        question = raw.get('question', '') or raw.get('prompt', '') or raw.get('input', '')
        full_response = (
            raw.get('response', '') or
            raw.get('generated', '') or
            raw.get('output', '') or
            raw.get('predict', '') or
            ''
        )

        if not full_response or not question:
            continue

        visible_reasoning = extract_visible_reasoning(full_response)
        final_answer = extract_final_answer(full_response)

        if not visible_reasoning or not final_answer:
            continue

        steps = [s.strip() for s in visible_reasoning.split(".") if s.strip()]
        start_idx = len(all_steps)
        all_steps.extend(steps)
        step_indices.append((start_idx, len(steps)))

        samples.append({
            'x': question,
            'r': visible_reasoning,
            'a': final_answer
        })
        valid_count += 1

    print(f"Evaluating {valid_count} valid samples...")

    if valid_count == 0:
        raise ValueError("No valid samples after filtering.")

    # Compute metrics
    consistency_scores = compute_consistency_scores_batch(llm, samples, args.batch_size)

    informativeness_gains_list = compute_informativeness_batch(llm, samples, args.batch_size)
    avg_informativeness = [sum(gains) / len(gains) if gains else 0.0 for gains in informativeness_gains_list]

    intra_scores = compute_intra_step_correctness_batch(llm, all_steps, args.batch_size)
    intra_avgs = []
    for start, length in step_indices:
        segment = intra_scores[start:start + length]
        intra_avgs.append(sum(segment) / len(segment) if segment else 0.0)

    # Final results
    results = {
        "evaluator_model": args.model,
        "average_consistency": sum(consistency_scores) / len(consistency_scores) if consistency_scores > 0.0001 else 0.0,
        "average_informativeness_gain": sum(avg_informativeness) / len(avg_informativeness) if avg_informativeness > 0.0001 else 0.0,
        "average_intra_step_correctness": sum(intra_avgs) / len(intra_avgs),
        "num_samples_evaluated": len(samples),
        "total_loaded": len(outputs),
        "note": "Metrics computed on visible reasoning only, using model's own final answer as target."
    }

    json_out = json.dumps(results, indent=2)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(args.output + "/results_phi3_medium.json", 'w') as f:
            f.write(json_out)
        print(f"\nResults saved to: {args.output}")
        print(f"Brief output:\n{json_out}")
    else:
        print("\n" + json_out)