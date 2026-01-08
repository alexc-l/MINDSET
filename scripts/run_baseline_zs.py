# run_baselines_zero_shot_demographics.py
import argparse
import os
from typing import List, Tuple

import pandas as pd
import random
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

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

# Embedding model (you can change to any SentenceTransformer model)
EMBEDDING_MODEL = "cache/all-MiniLM-L6-v2"  # Fast and good for English/multilingual


class OpinionRetriever:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.question_texts = []           # list of unique question descriptions
        self.question_to_answers = {}      # q_desc → list of answers (str)

    def build_index(self, wvs_excel_path: Path):
        print(f"Building opinion vector store from {wvs_excel_path.name}...")
        qa_df = pd.read_excel(wvs_excel_path, sheet_name="QA_pairs", header=0)

        question_codes = qa_df.columns[2:]  # skip D_INTERVIEW, B_COUNTRY

        question_texts = []
        question_to_answers = {}

        for q_code in question_codes:
            q_desc = str(qa_df[q_code].iloc[0]).strip()  # description row
            if not q_desc or q_desc.lower() in ["nan", ""]:
                continue

            question_texts.append(q_desc)

            # All individual answers
            answers = qa_df[q_code].iloc[2:].dropna().astype(str).tolist()
            answers = [a.strip() for a in answers if a.strip() and a.lower() not in ["nan", ""]]
            if answers:
                question_to_answers[q_desc] = answers

        if not question_texts:
            print("No valid questions found!")
            return

        # Embed questions
        print(f"Embedding {len(question_texts)} unique questions...")
        embeddings = self.model.encode(question_texts, normalize_embeddings=True, batch_size=32)

        # FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        self.question_texts = question_texts
        self.question_to_answers = question_to_answers

        total_responses = sum(len(v) for v in question_to_answers.values())
        print(f"Vector store built: {len(question_texts)} questions, {total_responses} responses")

    def retrieve_top_k(self, query: str, k: int = 3) -> str:
        """
        Retrieve top-k similar questions (excluding exact match to query).
        Return formatted QA pairs.
        """
        if self.index is None or self.index.ntotal == 0:
            return "No prior opinions available."

        query = query.strip()
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # Search top-(k+1) to allow skipping exact match
        scores, indices = self.index.search(query_emb.astype(np.float32), k + 5)

        retrieved_pairs = []
        used_questions = set()

        for idx in indices[0]:
            if idx == -1:
                continue
            q_text = self.question_texts[idx]

            # Skip if question is identical to current query (anti-leakage)
            if q_text.strip().lower() == query.lower():
                continue

            if q_text in used_questions:
                continue
            used_questions.add(q_text)

            answers = self.question_to_answers.get(q_text, [])
            if not answers:
                continue

            # Sample up to 2 answers per question (avoid flooding)
            sampled = answers[:2]

            for ans in sampled:
                retrieved_pairs.append(f"Q: {q_text}\nA: {ans}")

            if len(retrieved_pairs) >= k * 2:
                break

        if not retrieved_pairs:
            return "No similar prior opinions found (excluding current question)."

        return "Prior similar opinions from respondents in this country:\n\n" + "\n\n".join(retrieved_pairs[:k])

def main(args):
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    country = args.country
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

    # Initialize OpenAI client
    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key)

    results = []
    for _, subject in tqdm(subjects.iterrows(), total=len(subjects), desc="Subjects", disable=False):
        iid = subject['D_INTERVIEW']

        # NEW: Inner progress bar for questions
        question_tqdm = tqdm(questions, total=len(questions), desc=f"Questions for ID={iid}", disable=False,
                             leave=False)
        # Extract demographics for Hwang & Zhao
        age = subject.get('Q262', 'unknown')
        income = subject.get('Q288R', 'unknown')
        religion = subject.get('Q289', 'unknown')
        region = subject.get('N_REGION_WVS', 'unknown')
        urban_rural = subject.get('H_URBRURAL', 'unknown')
        education = subject.get('Highest educational level: Respondent', 'unknown')

        for q in question_tqdm:
            q_code = q["code"]
            q_desc = q["desc"]

            # Build prompt per baseline
            if args.baseline == "durmus_2023":
                prompt = f"How would someone from {country} answer the following question:\n{q_desc}\nAssistant: If had to select one of the options, my answer would be ("

            elif args.baseline == "liu_2025":
                prompt = f"I want you to act as a citizen from {country}.\nYou should obtain an answer from the choices.\nYou should tell me the answer in the format Answer:\"\nRequest: {q_desc}"

            elif args.baseline == "hwang_2023":
                prior_opinions = retriever.retrieve_top_k(q_desc, k=3)
                prompt = f"A person can be described as follows: Age: {age} Income: {income} Religion: {religion}\nThe person has the following opinions. Opinions: {prior_opinions}\nBased on the above list of opinions and the demographic information, which answer choice will this person select for the question:\nQuestion: {q_desc}\nAnswer:"

            elif args.baseline == "zhao_2024":
                prompt = f"Person X provided the following demographic information in an interview: Region: {region}, Urban-Rural: {urban_rural}, Education: {education}.\nWhat would Person X answer to the following question?\nQuestion: {q_desc}"

            elif args.baseline == "only_theory":
                theory = "MBTI" if "MBTI" in args.theory_name.lower() else "Big Five"
                prompt = f"Based on {theory} personality theory and typical patterns in {country}, how would a representative person answer:\n{q_desc}\nAnswer with only the letter."

            elif args.baseline == "unrelated_theory":
                theory = random.choice(UNRELATED_THEORIES)
                concept = random.choice(THEORY_CONCEPTS[theory])
                prompt = f"Based on {theory} ({concept}), metaphorically applied to human behavior in {country}, how would such a system respond to:\n{q_desc}\nAnswer with only the single letter."

            # Call API
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"API error {iid}/{q_code}: {e}")
                answer = "ERROR"

            results.append({
                "country": country,
                "interview_id": iid,
                "question_id": q_code,
                "simulated_answer": answer
            })

            # NEW: Update question bar description
            question_tqdm.set_description(f"Q {q_code} for ID={iid}")

    # Save
    df_out = pd.DataFrame(results)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = output_dir / f"{args.baseline}_{data_path.stem}.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved {len(df_out)} responses → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--country', type=str, required=True)
    parser.add_argument('--baseline', type=str, required=True,
                        choices=["durmus_2023", "liu_2025", "hwang_2023", "zhao_2024", "only_theory", "unrelated_theory"])
    parser.add_argument('--theory_name', type=str, default="MBTI")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--api_key', type=str, default="EMPTY")
    parser.add_argument('--output_dir', type=str, default="baseline_outputs")
    args = parser.parse_args()
    main(args)