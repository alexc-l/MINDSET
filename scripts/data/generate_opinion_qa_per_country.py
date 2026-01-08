# generate_opinion_qa_200_fixed.py
import pandas as pd
import json
import re
import random
import argparse
from pathlib import Path

# Your 38 original human feature columns
WVS_ITEMS = [
    'N_REGION_WVS', 'N_TOWN', 'G_TOWNSIZE2', 'H_SETTLEMENT', 'H_URBRURAL', 'E_RESPINT',
    'Q260', 'Q261', 'Q262', 'Q263', 'Q264', 'Q265', 'Q266', 'Q267', 'Q268', 'Q269',
    'Q270', 'Q271', 'Q272', 'Q273', 'Q274', 'Q275', 'Q276', 'Q277', 'Q278', 'Q279',
    'Q280', 'Q281', 'Q282', 'Q283', 'Q284', 'Q285', 'Q286', 'Q287',
    'Q288R', 'Q289', 'Q290'
]

def extract_country(instruction):
    m = re.search(r"someone from ([^\s]+(?:\s+[^\s]+)*?) answer", instruction, re.IGNORECASE)
    return m.group(1).strip() if m else None

def main(args):
    random.seed(args.seed)

    # Load opinion QA
    print(f"Loading {args.qa_json} ...")
    with open(args.qa_json, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)

    # Load WVS representatives → get country list
    rep_path = Path(args.representatives_dir)
    reps = {}
    wvs_countries = set()
    for f in rep_path.glob("*_cleaned.xlsx"):
        country = f.stem.replace("_cleaned", "").replace("_", " ")
        df = pd.read_excel(f)
        if not df.empty:
            reps[country] = df
            wvs_countries.add(country)

    # Group and sample 200 questions per country (only countries with WVS reps)
    sampled_all = []
    country_questions = {}

    for q in qa_list:
        country = extract_country(q["instruction"])
        if country and country in wvs_countries:
            country_questions.setdefault(country, []).append(q)

    print(f"\nSampling max {args.max_questions} questions per country...")
    for country, qs in country_questions.items():
        n = min(args.max_questions, len(qs))
        chosen = random.sample(qs, n)
        country_questions[country] = chosen
        sampled_all.extend(chosen)
        print(f"  {country}: {len(qs)} → {n}")

    # Save sampled JSON (this is now the source of truth)
    sampled_json = Path(args.output_dir) / "sampled_200_global_opinion_qa_c1.json"
    sampled_json.parent.mkdir(parents=True, exist_ok=True)
    with open(sampled_json, 'w', encoding='utf-8') as f:
        json.dump(sampled_all, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sampled questions → {sampled_json}")

    # Generate Excel files
    excel_dir = Path(args.output_dir) / "excel_per_country"
    excel_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating Excel files...")
    for country, questions in country_questions.items():
        base_df = reps[country].copy()
        n_people = len(base_df)

        # Global Opinion QA columns: GOQ0, GOQ1, ...
        goq_cols = [f"GOQ{q['id']}" for q in questions]

        # Full column order
        all_columns_display = ['Interview ID', 'Country'] + WVS_ITEMS + goq_cols
        all_columns_codes    = ['D_INTERVIEW', 'B_COUNTRY'] + WVS_ITEMS + goq_cols

        # Header row 2 – descriptions
        descriptions = ['Interview ID', 'Country'] + [''] * len(WVS_ITEMS)
        for q in questions:
            opts = ", ".join(q["options"])
            text = f"{q['input'].strip()} {opts}".strip().replace("\n", " ")
            if len(text) > 32000:
                text = text[:32000] + "..."
            descriptions.append(text)

        # Build data rows
        data = []
        for i in range(n_people):
            r = base_df.iloc[i]
            row = [
                r.get('D_INTERVIEW', f"SIM{i+1:04d}"),
                country
            ]
            row += [r.get(c, '') for c in WVS_ITEMS]
            row += [''] * len(goq_cols)                     # empty answers
            data.append(row)

        df_total = pd.DataFrame(data, columns=all_columns_display)

        # Save
        safe_name = country.replace(' ', '_').replace('/', '_')
        out_file = excel_dir / f"{safe_name}_opinion_200.xlsx"

        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            # === Sheet: Total ===
            pd.DataFrame([all_columns_codes]).to_excel(writer, sheet_name='Total', index=False, header=False, startrow=0)
            pd.DataFrame([descriptions]).to_excel(writer, sheet_name='Total', index=False, header=False, startrow=1)
            df_total.to_excel(writer, sheet_name='Total', index=False, header=False, startrow=2)

            # === Sheet: Human_chara ===
            human_disp = ['Interview ID', 'Country'] + WVS_ITEMS
            human_code = ['D_INTERVIEW', 'B_COUNTRY'] + WVS_ITEMS
            human_desc = ['Interview ID', 'Country'] + [''] * len(WVS_ITEMS)

            pd.DataFrame([human_code]).to_excel(writer, sheet_name='Human_chara', index=False, header=False, startrow=0)
            pd.DataFrame([human_desc]).to_excel(writer, sheet_name='Human_chara', index=False, header=False, startrow=1)
            df_total[human_disp].to_excel(writer, sheet_name='Human_chara', index=False, header=False, startrow=2)

            # === Sheet: QA_pair ===
            qa_disp = ['Interview ID', 'Country'] + goq_cols
            qa_code = ['D_INTERVIEW', 'B_COUNTRY'] + goq_cols
            qa_desc = ['Interview ID', 'Country'] + descriptions[-len(goq_cols):]

            pd.DataFrame([qa_code]).to_excel(writer, sheet_name='QA_pair', index=False, header=False, startrow=0)
            pd.DataFrame([qa_desc]).to_excel(writer, sheet_name='QA_pair', index=False, header=False, startrow=1)
            df_total[qa_disp].to_excel(writer, sheet_name='QA_pair', index=False, header=False, startrow=2)

        print(f"  {country} → {out_file.name} ({n_people} people × {len(questions)} GOQs)")

    print(f"\nAll done!")
    print(f"   Sampled JSON : {sampled_json}")
    print(f"   Excel files  : {excel_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_json', type=str, default="global_opinion_qa.json")
    parser.add_argument('--representatives_dir', type=str, default="final_representatives/per_country")
    parser.add_argument('--output_dir', type=str, default="opinion_200_final")
    parser.add_argument('--max_questions', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)