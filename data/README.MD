# Global Representative Survey Simulation Dataset Creation Pipeline  

This repository contains the exact data-preprocessing and sampling pipeline used to create the final evaluation sets (both WVS-based demographic representatives and the corresponding Global Opinion QA simulation sheets).

## Overview of the Two Parallel Tracks

| Track | Purpose | Input | Main Scripts | Final Output |
|-------|--------|-------|--------------|--------------|
| **A. WVS Demographic Representatives** | Create 1–N most centrally-representative respondents per country using PCA on 38 core human-characteristic items | Raw WVS Wave 7 Excel (with headers in row 1 & 2) | 1. `archetypal_sampling.py` → 2. `notasked_clean.py` → 3. `organize_sheets.py` | `wvs_representatives_final/` with plots + cleaned per-country Excel files |
| **B. Global Opinion QA Simulation Sheets** | Pair the above representatives with 200 opinion questions per country (taken from the NAACL 2025 paper’s released QA set) | Sampled WVS representatives + `global_opinion_qa.json` | `generate_opinion_qa_per_country.py` | `opinion_sampled_200/` with per-country Excel sheets ready for LLM simulation |

Both tracks end up with identically-structured Excel files (3 sheets: `Total`, `Human_chara`, `QA_pair`) so that LLM simulation experiments are directly comparable.

## Detailed Step-by-Step Pipeline

### Track A – Creating Demographic Representatives from WVS

```bash
# 0. Put your full WVS Wave 7 Excel file here (the one with row-0 = variable codes, row-1 = descriptions)
#    └── raw_wvs_full.xlsx

# 1. Sample the most representative respondent(s) per country using PCA on 38 core items
python archetypal_sampling.py \
    --file_path raw_wvs_full.xlsx \
    --n 5 \                     # change to 1, 3, 5, 10 … depending on how many archetypes you want
    --output_dir wvs_representatives_final

# → Creates:
#     wvs_representatives_final/
#       ├── all_representatives_n5.xlsx
#       ├── per_country/*.xlsx
#       ├── 01_global_demographic_pca.png
#       └── 02_inglehart_welzel_cultural_map.png   ← the two figures shown in the paper
```

```bash
# 2. Remove columns that were “Not asked” in ≥95% of respondents for each country individually
python notasked_clean.py \
    --input_file raw_wvs_full.xlsx \
    --output_dir cleaned_per_country

# → cleaned_per_country/   (one file per country with only asked variables)
```

```bash
# 3. Re-organize every country file into the final 3-sheet format and generate metadata summary
python organize_sheets.py
# (it uses cleaned_per_country/ as input
# → processed_countries/          (final clean demographic files)
# → countries_metadata_summary.xlsx
```

At this point `processed_countries/` (or directly the files inside `wvs_representatives_final/per_country/` if you only need the representatives) contain the gold demographic archetypes used in all experiments.

### Track B – Attaching Global Opinion QA (200 questions per country)

We use the publicly released question set from  
**“Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations” (NAACL 2025)**  
available as `global_opinion_qa.json`.

```bash
# Make sure you have:
#   - the representatives from Track A  (default looks inside wvs_representatives_final/per_country)
#   - global_opinion_qa.json in the repo root

python generate_opinion_qa_per_country.py \
    --qa_json global_opinion_qa.json \
    --representatives_dir wvs_representatives_final/per_country \
    --output_dir opinion_sampled_200 \
    --max_questions 200 \
    --seed 42
```

Output structure:

```
opinion_sampled_200/
├── sampled_200_global_opinion_qa.json          # exact 200 × |countries| questions used
└── excel_per_country/
    ├── Argentina_opinion_200.xlsx
    ├── Japan_opinion_200.xlsx
    ├── Nigeria_opinion_200.xlsx
    └── ... (one file per country with 3 sheets)
```

Each Excel file contains:

- `Total`       – full table (demographics + empty GOQ columns ready to be filled by the LLM)
- `Human_chara` – only the 38 PCA items + identifiers (identical to the WVS track)
- `QA_pair`     – only the 200 Global Opinion Questions (columns GOQ0 … GOQ199)

This is the exact format used for all simulation runs and distribution divergence calculations in the paper.

## Folder Overview (final state)

```
.
├── raw_wvs_full.xlsx                              # ← your original WVS file (not shared)
├── global_opinion_qa.json                         # ← from NAACL 2025 paper
├── wvs_representatives_final/                     # ← output of step 1 (figures + representatives)
├── cleaned_per_country/                           # ← output of step 2
├── processed_countries/                           # ← output of step 3 (optional clean full files)
├── countries_metadata_summary.xlsx                # ← overview
├── opinion_sampled_200/
│   ├── sampled_200_global_opinion_qa.json
│   └── excel_per_country/                         # ← final simulation sheets
└── *.py                                           # pipeline scripts
```

## Notes & Assumptions (important for reproducibility)

1. All scripts assume the original WVS Excel has:
   - Row 1 → variable codes (e.g., Q260, B_COUNTRY, …)
   - Row 2 → human-readable descriptions
   - Row 3+ → actual respondent data

2. The 38 human-characteristic items used for PCA are hard-coded in every script (see `WVS_ITEMS` list).

3. We intentionally do **not** have access to the original Global Opinion QA survey responses, so we evaluate only via demographic representativeness + simulated answer distributions (as done in the NAACL 2025 paper).

4. Changing `--n` in step 1 lets you experiment with 1 / 3 / 5 / 10 archetypes per country; all later scripts automatically pick up whatever is present.

5. Random seed is fixed to 42 for the 200-question sampling → fully reproducible.