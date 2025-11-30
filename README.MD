## Data Generation Pipeline  

### Overview
We construct a **global, culturally grounded public opinion simulation dataset** by combining:

1. **World Values Survey (WVS) Wave 7** – real human respondents  
2. **GlobalOpinionQA** – a large-scale, distribution-matched set of factual + opinion questions about international affairs, politics, values, and current events (approximately 120,000 questions across 80+ countries)

The final deliverable consists of:
- One Excel file per country (`XXX_opinion_200.xlsx`)
- Three sheets per file: `Total`, `Human_chara`, `QA_pair`
- Exactly **200 randomly sampled GlobalOpinionQA questions** per country (seed 42)
- Real WVS respondents as simulation agents (most centrally located in demographic PCA space)

### Step-by-step Reproducibility

# WVS Wave 7 – Final Processed Dataset (Country-Level Split + Metadata)

## Overview
This folder contains the final, ready-to-model version of the **World Values Survey (WVS) Wave 7** data after extensive cleaning, harmonization, and splitting into meaningful components.

- One Excel file per country → `processed_countries/`
- One metadata summary → `countries_metadata_summary.xlsx`
- All technical/PCA columns removed from modeling sheets
- Clear separation between **human socio-demographic characteristics** and **actual survey Q/A items**

## Folder Structure
```
.
├── cleaned_countries/                  # INPUT: one cleaned .xlsx per country (your previous step)
├── processed_countries/                # OUTPUT: final files used for modeling
│   ├── Argentina.xlsx
│   ├── Japan.xlsx
│   └── ...
├── countries_metadata_summary.xlsx     # Summary of row/col counts per country
├── create_final_dataset.py             # Main script that generates everything
└── README.md                           # This file
```

## What Each Sheet Contains

| Sheet Name     | Content                                                                                   | Index Columns                  | Used for Modeling? |
|----------------|-------------------------------------------------------------------------------------------|--------------------------------|---------------------|
| `Total`        | Full cleaned country dataset (unchanged from input)                                       | None (regular columns)         | No (reference only) |
| `Human_chara`  | Only true socio-demographic & interviewer variables (age, sex, education, region, etc.)  | `B_COUNTRY`, `D_INTERVIEW`     | Yes – as control features |
| `QA_pairs`     | All actual survey questions and answers (the real dependent/independent variables)       | `B_COUNTRY`, `D_INTERVIEW`     | Yes – main modeling data |

### Columns EXCLUDED from both `Human_chara` and `QA_pairs`
These are technical or post-processing variables and must **never** be used as features:

- `PC1`, `PC2`, `PC3`, `PC4`
- `selection_rank`
- `distance_to_centroid`

### Human Characteristics Included in `Human_chara` (WVS_ITEMS)
```python
[
    'N_REGION_WVS', 'N_TOWN', 'G_TOWNSIZE2', 'H_SETTLEMENT', 'H_URBRURAL', 'E_RESPINT',
    'Q260', 'Q261', 'Q262', 'Q263', 'Q264', 'Q265', 'Q266', 'Q267', 'Q268', 'Q269',
    'Q270', 'Q271', 'Q272', 'Q273', 'Q274', 'Q275', 'Q276', 'Q277', 'Q278', 'Q279',
    'Q280', 'Q281', 'Q282', 'Q283', 'Q284', 'Q285', 'Q286', 'Q287',
    'Q288R', 'Q289', 'Q290'
]
```
→ These are standard WVS demographic, settlement, and interviewer assessment variables.

## How to Reproduce This Exact Dataset

1. Place your cleaned country files (one .xlsx per country) into the folder:  
   `cleaned_countries/`

2. Run the script (Python 3.8+ recommended):
   ```bash
   python create_final_dataset.py
   ```

3. Output will appear in:
   - `processed_countries/` → final modeling files
   - `countries_metadata_summary.xlsx` → quick sanity-check table

### Required Python packages
```bash
pip install pandas openpyxl
```

## Metadata Summary (countries_metadata_summary.xlsx)
Contains for each country:
- File name
- Country code (`B_COUNTRY`)
- Total rows & columns
- Number of columns in `Human_chara` and `QA_pairs`
- How many of the 38 expected human variables were actually present

Use this file to verify consistency across countries (e.g., no country accidentally missing all demographics).

## Important Notes
- `B_COUNTRY` and `D_INTERVIEW` are used as a composite index in the modeling sheets → guarantees unique respondent IDs.
- Never use the `Total` sheet for modeling (it still contains PCA and ranking columns).
- The split is designed for **causal inference / LLM prompting / survey analysis** where you want clean separation between controls (human chara) and outcomes/predictors (QA items).

## Contact / Last Updated
- Created by: [Your Name]
- Last updated: November 2025
- Questions? Just open this README again – everything is documented here

You now have a fully reproducible, publication-ready WVS Wave 7 dataset.
```

Save this as `README.md` in your project root.  
Future-you will thank present-you!
```

#### 1. Start with clean WVS Wave 7 data
```text
WVS_Longitudinal_1981_2022_excel_v5_0.xls
→ Extract only Wave 7 (S003 = 7)
→ Keep only countries with ≥50 respondents
```

#### 2. Clean per country and remove “Not asked” columns
```bash
python clean_wvs_remove_not_asked_per_country_FIXED.py \
  --input_file WVS_Wave7_full.xlsx \
  --output_dir cleaned_per_country
```
→ Produces one clean file per country with original two header rows preserved.

#### 3. Select the most representative respondent(s) per country
```bash
python sample_representative_wvs_perfect.py \
  --file_path WVS_Wave7_full.xlsx \
  --n 5 \
  --output_dir final_representatives
```
→ Uses PCA on 38 core demographic/value items  
→ Selects 1–5 most central individuals (closest to country centroid)  
→ Saves `per_country/*.xlsx` and master file

#### 4. Sample exactly 200 GlobalOpinionQA questions per country
```bash
python generate_opinion_qa_200_fixed.py \
  --qa_json            global_opinion_qa.json \
  --representatives_dir final_representatives/per_country \
  --output_dir          opinion_200_final \
  --max_questions       200 \
  --seed                42
```

This script:
- Keeps **only countries present in both WVS and GlobalOpinionQA**
- Randomly samples **exactly 200** questions per country (or all if fewer)
- Renames opinion questions to `GOQ0`, `GOQ1`, ..., `GOQ199` (to avoid clash with WVS Q-codes)
- Saves the **authoritative sampled question set** as  
  `opinion_200_final/sampled_200_global_opinion_qa.json`
- Generates one final simulation-ready Excel file per country in  
  `opinion_200_final/excel_per_country/`

#### Final file format (example: Australia_opinion_200.xlsx)

| Sheet        | Columns                                                                                 | Header rows |
|--------------|-----------------------------------------------------------------------------------------|-------------|
| Total        | `Interview ID`, `Country`, 38 WVS items (`Q260`…`Q290`), `GOQ0`…`GOQ199`                 | Row 1: codes<br>Row 2: full question text + options |
| Human_chara  | `Interview ID`, `Country`, 38 WVS items only                                            | Same as original WVS structure   | Row 1: codes<br>Row 2: empty for WVS items |
| QA_pair      | `Interview ID`, `Country`, `GOQ0`…`GOQ199` only                                         | Row 1: codes<br>Row 2: full GlobalOpinionQA text |

All answer cells for `GOQ*` columns are intentionally **blank** – they are filled later by the LLM simulation.

### Key files you must keep for full reproducibility
```
global_opinion_qa.json                     ← original ∼120k questions
final_representatives/per_country/*.xlsx   ← selected human agents
opinion_200_final/
   ├─ sampled_200_global_opinion_qa.json   ← canonical 200-per-country sample (seed 42)
   └─ excel_per_country/*.xlsx             ← final simulation datasets
```

With these files and the scripts above, anyone can **exactly reproduce** the entire agent + question dataset used in the paper.

Seed 42 is used everywhere → results are deterministic.

You’re all set – happy simulating!

### Using LMDeploy for Experiments on NVIDIA Spark

#### Step 1: Install LMDeploy Package
LMDeploy is primarily installed via pip. Use the pre-built CUDA 12 wheel for Spark compatibility (avoids compilation). Run in your conda env:

```bash
# Create/activate env
conda create -n lmdeploy python=3.10
conda activate lmdeploy

# Install LMDeploy (CUDA 12 pre-built; includes TurboMind)
pip install lmdeploy[all,cuda12]  # 'all' for full backends (Turbomind, PyTorch, vLLM)

# Verify
python -c "import lmdeploy; print(lmdeploy.__version__)"  # Should be 0.10.2+
nvidia-smi  # Confirm GPU (Spark B200)
```

- **If CUDA 11.8 needed** (rare for Spark): `pip install lmdeploy[all,cuda11]`.
- **AMD/ROCm support**: If not Spark, add `pip install lmdeploy[all,rocm]`.
- **Size**: ~500MB; includes engines for batching.

#### Step 2: Download/Prepare Model
Clone a Hugging Face model (e.g., Llama-3-8B for testing):

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct /path/to/model
```

#### Step 3: Initiate the Server
Use `lmdeploy serve api` for an OpenAI-compatible server (supports batch via `/v1/chat/completions`). For Spark, enable TurboMind backend (optimized for Blackwell; up to 1.8x vLLM throughput).

```bash
# Basic server (single GPU, TurboMind backend)
lmdeploy serve api \
  --model /path/to/model  # HF repo or local path
  --backend turbomind     # For Spark/Blackwell acceleration
  --tp 1                 # Tensor parallelism (1 for single GPU; scale to 8+ for multi-Spark)
  --host 0.0.0.0         # Bind to network
  --port 23333           # Default port
  --trust-remote-code    # For HF models

# Advanced (batch-optimized, multi-GPU)
lmdeploy serve api \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --backend turbomind \
  --tp 4                 # e.g., 4 Spark GPUs
  --quant-policy w4a16   # Optional: 4-bit quantization for speed (1.5x on Spark)
  --host 0.0.0.0 \
  --port 23333 \
  --max-batch-size 128   # Batch up to 128 requests
```

- **Server URL**: `http://localhost:23333/v1` (OpenAI API format).
- **Batch Inference**: Send array of `messages` (see code below). Server handles persistent batching (dynamic sizing).
- **Spark Optimization**: TurboMind auto-detects Blackwell; use `--quant-policy w4a16` for 1.5x speed on large models.
- **Logs**: Check `~/.lmdeploy/logs` for errors; monitor with `nvidia-smi`.
- **Stop**: Ctrl+C; restart with same command for persistence.

#### Troubleshooting
- **GPU Detection**: Run `lmdeploy --version` and `nvidia-smi`. Ensure Spark drivers (R550+).
- **Memory Errors**: Use `--quant-policy w4a16` or `--tp 8` for multi-Spark.
- **Batch Limits**: Server default max_batch_size=128; adjust with `--max-batch-size 256`.
- **Docs**: [LMDeploy Server Guide](https://lmdeploy.readthedocs.io/en/latest/serving/server.html) (2025 updates include Blackwell support).

This setup is **production-ready** for experiments—batch throughput on Spark can hit 1000+ tokens/sec for 8B models. If you need multi-node scaling, add `--server-port 23333 --worker-port 50000`. Let me know for tweaks!**