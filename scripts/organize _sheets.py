import pandas as pd
from pathlib import Path

# ==============================================================
# Configuration
# ==============================================================
input_folder = Path("cleaned_per_country")  # ← your cleaned country files
output_folder = Path("processed_countries")  # ← processed files with 3 sheets
metadata_file = Path("countries_metadata_summary.xlsx")  # ← final summary

output_folder.mkdir(parents=True, exist_ok=True)

# Human characteristic columns
WVS_ITEMS = [
    'N_REGION_WVS', 'N_TOWN', 'G_TOWNSIZE2', 'H_SETTLEMENT', 'H_URBRURAL', 'E_RESPINT',
    'Q260', 'Q261', 'Q262', 'Q263', 'Q264', 'Q265', 'Q266', 'Q267', 'Q268', 'Q269',
    'Q270', 'Q271', 'Q272', 'Q273', 'Q274', 'Q275', 'Q276', 'Q277', 'Q278', 'Q279',
    'Q280', 'Q281', 'Q282', 'Q283', 'Q284', 'Q285', 'Q286', 'Q287',
    'Q288R', 'Q289', 'Q290'
]

INDEX_COLS = ['D_INTERVIEW', 'B_COUNTRY']
EXCLUDE_FROM_SPLITS = {'PC1', 'PC2', 'PC3', 'PC4', 'sacsecval', 'resemaval', 'selection_rank', 'distance_to_centroid'}

# ==============================================================
# Prepare list to collect metadata
# ==============================================================
metadata_records = []

# ==============================================================
# Process each country
# ==============================================================
for file_path in sorted(input_folder.glob("*.xlsx")):
    country_name = file_path.stem  # filename without .xlsx
    print(f"Processing: {country_name}")

    df = pd.read_excel(file_path)

    # --- Basic checks ---
    if not all(col in df.columns for col in INDEX_COLS):
        print(f"    Skipping {country_name}: missing index columns")
        continue

    # 1. Total sheet
    df_total = df.copy()
    n_total_rows, n_total_cols = df_total.shape

    # 2. Human_chara
    human_cols = [c for c in WVS_ITEMS if c in df.columns]
    df_human = df[INDEX_COLS + human_cols].set_index(INDEX_COLS)
    n_human_rows, n_human_cols = df_human.shape

    # 3. QA_pairs (everything except human vars & excluded technical vars)
    cols_to_remove = (set(WVS_ITEMS) | EXCLUDE_FROM_SPLITS) - set(INDEX_COLS)
    qa_columns = [c for c in df.columns if c not in cols_to_remove]
    df_qa = df[qa_columns].set_index(INDEX_COLS)
    n_qa_rows, n_qa_cols = df_qa.shape

    # --- Save the three-sheet file ---
    output_file = output_folder / file_path.name
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_total.to_excel(writer, sheet_name='Total', index=False)
        df_human.to_excel(writer, sheet_name='Human_chara')
        df_qa.to_excel(writer, sheet_name='QA_pairs')

    # --- Record metadata ---
    metadata_records.append({
        'Country_File': country_name,
        'Country_Code': df['B_COUNTRY'].iloc[0] if 'B_COUNTRY' in df.columns else None,
        'Total_Rows': n_total_rows,
        'Total_Columns': n_total_cols,
        'Human_chara_Rows': n_human_rows,
        'Human_chara_Columns': n_human_cols,
        'Human_Variables_Included': len(human_cols),
        'QA_pairs_Rows': n_qa_rows,
        'QA_pairs_Columns': n_qa_cols,
    })

    print(f"    Saved → {output_file.name}")
    print(f"        Human_chara: {n_human_cols} cols | QA_pairs: {n_qa_cols} cols\n")

# ==============================================================
# Save metadata summary
# ==============================================================
if metadata_records:
    metadata_df = pd.DataFrame(metadata_records)

    # Reorder columns nicely
    col_order = [
        'Country_File', 'Country_Code',
        'Total_Rows', 'Total_Columns',
        'Human_chara_Rows', 'Human_chara_Columns', 'Human_Variables_Included',
        'QA_pairs_Rows', 'QA_pairs_Columns'
    ]
    metadata_df = metadata_df[col_order]

    metadata_df.to_excel(metadata_file, index=False)
    print(f"\nMetadata summary saved to: {metadata_file.resolve()}")
    print(metadata_df[['Country_File', 'Human_chara_Columns', 'QA_pairs_Columns', 'Total_Rows']])
else:
    print("No files were processed. Metadata file not created.")

print("\nAll done!")