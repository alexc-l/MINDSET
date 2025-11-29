# clean_wvs_remove_not_asked_per_country_FIXED.py
import pandas as pd
import argparse
from pathlib import Path

# All known "Not asked" / missing patterns
NOT_ASKED_PATTERNS = {
    "not asked", "not asked in survey", "not asked in this wave", "not applicable", "no answer", "don't know", "refused", "missing", "inapplicable",
    "-1", "-2", "-3", "-4", "-5", "na", "n/a", "nan", "none", "dk", "ref", "inapp", ""
}

def is_not_asked_column(series):
    """Return True if >95% of values are missing or 'Not asked' variants"""
    if len(series) == 0:
        return True
    cleaned = series.astype(str).str.strip().str.lower()
    invalid = cleaned.isin(NOT_ASKED_PATTERNS) | cleaned.isin(["nan", "<na>"])
    nulls = series.isna()
    return (invalid.sum() + nulls.sum()) / len(series) > 0.95

def main(input_file, output_dir):
    print(f"Loading full dataset from {input_file}...")
    # Load without header → row 0 = codes, row 1 = descriptions
    df_full = pd.read_excel(input_file, header=None)
    print(f"Total rows loaded: {len(df_full)}")

    # Extract global headers and description rows
    header_row = df_full.iloc[0]      # Row 0: column codes (e.g., Q260)
    desc_row   = df_full.iloc[1]      # Row 1: human-readable descriptions
    data_df    = df_full.iloc[2:].copy()

    # Assign headers from row 0 so we can group
    data_df.columns = header_row

    # Ensure B_COUNTRY exists
    if 'B_COUNTRY' not in data_df.columns:
        raise ValueError("Column 'B_COUNTRY' not found. Check your file.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_countries = data_df['B_COUNTRY'].nunique()
    print(f"Found {total_countries} countries. Processing...\n")

    for idx, (country, group) in enumerate(data_df.groupby('B_COUNTRY'), 1):
        print(f"[{idx}/{total_countries}] Processing: {country} ({len(group)} respondents)")

        # Find columns that are >95% "Not asked" in THIS country
        cols_to_check = [col for col in group.columns if col in header_row.values]
        not_asked_cols = []
        for col in cols_to_check:
            if is_not_asked_column(group[col]):
                sample_vals = group[col].dropna().unique()[:3]
                print(f"    → Removing '{col}' | Sample: {list(sample_vals)}")
                not_asked_cols.append(col)

        # Only drop columns that actually exist in this group
        cols_to_drop = [c for c in not_asked_cols if c in group.columns]
        cleaned_data = group.drop(columns=cols_to_drop)

        # Apply same removal to global header and description rows
        # (safe: use .loc with existing labels)
        remaining_cols = cleaned_data.columns
        cleaned_header = header_row[header_row.isin(remaining_cols)]
        cleaned_desc   = desc_row[header_row.isin(remaining_cols)]

        # Reconstruct final DataFrame: header + desc + data
        final_df = pd.concat([
            pd.DataFrame([cleaned_header.values], columns=remaining_cols),
            pd.DataFrame([cleaned_desc.values],   columns=remaining_cols),
            cleaned_data[remaining_cols].reset_index(drop=True)
        ], ignore_index=True)

        # Save per country
        safe_name = str(country).replace(' ', '_').replace('/', '_').replace('\\', '_')
        out_file = output_path / f"{safe_name}_cleaned.xlsx"
        final_df.to_excel(out_file, index=False, header=False)
        removed = len(not_asked_cols)
        kept = len(remaining_cols)
        print(f"    → Saved {out_file.name} | Kept {kept} columns | Removed {removed}\n")

    print(f"All done! Cleaned files saved to: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean WVS per country: remove 'Not asked' columns")
    parser.add_argument('--input_file', type=str, required=True, help='Path to full WVS Excel file')
    parser.add_argument('--output_dir', type=str, default='cleaned_per_country', help='Output folder')
    args = parser.parse_args()
    main(args.input_file, args.output_dir)