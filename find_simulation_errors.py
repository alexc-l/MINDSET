# find_simulation_errors.py
import pandas as pd
from pathlib import Path
import argparse


def find_error_files(search_dir: str, pattern: str = "*.xlsx"):
    """
    Scan all XLSX files in search_dir and report which contain the word "ERROR"
    in the simulated_answer column.
    """
    search_path = Path(search_dir)
    files = list(search_path.rglob(pattern))

    if not files:
        print(f"No files found matching {pattern} in {search_dir}")
        return

    print(f"Scanning {len(files)} files for 'ERROR' in simulated answers...\n")

    error_files = []
    total_errors = 0

    for file_path in files:
        try:
            # Try reading from common sheet names
            df =  pd.read_csv(file_path)
            if df is None:
                print(f"  Could not read {file_path.name}")
                continue

            # Look for column that might contain answers
            answer_col = None
            for col in df.columns:
                if 'answer' in str(col).lower() or 'simulated' in str(col).lower():
                    answer_col = col
                    break

            if answer_col is None:
                # Fallback: check all string columns
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].astype(str).str.contains("ERROR", case=True, na=False).any():
                        error_files.append(file_path)
                        count = df[col].astype(str).str.contains("ERROR", case=True, na=False).sum()
                        total_errors += count
                        print(f"  ERROR found in {file_path.name} ({count} rows)")
                        break
            else:
                if df[answer_col].astype(str).str.contains("ERROR", case=True, na=False).any():
                    error_files.append(file_path)
                    count = df[answer_col].astype(str).str.contains("ERROR", case=True, na=False).sum()
                    total_errors += count
                    print(f"  ERROR found in {file_path.name} → column '{answer_col}' ({count} errors)")

        except Exception as e:
            print(f"  Failed to read {file_path.name}: {e}")

    print(f"\nSUMMARY:")
    print(f"  Total files scanned : {len(files)}")
    print(f"  Files with ERROR    : {len(error_files)}")
    print(f"  Total ERROR rows    : {total_errors}")

    if error_files:
        error_list_file = Path("ERROR_FILES_TO_RERUN.txt")
        with open(error_list_file, 'w') as f:
            f.write("# Files containing simulation errors (rerun these)\n")
            for ef in error_files:
                f.write(str(ef) + "\n")
        print(f"\nList saved to: {error_list_file}")
        print("Just rerun these files — your simulation will resume perfectly!")
    else:
        print("No errors found! Your simulation is clean!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find simulation output files containing 'ERROR'")
    parser.add_argument('--dir', type=str, default="output",
                        help="Directory containing simulation XLSX files (default: ./output)")
    parser.add_argument('--pattern', type=str, default="*.xlsx",
                        help="File pattern (default: *.xlsx)")
    args = parser.parse_args()

    find_error_files(args.dir, args.pattern)