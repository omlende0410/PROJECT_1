import pandas as pd
from pathlib import Path
import sys

# ------------------ PATH SETUP ------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils import fix_data_types

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ PROCESS ALL DATASETS ------------------
for raw_file in RAW_DIR.glob("*.csv"):
    print(f"\nProcessing file: {raw_file.name}")

    df = pd.read_csv(raw_file)

    file_stem = raw_file.stem
    processed_file = PROCESSED_DIR / f"{file_stem}_cleaned.csv"
    report_file = OUTPUT_DIR / f"{file_stem}_report.txt"

    # ------------------ BEFORE CLEANING ------------------
    rows_before = df.shape[0]
    cols_before = df.shape[1]
    nulls_before = df.isnull().sum()
    duplicates_before = df.duplicated().sum()

    # ------------------ CLEANING LOGIC ------------------
    # Step A: Rescue "String Numbers"
    df = fix_data_types(df) 

    # Step B: Domain Physics Validation (Garbage Removal)
    # Remove 'Sun-hot' outliers (like 9999K) and negative speeds
    df = df[df['Air_Temp_K'] < 500]
    df = df[df['Rotational_Speed_RPM'] >= 0]

    # Step C: Strict Cleaning (Remove duplicates and remaining NaNs)
    df = df.drop_duplicates() 
    df = df.dropna() 

    # ------------------ AFTER CLEANING ------------------
    rows_after = df.shape[0]
    cols_after = df.shape[1]
    nulls_after = df.isnull().sum()
    duplicates_after = df.duplicated().sum()

    # ------------------ SAVE CLEANED DATA ------------------
    # We save as '_cleaned.csv' so Step 5 (Uncertainty) can find it
    df.to_csv(processed_file, index=False)

    # ------------------ WRITE REPORT ------------------
    with open(report_file, "w") as f:
        f.write(f"DATA CLEANING REPORT\n")
        f.write(f"Dataset: {raw_file.name}\n")
        f.write("-" * 45 + "\n\n")

        f.write("STRUCTURE\n")
        f.write(f"Columns before: {cols_before}\n")
        f.write(f"Columns after: {cols_after}\n\n")

        f.write("ROWS\n")
        f.write(f"Rows before cleaning: {rows_before}\n")
        f.write(f"Rows after cleaning: {rows_after}\n")
        f.write(f"Net Rows Lost: {rows_before - rows_after}\n\n")

        f.write("ACTIONS TAKEN\n")
        f.write("- Applied numeric type rescue (utils.py)\n")
        f.write("- Filtered physical anomalies (Temp < 500K, RPM >= 0)\n")
        f.write("- Removed duplicates and null values\n\n")

        f.write("NULL VALUES (Before)\n")
        f.write(nulls_before.to_string())
        f.write("\n\n")

        f.write("DUPLICATES (Before)\n")
        f.write(f"Count: {duplicates_before}\n")

    print(f"Cleaned data saved to: {processed_file.name}")
    print(f"Report generated: {report_file.name}")

print("\nAll datasets processed successfully ✅")