# Purpose:
# Fix bad data and create clean dataset.

# This file should:
# Load raw data
# Handle missing values:
# numerical → mean / median
# categorical → mode
# Remove duplicates
# Fix obvious garbage:
# negative values where impossible
# incorrect data types
# Rename columns to clean names
# Save cleaned data to data/processed/
import pandas as pd
from pathlib import Path

# ------------------ PATH SETUP ------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

# Create folders if they don't exist
PROCESSED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------ PROCESS ALL DATASETS ------------------
for raw_file in RAW_DIR.glob("*.csv"):
    print(f"\nProcessing file: {raw_file.name}")

    # Load data
    df = pd.read_csv(raw_file)

    # File naming
    file_stem = raw_file.stem
    processed_file = PROCESSED_DIR / f"{file_stem}_cleaned.csv"
    report_file = OUTPUT_DIR / f"{file_stem}_report.txt"

    # ------------------ BEFORE CLEANING ------------------
    rows_before = df.shape[0]
    cols_before = df.shape[1]
    nulls_before = df.isnull().sum()
    duplicates_before = df.duplicated().sum()

    # ------------------ CLEANING LOGIC ------------------
    df = df.drop_duplicates()
    df = df.dropna()

    # ------------------ AFTER CLEANING ------------------
    rows_after = df.shape[0]
    cols_after = df.shape[1]
    nulls_after = df.isnull().sum()
    duplicates_after = df.duplicated().sum()

    # ------------------ SAVE CLEANED DATA ------------------
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
        f.write(f"Rows after cleaning: {rows_after}\n\n")

        f.write("NULL VALUES (Before)\n")
        f.write(nulls_before.to_string())
        f.write("\n\n")

        f.write("NULL VALUES (After)\n")
        f.write(nulls_after.to_string())
        f.write("\n\n")

        f.write("DUPLICATES\n")
        f.write(f"Duplicates before cleaning: {duplicates_before}\n")
        f.write(f"Duplicates after cleaning: {duplicates_after}\n")

    print(f"Cleaned data saved to: {processed_file.name}")
    print(f"Report generated: {report_file.name}")

print("\nAll datasets processed successfully ✅")
