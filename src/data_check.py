# Purpose:
# Inspect the dataset before touching it.

# This file should:
# Load raw CSV using utils.py

# Print:
# number of rows & columns
# column names
# data types
# missing values per column
# duplicate rows count
# basic statistics

# Output:
# Only terminal output
# No plots
# No saving files
from pathlib import Path
import pandas as pd

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

def check_data(file_name):
    # Load data
    data_path = ROOT_DIR / "data" / "raw" / file_name
    df = pd.read_csv(data_path)
    
    print("\n--- DATA CHECK ---")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nStatistical summary:\n", df.describe(include="all"))

# ------------------ RUN AS SCRIPT ------------------
if __name__ == "__main__":
    RAW_DIR = ROOT_DIR / "data" / "raw"
    for raw_file in RAW_DIR.glob("*.csv"):
        check_data(raw_file.name)
