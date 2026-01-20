
# Purpose:
# Run everything in order.

# This file should:
# Import functions from:
# data_check
# data_cleaning
# feature_analysis
# visualize_data
# Execute steps sequentially

from data_check import check_data
import data_cleaning  # runs the cleaning script automatically
import feature_analysis  # merged file that does analysis + visualization automatically

from pathlib import Path

# ---------------- STEP 1: Check all raw datasets ----------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

for raw_file in RAW_DIR.glob("*.csv"):
    check_data(raw_file.name)  # runs check_data for each dataset

# ---------------- STEP 2: Clean all raw datasets ----------------
# data_cleaning.py already loops over all CSVs in RAW_DIR
# so just importing it is enough

# ---------------- STEP 3: Feature analysis & visualization ----------------
# feature_analysis.py now handles everything automatically

print("\nPipeline completed successfully ✅")
