
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
from data_cleaning import clean_data
import feature_analysis  # feature_analysis.py runs its code on import
import visualize_salary  # visualize_salary.py runs its code on import

# Step 1: Check raw data
check_data("sample.csv")

# Step 2: Clean raw data
clean_data("sample.csv")

# Step 3 & 4: feature_analysis.py and visualize_salary.py will execute automatically on import
