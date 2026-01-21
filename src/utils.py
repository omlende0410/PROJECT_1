# Purpose:
# Common functions used everywhere.

# This file should contain:
# Function to get project root path
# Function to load CSV
# Function to save CSV

# Conceptually, it should:
# Use pathlib
# Avoid hardcoding paths
# Be reused by all other files

from pathlib import Path
import pandas as pd

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

def load_csv(relative_path):
    """
    Load a CSV file using a relative path from project root.
    """
    file_path = ROOT_DIR / relative_path
    return pd.read_csv(file_path)

def save_csv(df, relative_path):
    """
    Save a DataFrame to CSV using a relative path from project root.
    """
    file_path = ROOT_DIR / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)

def detect_target_column(df):
    """
    Automatically detect target column.
    Priority:
    1. Column named 'target', 'label', or 'y'
    2. Fallback: last column
    """
    for col in df.columns:
        if col.lower() in ["target", "label", "y", "machine_failure"]:
            return col
    return df.columns[-1]

def fix_data_types(df):
    """
    Automatically converts 'object' columns to numeric if they contain 
    numbers disguised as strings.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # Attempt to convert to numeric, turning non-convertible text to NaN
            converted = pd.to_numeric(df[col], errors='coerce')
            # If the column wasn't purely text (contains actual numbers), update it
            if not converted.isna().all():
                df[col] = converted
    return df