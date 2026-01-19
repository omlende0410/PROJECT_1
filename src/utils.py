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
