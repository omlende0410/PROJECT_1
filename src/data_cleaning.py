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

from pathlib import Path
import pandas as pd

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

def clean_data(file_name):
    data_path = ROOT_DIR / "data" / "raw" / file_name
    df = pd.read_csv(data_path)
    
    print("\n--- BEFORE CLEANING ---")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Fill missing numerical values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 3. Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Reset index
    df = df.reset_index(drop=True)

    print("\n--- AFTER CLEANING ---")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

    # Save cleaned data
    processed_path = ROOT_DIR / "data" / "processed" / file_name
    df.to_csv(processed_path, index=False)
    print(f"\nCleaned data saved to: {processed_path}")

if __name__ == "__main__":
    clean_data("sample.csv")
