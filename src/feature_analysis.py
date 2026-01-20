# Purpose:
# Analyze each feature independently.

# This file should:
# Load cleaned data
# For numerical columns:
# mean, median, std
# min, max
# outlier detection (IQR method)
# For categorical columns:
# unique values
# value counts
# Correlation between numerical columns

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # better heatmaps

# ------------------ PATH SETUP ------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ------------------ SETTINGS ------------------
MAX_POINTS = 5000  # maximum points to plot for big datasets

# ------------------ PROCESS ALL DATASETS ------------------
for processed_file in PROCESSED_DIR.glob("*.csv"):
    print(f"\n--- ANALYSIS & VISUALIZATION: {processed_file.name} ---")

    # Load data
    df = pd.read_csv(processed_file)

    # ------------------ NUMERICAL FEATURES ------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        print("\nNumerical Columns Stats:")
        print(df[num_cols].describe())

        # Sample for large datasets
        df_sample = df[num_cols].sample(min(len(df), MAX_POINTS))

        # Histograms
        df_sample.hist(bins=20, figsize=(12,6), edgecolor="black")
        plt.suptitle(f"Histograms of Numerical Features ({processed_file.name})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(8,6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Heatmap ({processed_file.name})")
        plt.tight_layout()
        plt.show()

    # ------------------ CATEGORICAL FEATURES ------------------
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print("\nCategorical Columns Value Counts:")
        for col in cat_cols:
            print(f"\n{col}:")
            print(df[col].value_counts())

            plt.figure(figsize=(8,4))
            df[col].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
            plt.title(f"{col} Value Counts ({processed_file.name})")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
