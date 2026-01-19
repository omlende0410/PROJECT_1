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
import seaborn as sns

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
data_path = ROOT_DIR / "data" / "processed" / "sample.csv"
df = pd.read_csv(data_path)

print("\n--- FEATURE ANALYSIS ---")

# Numerical features
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("\nNumerical Columns Stats:")
print(df[num_cols].describe())

# Categorical features
cat_cols = df.select_dtypes(include=["object"]).columns
print("\nCategorical Columns Value Counts:")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# --- PLOTS ---

# 1. Histograms for numerical columns
df[num_cols].hist(bins=10, figsize=(10,5), edgecolor="black")
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(6,4))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
