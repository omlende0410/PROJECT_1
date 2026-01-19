# Purpose:
# Convert numbers into visuals.

# This file should:
# Load cleaned data
# Create:
# histogram
# box plot
# scatter plot
# count plot (for categorical)
# Save plots to outputs/plots/

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
data_path = ROOT_DIR / "data" / "processed" / "sample.csv"
df = pd.read_csv(data_path)

# Salary vs Experience
plt.figure(figsize=(8,5))
plt.scatter(df["experience_years"], df["salary"], color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Salary distribution
plt.figure(figsize=(8,5))
plt.hist(df["salary"], bins=10, color="green", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
