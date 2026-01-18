import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT_DIR / "data" / "salary_data.csv")

df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Experience_Years"] = df["Experience_Years"].fillna(df["Experience_Years"].median())

print("After cleaning:")
print(df.isnull().sum())
