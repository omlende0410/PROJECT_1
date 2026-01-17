import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
data_path = ROOT_DIR / "data" / "salary_data.csv"

df = pd.read_csv(data_path)

print("Dataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
