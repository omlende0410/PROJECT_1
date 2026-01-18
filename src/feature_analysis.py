import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT_DIR / "data" / "salary_data.csv")

plt.figure()
plt.scatter(df["Experience_Years"], df["Salary"])
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Experience vs Salary (Scatter)")
plt.grid()
plt.show()

plt.figure()
plt.hist(df["Salary"], bins=6)
plt.xlabel("Salary")
plt.title("Salary Distribution")
plt.grid()
plt.show()
