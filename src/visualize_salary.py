import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load data
data_path = ROOT_DIR / "data" / "salary_data.csv"
df = pd.read_csv(data_path)

# Plot salary vs experience
plt.figure()
plt.grid()
plt.plot(df["Experience_Years"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")

# Show plot
plt.show()
