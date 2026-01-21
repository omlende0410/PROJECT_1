import pandas as pd
import numpy as np

np.random.seed(42)
rows = 4999

# 1. Generate base data
data = {
    'Type': np.random.choice(['Low', 'Medium', 'High', '??'], rows), # Added a '??' noise category
    'Air_Temp_K': np.random.uniform(295, 305, rows),
    'Process_Temp_K': np.random.uniform(305, 315, rows),
    'Rotational_Speed_RPM': np.random.uniform(1200, 2800, rows),
    'Torque_Nm': np.random.uniform(10, 80, rows),
    'Tool_Wear_Min': np.random.randint(0, 250, rows),
}

df = pd.DataFrame(data)

# 2. INTENTIONALLY ADD PROBLEMS
# Add Missing Values (NaNs) - about 5% of the data
for col in ['Air_Temp_K', 'Torque_Nm']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Add Duplicate Rows
duplicates = df.iloc[:50] # Take first 50 rows
df = pd.concat([df, duplicates], ignore_index=True)

# Add Outliers (Impossible values)
df.loc[10, 'Air_Temp_K'] = 9999.0 
df.loc[20, 'Rotational_Speed_RPM'] = -500.0

# Add Wrong Data Types (Store numbers as strings in one column)
df['Tool_Wear_Min'] = df['Tool_Wear_Min'].astype(str)

# 3. Create Target
failure_prob = (np.random.rand(len(df)) < 0.05)
df['Machine_Failure'] = np.where(failure_prob, 1, 0)

# Save to your raw folder
df.to_csv('data/raw/messy_machine_data.csv', index=False)
print(f"✅ Created messy_machine_data.csv with {len(df)} rows.")
print("⚠️ Features: Missing values, Duplicates, Outliers, and String-types included!")