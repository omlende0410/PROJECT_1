import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import your custom tools
from src.utils import fix_data_types, detect_target_column

def run_predictions(new_data_path):
    # 1. Load the Trained Model
    model_files = list((ROOT_DIR / "models").glob("*.pkl"))
    if not model_files:
        print("❌ No trained model found! Please run the training pipeline first.")
        return
    
    # Load the first available model
    model = joblib.load(model_files[0])
    expected_features = model.feature_names_in_
    
    # 2. Load and Clean New Data
    df = pd.read_csv(new_data_path)
    print(f"--- Processing: {new_data_path.name} ---")
    
    # Rescue malformed types and remove invalid rows
    df = fix_data_types(df)
    df = df.dropna()

    # 3. Feature Engineering & Alignment
    # One-hot encode categorical variables (like 'Type')
    X = pd.get_dummies(df)
    
    # Remove the target column if it exists in the input
    target = detect_target_column(df)
    if target in X.columns:
        X = X.drop(columns=[target])

    # Ensure X matches the model's expected features exactly
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]

    # 4. Generate Predictions & Risk Scores
    # Predict failure probability (0.0 to 1.0)
    probabilities = model.predict_proba(X)[:, 1]
    # Predict hard labels (0 or 1)
    preds = model.predict(X)
    
    # Calculate Variance across the Random Forest ensemble for Uncertainty
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    uncertainty = np.var(tree_preds, axis=0)

    # 5. Compile Results
    results = df.copy()
    results['Prediction'] = preds
    results['Risk_Percentage'] = (probabilities * 100).round(2)
    results['Uncertainty_Score'] = uncertainty
    
    # Logic: High Trust requires Low Uncertainty AND a clear high/low risk
    results['Trust_Level'] = np.where(
        (uncertainty < 0.1) & ((results['Risk_Percentage'] > 80) | (results['Risk_Percentage'] < 20)), 
        "✅ High Trust", 
        "⚠️ Low Trust (Check Manually)"
    )
    # 6. Save Output
    output_path = ROOT_DIR / "output" / f"FINAL_PREDICTIONS_{new_data_path.stem}.csv"
    results.to_csv(output_path, index=False)
    print(f"✅ Success! Results saved to: {output_path.name}")

if __name__ == "__main__":
    # Point this to your raw data file
    RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "messy_machine_data.csv"
    run_predictions(RAW_DATA_PATH)