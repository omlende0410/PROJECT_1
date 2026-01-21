import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Setup Paths
# Move up two levels from src/modeling/ to reach the root PRO_1/
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils import detect_target_column

def check_ood(df_new, model_name):
    """
    Checks if the new data is 'Out-of-Distribution' (OOD).
    Specifically flags impossible values like 9999K temperatures.
    """
    print(f"--- Running OOD Check for {model_name} ---")
    
    # Flag temperatures that are physically impossible for this machine context
    ood_mask = (df_new['Air_Temp_K'] > 500) | (df_new['Rotational_Speed_RPM'] < 0)
    return ood_mask

def estimate_uncertainty():
    MODEL_DIR = ROOT_DIR / "models"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    
    # Check if directories exist
    if not MODEL_DIR.exists():
        print("❌ Model directory not found!")
        return

    for model_file in MODEL_DIR.glob("*.pkl"):
        # Get base name (e.g., 'messy_machine_data_cleaned')
        dataset_name = model_file.stem.replace("_model", "")
        
        # Look for the matching CSV file
        csv_path = PROCESSED_DIR / f"{dataset_name}.csv"
        
        if not csv_path.exists():
            print(f"⚠️ Skipping: {csv_path.name} not found in processed folder.")
            continue

        print(f"--- Estimating Uncertainty for: {csv_path.name} ---")
        df = pd.read_csv(csv_path)
        
        # Load the model
        model = joblib.load(model_file)
        
        # 1. Run OOD Check (Out-of-Distribution)
        is_ood_row = check_ood(df, dataset_name)
        
        # 2. Prepare Features
        target = detect_target_column(df)
        X = df.drop(columns=[target])
        
        # Ensure only numeric columns and match expected feature names
        X = pd.get_dummies(X, drop_first=True)
        expected_features = model.feature_names_in_
        
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
        # 3. Calculate Ensemble Variance
        # Get predictions from every individual tree in the forest
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Variance measures how much the trees 'disagree'
        uncertainty_score = np.var(tree_preds, axis=0)

        # 4. Flag High Risk & OOD
        df['uncertainty_score'] = uncertainty_score
        df['failure_risk'] = np.where(df['uncertainty_score'] > 0.15, "High Risk", "Low Risk")
        
        # If the row was marked as OOD, override the risk label
        df.loc[is_ood_row, 'failure_risk'] = "OOD - PHYSICAL ANOMALY"

        # 5. Save Results to Output
        output_path = ROOT_DIR / "output" / f"{dataset_name}_failure_predictions.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Analysis saved to: {output_path.name}")

if __name__ == "__main__":
    estimate_uncertainty()