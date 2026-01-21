import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))
from utils import detect_target_column

def estimate_uncertainty():
    MODEL_DIR = ROOT_DIR / "models"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    
    for model_file in MODEL_DIR.glob("*.pkl"):
        print(f"\n--- Estimating Uncertainty for: {model_file.name} ---")
        
        # 1. Load the Model and the corresponding Cleaned Data
        model = joblib.load(model_file)
        dataset_name = model_file.stem.replace("_model", "")
        df = pd.read_csv(PROCESSED_DIR / f"{dataset_name}.csv")
        
        # 2. Prepare Data (X)
        target = detect_target_column(df)
        X = df.drop(columns=[target])
        X = pd.get_dummies(X, drop_first=True)
        
        # 3. Calculate Uncertainty
        # For Classification: We look at the distribution of probabilities across all trees
        if hasattr(model, "predict_proba"):
            predictions = []
            for tree in model.estimators_:
                predictions.append(tree.predict_proba(X))
            
            # Variance across trees (Higher variance = Higher Uncertainty)
            all_tree_preds = np.array(predictions)
            uncertainty_score = np.var(all_tree_preds, axis=0).mean(axis=1)
            
            print(f"Max Uncertainty detected: {uncertainty_score.max():.4f}")
            print(f"Average Confidence: {1 - uncertainty_score.mean():.4f}")
            
        # 4. Flag Potential Failures
        df['uncertainty_score'] = uncertainty_score
        df['failure_risk'] = np.where(df['uncertainty_score'] > 0.15, "High Risk", "Low Risk")
        
        # 5. Save Results
        output_path = ROOT_DIR / "output" / f"{dataset_name}_failure_predictions.csv"
        df.to_csv(output_path, index=False)
        print(f"Failure risk analysis saved to: {output_path.name}")

if __name__ == "__main__":
    estimate_uncertainty()