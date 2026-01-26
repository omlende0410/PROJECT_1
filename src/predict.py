import pandas as pd
import joblib
from pathlib import Path
import sys
import numpy as np

# ---------------- PATH SETUP ----------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.utils import fix_data_types, detect_target_column

# ---------------- RISK LABEL LOGIC ----------------
def get_risk_level(risk):
    if risk >= 80:
        return "🔴 High Risk"
    elif risk >= 40:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# ---------------- MAIN PREDICTION FUNCTION ----------------
def run_predictions(new_data_path):
    # 1. Load trained model
    model_files = list((ROOT_DIR / "models").glob("*.pkl"))
    if not model_files:
        print("❌ No trained model found! Run training first.")
        return

    model = joblib.load(model_files[0])
    expected_features = model.feature_names_in_

    # 2. Load & clean data
    df = pd.read_csv(new_data_path)
    print(f"\n📂 Processing file: {new_data_path.name}")

    df = fix_data_types(df)
    df = df.dropna()

    # 3. Feature engineering
    X = pd.get_dummies(df)

    target = detect_target_column(df)
    if target in X.columns:
        X = X.drop(columns=[target])

    # Align features
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]

    # 4. Predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    # 5. Uncertainty estimation (Random Forest ensemble variance)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    uncertainty = np.var(tree_preds, axis=0)

    # 6. Compile results
    results = df.copy()
    results["Prediction"] = predictions
    results["Risk_Percentage"] = (probabilities * 100).round(2)
    results["Risk_Level"] = results["Risk_Percentage"].apply(get_risk_level)
    results["Uncertainty_Score"] = uncertainty.round(4)

    results["Trust_Level"] = np.where(
        (results["Uncertainty_Score"] < 0.1) &
        ((results["Risk_Percentage"] >= 80) | (results["Risk_Percentage"] <= 20)),
        "✅ High Trust",
        "⚠️ Low Trust (Manual Review)"
    )

    # 7. Save output
    output_path = ROOT_DIR / "output" / f"FINAL_PREDICTIONS_{new_data_path.stem}.csv"
    results.to_csv(output_path, index=False)

    print("✅ Prediction completed successfully!")
    print(f"📁 Output saved at: {output_path.name}")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "messy_machine_data.csv"
    run_predictions(RAW_DATA_PATH)
