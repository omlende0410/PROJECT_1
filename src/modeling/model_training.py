from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ---------------- PATH SETUP ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("\n--- MODEL TRAINING ---")

# ---------------- TRAIN ON ALL DATASETS ----------------
for processed_file in PROCESSED_DIR.glob("*.csv"):
    print(f"\nTraining model for: {processed_file.name}")

    df = pd.read_csv(processed_file)

    if df.shape[1] < 2:
        print("Skipping (not enough columns)")
        continue

    # Features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Handle non-numeric data
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Save model
    model_path = MODEL_DIR / f"{processed_file.stem}_model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved as: {model_path.name}")

print("\nModel training completed ✅")
