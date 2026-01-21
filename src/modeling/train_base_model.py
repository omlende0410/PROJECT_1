import pandas as pd
import joblib  # Used to save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from pathlib import Path
import sys

# Adding src to path so we can import utils
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import load_csv, detect_target_column

def train_model():
    # Setup Paths
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MODEL_SAVE_DIR = ROOT_DIR / "models"
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    
    for cleaned_file in PROCESSED_DIR.glob("*_cleaned.csv"):
        print(f"\n--- Training Model for: {cleaned_file.name} ---")
        
        # 1. Load Data
        df = pd.read_csv(cleaned_file)
        
        # 2. Detect Target & Features
        target = detect_target_column(df)
        X = df.drop(columns=[target])
        y = df[target]
        
        # Convert text to numbers (One-Hot Encoding)
        X = pd.get_dummies(X, drop_first=True)
        
        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Choose Model & Train
        if y.dtype == 'object' or y.nunique() < 10:
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(f"Type: Classification | Accuracy: {accuracy_score(y_test, preds):.2f}")
        else:
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(f"Type: Regression | RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}")

        # 5. Save the trained model file
        model_path = MODEL_SAVE_DIR / f"{cleaned_file.stem}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to: models/{model_path.name}")

if __name__ == "__main__":
    train_model()