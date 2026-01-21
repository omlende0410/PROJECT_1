import sys
from pathlib import Path

# 1. Setup Project Root properly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 2. Import your modules using the 'src.' prefix for reliability
from src.data_check import check_data
from src.modeling.train_base_model import train_model
from src.modeling.uncertainty import estimate_uncertainty
from src.modeling.visualize_failure import plot_failure_distribution

def run_pipeline():
    print("\n" + "="*60)
    print("🚀 STARTING AI-BASED MODEL FAILURE PREDICTOR PIPELINE 🚀")
    print("="*60)

    # ---------------- STEP 1: Data Quality Checks ----------------
    print("\n[STEP 1] Running Data Quality Checks...")
    RAW_DIR = ROOT_DIR / "data" / "raw"
    csv_files = list(RAW_DIR.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw/!")
        return

    for raw_file in csv_files:
        check_data(raw_file.name)

    # ---------------- STEP 2: Automated Cleaning ----------------
    print("\n[STEP 2] Cleaning & Preprocessing All Datasets...")
    import src.data_cleaning  # Executes cleaning loop

    # ---------------- STEP 3: Stats & Visuals ----------------
    print("\n[STEP 3] Generating Statistical Analysis & Plots...")
    import src.feature_analysis # Executes analysis/viz loop

    # ---------------- STEP 4: Model Training ----------------
    print("\n[STEP 4] Training Baseline Models & Saving .pkl files...")
    train_model()

    # ---------------- STEP 5: Uncertainty Logic ----------------
    print("\n[STEP 5] Estimating Model Uncertainty & Failure Risks...")
    estimate_uncertainty()

    # ---------------- STEP 6: Failure Visualization ----------------
    print("\n[STEP 6] Generating Final Risk Charts (PNGs)...")
    plot_failure_distribution()

    print("\n" + "="*60)
    print("✅ ALL PHASES COMPLETED SUCCESSFULLY!")
    print("Check 'output/' for CSVs/PNGs and 'models/' for saved models.")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()