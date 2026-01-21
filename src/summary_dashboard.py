import pandas as pd
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"

def generate_dashboard():
    # 1. Find the latest final prediction file
    prediction_files = list(OUTPUT_DIR.glob("FINAL_PREDICTIONS_*.csv"))
    if not prediction_files:
        print("❌ No final predictions found. Run src/predict.py first!")
        return
    
    # Pick the most recent one
    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)

    # 2. Calculate Enhanced Statistics
    total_samples = len(df)
    total_failures = df['Prediction'].sum()
    avg_risk = df['Risk_Percentage'].mean()
    
    # Identify high-risk critical machines (Risk > 90%)
    critical_machines = len(df[df['Risk_Percentage'] >= 90])
    
    # Identify machines where the AI is "Unsure" (Uncertainty > threshold)
    low_trust_count = (df['Trust_Level'].str.contains("Low Trust")).sum()
    
    # 3. Print the Professional Dashboard
    print("\n" + "="*60)
    print(" 🛡️  INDUSTRIAL MACHINE SAFETY: ADVANCED RISK DASHBOARD ")
    print("="*60)
    print(f"📊 Dataset Analyzed:      {latest_file.name.replace('FINAL_PREDICTIONS_', '')}")
    print(f"✅ Total Units Scanned:    {total_samples}")
    print(f"🚨 Predicted Failures:     {total_failures}")
    print(f"📉 Average Factory Risk:   {avg_risk:.2f}%")
    print("-" * 60)
    
    print(f"🔥 CRITICAL RISK (>=90%):  {critical_machines} units")
    print(f"⚠️  UNRELIABLE (Low Trust): {low_trust_count} units")
    print("-" * 60)
    
    # 4. Actionable Insight
    if critical_machines > 0:
        print(f"💡 IMMEDIATE ACTION: Shutdown/Inspect the {critical_machines} critical units.")
    if low_trust_count > 0:
        print(f"💡 SAFETY NOTICE: {low_trust_count} predictions require expert verification.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_dashboard()