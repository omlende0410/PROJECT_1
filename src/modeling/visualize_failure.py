import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output"

def plot_failure_distribution():
    print("\n--- Generating Failure Risk Visualizations ---")
    
    # Automatically find any file ending with '_failure_predictions.csv'
    prediction_files = list(OUTPUT_DIR.glob("*_failure_predictions.csv"))
    
    if not prediction_files:
        print("⚠️ No failure prediction CSVs found. Did you run uncertainty.py first?")
        return

    for prediction_file in prediction_files:
        # Load the data
        df = pd.read_csv(prediction_file)
        
        # Clean up name for the title (e.g., 'sales_cleaned_failure_predictions' -> 'SALES')
        dataset_label = prediction_file.stem.replace("_failure_predictions", "").replace("_cleaned", "").upper()
        
        print(f"Creating plot for: {dataset_label}")

        plt.figure(figsize=(10, 6))
        
        # Plot the distribution of uncertainty
        # Green = Low Risk, Red = High Risk
        sns.histplot(data=df, x='uncertainty_score', hue='failure_risk', 
                     element="step", palette={'High Risk': 'red', 'Low Risk': 'green'},
                     alpha=0.6)
        
        plt.title(f"AI Failure Risk Distribution: {dataset_label}")
        plt.xlabel("Model Uncertainty Score (Variance)")
        plt.ylabel("Count of Data Points")
        
        # Draw the threshold line where the model starts "failing"
        plt.axvline(x=0.15, color='black', linestyle='--', label='Risk Threshold (0.15)')
        plt.legend()

        # Save the file automatically in the output folder
        save_path = OUTPUT_DIR / f"{dataset_label.lower()}_uncertainty_plot.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved visualization: {save_path.name}")

if __name__ == "__main__":
    plot_failure_distribution()