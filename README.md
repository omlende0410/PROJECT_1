# Automated CSV Data Cleaning, Risk Analysis & Visualization Pipeline 🚀

## Project Description
A robust Python framework designed to detect and predict machine learning model failures. Unlike traditional "black-box" AI, this system uses **Uncertainty Estimation** and **Confidence Analysis** to flag when a model is guessing, ensuring reliability in high-stakes environments.

## Problem Statement
Machine learning models often fail "silently"—they provide incorrect predictions with high statistical confidence on unseen or out-of-distribution data. This project aims to bridge the safety gap by detecting high-risk predictions before they lead to real-world failures.

## The Need
In safety-critical domains such as **Healthcare, Autonomous Systems, and Finance**, a wrong prediction can be catastrophic. Predicting failure probability improves system reliability and allows for human-in-the-loop intervention when the AI is uncertain.

## Objectives
* **Automated Pipeline:** Build a scalable, dataset-agnostic engine for any CSV data.
* **Uncertainty Estimation:** Quantify model "shakiness" using ensemble variance.
* **Risk Detection:** Categorize predictions into "Low Risk" and "High Risk" zones.
* **Scalability:** Process datasets ranging from 10k to 50k+ rows automatically.

## Tech Stack
* **Core:** Python, NumPy, Pandas
* **ML/Modeling:** Scikit-learn, (Planned) PyTorch for MC Dropout
* **Visualization:** Matplotlib, Seaborn
* **Persistence:** Joblib (Model Serialization)

## Workflow
1.  **Data Validation:** Automatic quality checks on raw CSV files.
2.  **Cleaning:** Automated handling of missing values and duplicates.
3.  **Analysis:** Statistical profiling and feature visualization.
4.  **Baseline Training:** Automated model fitting for classification or regression.
5.  **Uncertainty Calculation:** Measuring prediction variance across the model ensemble.
6.  **Failure Visualization:** Generating risk distribution plots and flagging anomalies.

## How to Run
1.  **Prepare Data:** Drop any number of CSV files into `data/raw/`.
2.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn joblib
    ```
3.  **Execute Pipeline:**
    ```bash
    python src/analysis_pipeline.py
    ```

## Project Status & Highlights
* ✅ **Fully Data-Agnostic:** Supports multiple datasets without code changes.
* ✅ **End-to-End Automation:** From raw data to failure-risk visualization.
* ✅ **Modular Architecture:** Clean separation of concerns (src/modeling, src/utils).
* 🚧 **In Progress:** Deep Learning integration for MC Dropout and OOD (Out-of-Distribution) detection.

## Project Structure

PRO_1/
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Project dependencies
├── README.md                  # Project overview & usage
├── industrial_data.py         # Standalone execution script
│
├── .venv/                     # Virtual environment
│
├── data/
│   ├── raw/                   # Original input CSV files
│   ├── processed/             # Cleaned & transformed data
│   └── external/              # External datasets (optional)
│
├── docs/                      # Documentation & notes
│
├── models/                    # Saved ML models
│   └── messy_machine_data_cleaned_model.pkl
│
├── src/                       # Source code
│   ├── modeling/              # Model training logic
│   ├── analysis_pipeline.py   # End-to-end pipeline runner
│   ├── data_check.py          # Data validation
│   ├── data_cleaning.py       # Data preprocessing
│   ├── feature_analysis.py    # EDA & feature analysis
│   ├── predict.py             # Prediction logic
│   ├── summary_dashboard.py   # Visualizations & summaries
│   └── utils.py               # Helper utilities
│
├── output/                    # Generated results
│   ├── FINAL_PREDICTIONS_messy_machine.csv
│   ├── messy_machine_data_cleaned_failure.csv
│   ├── messy_machine_data_uncertainty_plot.png
│   └── messy_machine_data_report.txt
