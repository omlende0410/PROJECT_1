import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "output"

st.set_page_config(
    page_title="AI Failure Risk Predictor",
    layout="wide"
)

st.title("🛡️ AI-Based Model Failure & Risk Predictor")
st.caption("Hackathon-ready uncertainty & risk analysis dashboard")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📤 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload Machine CSV", type=["csv"])

if uploaded_file:
    file_path = DATA_RAW / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully")

# ---------------- RUN PIPELINE ----------------
if st.sidebar.button("🚀 Run Full Pipeline"):
    with st.spinner("Running analysis pipeline..."):
        subprocess.run(["python", "src/analysis_pipeline.py"])
    st.success("Pipeline completed successfully!")

# ---------------- LOAD RESULTS ----------------
prediction_files = list(OUTPUT_DIR.glob("FINAL_PREDICTIONS_*.csv"))

if prediction_files:
    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)

    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", len(df))
    col2.metric("Predicted Failures", int(df["Prediction"].sum()))
    col3.metric("Avg Risk %", f"{df['Risk_Percentage'].mean():.2f}%")
    col4.metric("Low Trust Cases", (df["Trust_Level"].str.contains("Low")).sum())

    # ---------------- FILTER ----------------
    st.subheader("🔍 Filter Results")
    min_risk = st.slider("Minimum Risk %", 0, 100, 0)
    filtered_df = df[df["Risk_Percentage"] >= min_risk]

    # ---------------- TABLE ----------------
    st.subheader("📋 Prediction Results")
    st.dataframe(
        filtered_df.style.applymap(
            lambda x: "background-color: red" if isinstance(x, str) and "High Risk" in x
            else "background-color: yellow" if isinstance(x, str) and "Medium Risk" in x
            else "background-color: lightgreen" if isinstance(x, str) and "Low Risk" in x
            else ""
        ),
        use_container_width=True
    )

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇️ Download Results CSV",
        filtered_df.to_csv(index=False),
        file_name=latest_file.name,
        mime="text/csv"
    )

else:
    st.info("Upload data and run pipeline to see results.")
