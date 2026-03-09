import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detector")
st.markdown("Upload a CSV file with transaction data to detect fraud.")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("### Preview", df.head())

    # -------------------------------
    # Preprocessing
    # -------------------------------
    scaler = StandardScaler()

    if "Amount" in df.columns:
        df["Amount"] = scaler.fit_transform(df[["Amount"]])

    if "Time" in df.columns:
        df["Time"] = scaler.fit_transform(df[["Time"]])

    # Remove target column if present
    X = df.drop("Class", axis=1) if "Class" in df.columns else df

    # -------------------------------
    # Prediction
    # -------------------------------
    preds = model.predict(X)

    df["Prediction"] = preds
    df["Prediction"] = df["Prediction"].map({
        0: "✅ Normal",
        1: "🚨 Fraud"
    })

    # -------------------------------
    # Metrics
    # -------------------------------
    col1, col2 = st.columns(2)

    col1.metric(
        "✅ Normal Transactions",
        int((preds == 0).sum())
    )

    col2.metric(
        "🚨 Fraudulent Transactions",
        int((preds == 1).sum())
    )

    # -------------------------------
    # Results
    # -------------------------------
    st.write("### Results")

    display_cols = ["Prediction"]

    if "Time" in df.columns:
        display_cols.insert(0, "Time")

    if "Amount" in df.columns:
        display_cols.insert(1, "Amount")

    st.dataframe(df[display_cols].head(50))