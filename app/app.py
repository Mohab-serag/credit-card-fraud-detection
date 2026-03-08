import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detector")
st.markdown("Upload a CSV file with transaction data to detect fraud.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time']   = scaler.fit_transform(df[['Time']])

    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
    else:
        X = df

    preds = model.predict(X)
    df['Prediction'] = preds
    df['Prediction'] = df['Prediction'].map({0: '✅ Normal', 1: '🚨 Fraud'})

    fraud_count  = (preds == 1).sum()
    normal_count = (preds == 0).sum()

    col1, col2 = st.columns(2)
    col1.metric("✅ Normal Transactions", normal_count)
    col2.metric("🚨 Fraudulent Transactions", fraud_count)

    st.write("### Results", df[['Time', 'Amount', 'Prediction']].head(50))