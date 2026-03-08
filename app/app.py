import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    
    st.info("Training model for the first time...")
    df = pd.read_csv('data/creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time']   = scaler.fit_transform(X[['Time']])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model

model = load_or_train_model()

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detector")
st.markdown("Upload a CSV file with transaction data to detect fraud.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time']   = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1) if 'Class' in df.columns else df

    preds = model.predict(X)
    df['Prediction'] = pd.Series(preds).map({0: '✅ Normal', 1: '🚨 Fraud'})

    col1, col2 = st.columns(2)
    col1.metric("✅ Normal Transactions",      (preds == 0).sum())
    col2.metric("🚨 Fraudulent Transactions",  (preds == 1).sum())

    st.write("### Results", df[['Time', 'Amount', 'Prediction']].head(50))