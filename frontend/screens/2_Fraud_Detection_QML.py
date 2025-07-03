import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load as joblib_load

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Fraud Detection QML", layout="wide")
st.title("Fraud Detection: Quantum vs Classical Models")

# --- Helper functions ---
@st.cache_data
def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def get_transaction_data():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    df = pd.read_sql(
        "SELECT transaction_id, amount, EXTRACT(hour FROM transaction_time) AS hour, location FROM transactions_qml",
        conn
    )
    df['location'] = pd.factorize(df['location'])[0]
    conn.close()
    return df

def preprocess_transaction(df, tx_id):
    row = df[df['transaction_id'] == tx_id]
    features = row[['amount', 'hour', 'location']].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    return pca.fit_transform(features_scaled)

def load_model_and_predict(tx_vector, model_path):
    model = joblib_load(model_path)
    return int(model.predict(tx_vector)[0])

# --- UI: ROC Comparison ---
st.subheader("ROC Curve Comparison")

try:
    qsvm = load_metrics("backend/metrics_data/quantum_qsvm.json")
    vqc = load_metrics("backend/metrics_data/quantum_vqc.json")

    fig, ax = plt.subplots()
    ax.plot(qsvm["roc_curve"]["fpr"], qsvm["roc_curve"]["tpr"], label=f"QSVM (AUC = {qsvm['roc_curve']['auc']:.2f})")
    ax.plot(vqc["roc_curve"]["fpr"], vqc["roc_curve"]["tpr"], label=f"VQC (AUC = {vqc['roc_curve']['auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Quantum Model ROC Curves")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning("Could not load both ROC curves. Check metrics files.")
    st.text(str(e))

# --- UI: Live Prediction Demo ---
st.markdown("---")
st.subheader("Live Fraud Prediction")

try:
    df = get_transaction_data()
    tx_id = st.selectbox("Select a Transaction ID", df["transaction_id"].unique())
    tx_vector = preprocess_transaction(df, tx_id)

    model_type = st.radio("Which model to use?", ["QSVM", "VQC"], horizontal=True)
    model_path = "backend/metrics_data/quantum_qsvm.pkl" if model_type == "QSVM" else "backend/metrics_data/quantum_vqc.pkl"

    prediction = load_model_and_predict(tx_vector, model_path)

    st.markdown(f"### 💳 Transaction {tx_id} is classified as:")
    if prediction == 1:
        st.error("Fraudulent")
    else:
        st.success("Legitimate")

except Exception as e:
    st.warning("Live prediction unavailable.")
    st.text(str(e))
