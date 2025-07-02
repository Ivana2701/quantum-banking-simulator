import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import json
from frontend.screens import employee_experiments

st.title("🧑‍💼 Employee Dashboard: Fraud Detection")
tabs = st.tabs(["Kaggle QML Experiments", "Real Transaction Detection", "ML Experiments"])

with tabs[0]:
    st.header("Kaggle Data: Quantum vs Classical")
    st.markdown("""
    Explore the power of quantum machine learning (QML) for fraud detection using the Kaggle credit card dataset.\n
    Compare classical and quantum models, visualize their performance, and see how QML could revolutionize fraud detection.
    """)

    # Visualize model comparison
    st.subheader("Model Performance Comparison")
    img1_path = os.path.join("backend", "fraud_detection", "visualizations", "qsvm_vqc_performance_comparison.png")
    if os.path.exists(img1_path):
        st.image(Image.open(img1_path), caption="Classical vs Quantum Model Performance", use_column_width=True)
    else:
        st.warning("Performance comparison image not found.")

    # Visualize quantum 100 comparison
    img2_path = os.path.join("backend", "fraud_detection", "visualizations", "quantum_100_performance_comparison.png")
    if os.path.exists(img2_path):
        st.image(Image.open(img2_path), caption="Quantum 100 Qubit Model Comparison", use_column_width=True)
    else:
        st.warning("Quantum 100 comparison image not found.")

    # Interactive prediction demo
    st.markdown("---")
    st.subheader("🔮 Interactive Fraud Prediction Demo (Kaggle Data)")
    @st.cache_data
    def load_kaggle_sample():
        df = pd.read_csv("creditcard.csv")
        return df.sample(1000, random_state=42).reset_index(drop=True)
    df = load_kaggle_sample()
    idx = st.selectbox("Select a transaction index", df.index)
    row = df.loc[idx]
    features = row.drop(["Class", "Time"])
    st.write("Transaction details:", features)

    # Load classical and quantum models/scalers if available
    classical_model_path = os.path.join("backend", "models", "classical_svm.pkl")
    quantum_model_path = os.path.join("backend", "models", "quantum_qsvm.pkl")
    scaler_path = os.path.join("backend", "models", "scaler.pkl")
    classical_pred, quantum_pred, classical_proba, quantum_proba = None, None, None, None
    if os.path.exists(classical_model_path) and os.path.exists(scaler_path):
        classical_model = joblib.load(classical_model_path)
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform([features])
        classical_pred = classical_model.predict(X_scaled)[0]
        if hasattr(classical_model, "predict_proba"):
            classical_proba = classical_model.predict_proba(X_scaled)[0][1]
    if os.path.exists(quantum_model_path) and os.path.exists(scaler_path):
        quantum_model = joblib.load(quantum_model_path)
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform([features])
        quantum_pred = quantum_model.predict(X_scaled)[0]
        if hasattr(quantum_model, "predict_proba"):
            quantum_proba = quantum_model.predict_proba(X_scaled)[0][1]
    st.markdown("### Predictions:")
    if classical_pred is not None:
        st.write(f"Classical Model: {'❌ Fraudulent' if classical_pred == 1 else '✅ Legitimate'}" + (f" (prob: {classical_proba:.2%})" if classical_proba is not None else ""))
    else:
        st.info("Classical model not available.")
    if quantum_pred is not None:
        st.write(f"Quantum Model: {'❌ Fraudulent' if quantum_pred == 1 else '✅ Legitimate'}" + (f" (prob: {quantum_proba:.2%})" if quantum_proba is not None else ""))
    else:
        st.info("Quantum model not available.")
    st.write(f"**Actual label:** {'Fraud' if row['Class'] == 1 else 'Not fraud'}")
    with st.expander("Show raw features and model details"):
        st.json(features.to_dict())

    # ROC and Confusion Matrix Visualizations
    st.markdown("---")
    st.subheader("Model ROC Curves and Confusion Matrices")
    metrics_dir = os.path.join("backend", "fraud_detection", "metrics_data")
    classical_metrics_path = os.path.join(metrics_dir, "creditcard_random_forest.json")
    quantum_metrics_path = os.path.join(metrics_dir, "creditcard_qsvm.json")
    # ROC Curve
    fig, ax = plt.subplots()
    legend_labels = []
    if os.path.exists(classical_metrics_path):
        with open(classical_metrics_path) as f:
            classical_metrics = json.load(f)
        roc = classical_metrics["roc_curve"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"Classical (AUC={roc['auc']:.2f})")
        legend_labels.append("Classical")
    if os.path.exists(quantum_metrics_path):
        with open(quantum_metrics_path) as f:
            quantum_metrics = json.load(f)
        roc = quantum_metrics["roc_curve"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"Quantum (AUC={roc['auc']:.2f})")
        legend_labels.append("Quantum")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    st.pyplot(fig)
    # Confusion Matrices
    cols = st.columns(2)
    if os.path.exists(classical_metrics_path):
        with open(classical_metrics_path) as f:
            classical_metrics = json.load(f)
        cm = classical_metrics["confusion_matrix"]
        with cols[0]:
            st.markdown("**Classical Model Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
    if os.path.exists(quantum_metrics_path):
        with open(quantum_metrics_path) as f:
            quantum_metrics = json.load(f)
        cm = quantum_metrics["confusion_matrix"]
        with cols[1]:
            st.markdown("**Quantum Model Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

with tabs[1]:
    st.header("Real Banking Transactions (Prototype)")
    st.markdown("""
    This section demonstrates how fraud detection could work on real banking transactions.\n
    As more labeled data is collected, quantum models can be trained and deployed for live fraud detection.
    """)
    st.info("Prototype: Real-time fraud detection will be available once enough labeled data is collected.")

with tabs[2]:
    employee_experiments.main() 