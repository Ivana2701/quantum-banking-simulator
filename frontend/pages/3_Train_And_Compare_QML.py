import streamlit as st
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Train & Compare QML", layout="wide")
st.title("🧪 Train & Compare QSVM / VQC Models")

# --- Training Buttons ---
st.subheader("🎓 Train Quantum Models")

col1, col2 = st.columns(2)
with col1:
    if st.button("Train QSVM"):
        result = subprocess.run(
            ["python3", "backend/fraud_detection/train_and_save_models.py"],
            capture_output=True, text=True
        )
        st.success("✅ QSVM model trained!")
        st.code(result.stdout)

with col2:
    if st.button("Train VQC"):
        result = subprocess.run(
            ["python3", "backend/fraud_detection/vqc_fraud_detection.py"],
            capture_output=True, text=True
        )
        st.success("✅ VQC model trained and metrics saved!")
        st.code(result.stdout)

# --- Milestone CSV Metrics View ---
st.markdown("---")
st.subheader("📊 Training Metrics from Milestones")

# Load milestone training output
milestone_path = "backend/quantum_ml/milestone_metrics.csv"
if os.path.exists(milestone_path):
    df = pd.read_csv(milestone_path)
    tabs = st.tabs(["📈 F1 Comparison", "📉 AUC Comparison", "📋 Raw Metrics"])
    
    with tabs[0]:
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=df, x="reps", y="vqc_f1", marker="o", label="VQC", ax=ax1)
        sns.lineplot(data=df, x="reps", y="svm_f1", marker="o", label="Classical SVM", ax=ax1)
        ax1.set_title("F1-Score vs Circuit Depth (Reps)")
        st.pyplot(fig1)

    with tabs[1]:
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=df, x="reps", y="vqc_auc", marker="o", label="VQC", ax=ax2)
        sns.lineplot(data=df, x="reps", y="svm_auc", marker="o", label="Classical SVM", ax=ax2)
        ax2.set_title("AUC Score vs Circuit Depth (Reps)")
        st.pyplot(fig2)

    with tabs[2]:
        st.dataframe(df)
else:
    st.warning("Milestone metrics not found. Please run `milestone.py` to generate.")

# --- Educational Block ---
st.markdown("---")
st.subheader("📘 Quantum Classifier Math & Theory")

st.markdown("### QSVM: Quantum Support Vector Machine")
st.latex(r\"\"\"K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2\"\"\")
st.markdown(\"\"\"QSVM computes a **quantum kernel** using inner products of qubit states \n
and uses a classical SVM to separate them. The mapping φ is done by a **feature map** quantum circuit.\"\"\")

st.markdown("### VQC: Variational Quantum Classifier")
st.latex(r\"\"\"|\psi(\vec{x}, \vec{\theta})\rangle = U(\vec{\theta}) \cdot \Phi(\vec{x}) |0\rangle\"\"\")
st.markdown(\"\"\"VQC applies a **parameterized ansatz** after encoding the input with a **feature map**.
It is trained via a hybrid classical optimizer like COBYLA.\"\"\")

# Optional link to backend
st.markdown(\"---\")\nst.markdown(\"For internal use. All training metrics are derived from `milestone.py`. Run it manually if needed:\")\n\nst.code(\"python3 backend/quantum_ml/milestone.py\")
