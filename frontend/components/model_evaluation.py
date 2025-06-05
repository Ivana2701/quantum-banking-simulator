import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

API_URL = "http://localhost:8000"

def display_metrics(metrics):
    st.write("### Metrics")
    st.write(f"Accuracy: {metrics['accuracy']}")
    st.write(f"Precision: {metrics['precision']}")
    st.write(f"Recall: {metrics['recall']}")
    st.write(f"F1-score: {metrics['f1_score']}")

    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    roc = metrics['roc_curve']
    fig2, ax2 = plt.subplots()
    ax2.plot(roc['fpr'], roc['tpr'], label=f"AUC={roc['auc']:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc='best')
    ax2.set_title('ROC Curve')
    st.pyplot(fig2)

def show_model_evaluation():
    algorithm = st.selectbox(
        "Choose Algorithm",
        ["quantum_vqc", "quantum_zfeaturemap", "quantum_qsvm", "quantum_qnn"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Fetch Results"):
            script_dir = os.path.dirname(os.path.realpath(__file__))
            metrics_path = os.path.join(script_dir, "..", "..", "backend", "fraud_detection", "metrics_data", f"{algorithm}.json")

            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    display_metrics(metrics)
            else:
                st.error("Results not found.")

    with col2:
        if st.button("Run Quantum Model"):
            with st.spinner("Running quantum model..."):
                resp = requests.post(f"{API_URL}/run_model/{algorithm}")
                if resp.status_code == 200:
                    st.success("Model run complete!")
                    script_dir = os.path.dirname(os.path.realpath(__file__))
                    metrics_path = os.path.join(script_dir, "..", "..", "backend", "fraud_detection", "metrics_data", f"{algorithm}.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                        display_metrics(metrics)
                    else:
                        st.error("Metrics file not found after running.")
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
