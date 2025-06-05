import streamlit as st
import requests
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

API_URL = "http://localhost:8000"

algorithm = st.selectbox(
    "Choose Algorithm",
    ["quantum_vqc", "quantum_qsvm", "quantum_qnn", "quantum_zfeaturemap"]
)

def display_metrics(metrics):
    st.write("### Metrics")
    st.write(f"**Accuracy**: {metrics['accuracy']:.2f}")
    st.write(f"**Precision**: {metrics['precision']:.2f}")
    st.write(f"**Recall**: {metrics['recall']:.2f}")
    st.write(f"**F1-score**: {metrics['f1_score']:.2f}")

    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    roc = metrics['roc_curve']
    fig2, ax2 = plt.subplots()
    ax2.plot(roc['fpr'], roc['tpr'], label=f"AUC={roc['auc']:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    st.pyplot(fig2)

    st.write("#### Raw Metrics JSON")
    st.json(metrics)

col1, col2 = st.columns(2)

with col1:
    if st.button("Fetch Results"):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print("script_dir:", script_dir)
        metrics_path = os.path.join(script_dir, "..", "..", "backend", "fraud_detection", "metrics_data", f"{algorithm}.json")
        print(metrics_path)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
                print(metrics)
            display_metrics(metrics)
        else:
            st.error("Results not found.")

with col2:
    if st.button("Run Quantum Model"):
        with st.spinner("Running quantum model..."):
            resp = requests.post(f"{API_URL}/run_model/{algorithm}")
            if resp.status_code == 200:
                st.success("Model run complete!")
                metrics_path = f"backend/fraud_detection/metrics_data/{algorithm}.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    display_metrics(metrics)
                else:
                    st.error("Metrics file not found after running.")
            else:
                st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
