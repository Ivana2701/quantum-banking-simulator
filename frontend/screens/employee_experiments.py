import os
import streamlit as st
import pandas as pd
from backend.fraud_detection.model_experiments import run_model_on_transaction, compare_all_models, get_qubit_comparison
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
csv_path = os.path.join(project_root, 'creditcard.csv')

class SingleModelTab:
    """
    Tab for running a selected ML model on a chosen Kaggle transaction.
    Employee selects model and transaction, runs prediction, and sees result.
    """
    def render(self):
        st.header("Run Single Model on Kaggle Transaction")
        model_name = st.selectbox("Select Model", [
            "random_forest", "svm", "logistic", "qsvm", "vqc", "quantum_100"
        ])
        # Load a sample of the Kaggle dataset
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Could not load creditcard.csv: {e}")
            return
        idx = st.number_input(
            "Select Transaction Index",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
            value=0
        )
        idx = int(idx)
        features = df.drop(["Time", "Class"], axis=1).iloc[idx].values
        actual_label = df.iloc[idx]["Class"]
        if st.button("Run Model"):
            with st.spinner("Running model..."):
                result = run_model_on_transaction(model_name, features)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write("Prediction:", "❌ Fraudulent" if result["prediction"] == 1 else "✅ Legitimate")
                if result["probability"] is not None:
                    st.write("Probability:", f"{result['probability']:.2%}")
                st.write(f"**Actual label:** {'Fraud' if actual_label == 1 else 'Not fraud'}")

class CompareModelsTab:
    """
    Tab for showing model performance comparison image (for fast UI testing), with a button to simulate running the comparison script and a selector for different graphics.
    """
    def render(self):
        import os
        import streamlit as st
        from PIL import Image
        st.header("Compare All Models")
        st.subheader("Model Performance Comparison")
        image_options = {
            "Quantum Only": "qsvm_vqc_performance_comparison.png",
            "100-Qubit Comparison": "modelcomp100qubit.png"
        }
        selected_image = st.selectbox("Select Comparison Graphic", list(image_options.keys()))
        img1_path = os.path.join("backend", "fraud_detection", "visualizations", image_options[selected_image])
        if st.button("Run All Models"):
            with st.spinner("Running all models and generating comparison..."):
                import time
                time.sleep(1.5)  # Simulate computation delay
            if os.path.exists(img1_path):
                st.image(Image.open(img1_path), caption=selected_image, use_container_width=True)
            else:
                st.warning("Selected image not found.")
        else:
            st.info("Click 'Run All Models' to generate and view the model performance comparison.")

class QubitComparisonTab:
    """
    Tab for showing quantum 100 qubit comparison image (for fast UI testing).
    """
    def render(self):
        import os
        import streamlit as st
        from PIL import Image
        st.header("6-Qubit vs 100-Qubit System Comparison")
        img2_path = os.path.join("backend", "fraud_detection", "visualizations", "quantum_100_performance_comparison.png")
        if os.path.exists(img2_path):
            st.image(Image.open(img2_path), caption="Quantum 100 Qubit Model Comparison", use_container_width=True)
        else:
            st.warning("Quantum 100 comparison image not found.")

def main():
    st.title("🧑‍💼 Employee ML Experiments")
    tab1, tab2, tab3 = st.tabs([
        "Single Model Prediction",
        "Compare All Models",
        "6-Qubit vs 100-Qubit"
    ])
    with tab1:
        SingleModelTab().render()
    with tab2:
        CompareModelsTab().render()
    with tab3:
        QubitComparisonTab().render()

if __name__ == "__main__":
    main()