# backend/fraud_detection/train_and_save_models.py
import os
from joblib import dump
from qsvm_fraud_detection import train_quantum_svm, generate_and_prepare_data

def train_and_save_qsvm(model_save_path):
    # Generate and prepare data
    X_train, _, y_train, _ = generate_and_prepare_data()

    # Train QSVM model
    qsvm_model = train_quantum_svm(X_train, y_train)

    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save trained model
    dump(qsvm_model, model_save_path)

    print(f"QSVM model trained and saved to {model_save_path}")

if __name__ == "__main__":
    MODEL_PATH = "backend/metrics_data/quantum_qsvm.pkl"
    train_and_save_qsvm(MODEL_PATH)

