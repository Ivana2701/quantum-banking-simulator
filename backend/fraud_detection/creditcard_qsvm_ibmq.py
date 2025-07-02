import os
import json
from qiskit_ibm_provider import IBMProvider
from qiskit.primitives import BackendSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# Load IBMQ account and select backend
provider = IBMProvider()
backend = provider.get_backend('ibmq_qasm_simulator')  # Change to a real backend if desired
sampler = BackendSampler(backend)

np.random.seed(42)

# Data loading and preprocessing (same as creditcard_qsvm.py)
def load_creditcard_data(filepath='../../creditcard.csv', sample_size=None):
    print(f"Loading credit card data from {filepath}...")
    df = pd.read_csv(filepath)
    if sample_size:
        fraud_indices = df[df['Class'] == 1].index
        normal_indices = df[df['Class'] == 0].index
        fraud_sample_size = min(len(fraud_indices), sample_size // 10)
        fraud_sample = np.random.choice(fraud_indices, fraud_sample_size, replace=False)
        normal_sample_size = sample_size - fraud_sample_size
        normal_sample = np.random.choice(normal_indices, normal_sample_size, replace=False)
        sample_indices = np.concatenate([fraud_sample, normal_sample])
        df = df.iloc[sample_indices].reset_index(drop=True)
    X = df.drop(['Time', 'Class'], axis=1).values
    y = np.array(df['Class'].values, dtype=float)
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {np.mean(y)*100:.3f}%")
    return X, y

def preprocess_features(X, y, n_qubits=6, test_size=0.2):
    scaler = RobustScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # PCA to n_qubits
    pca = PCA(n_components=n_qubits, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, y_train, y_test

def train_and_evaluate_ibmq_qsvm(sample_size=1000, n_qubits=6):
    X, y = load_creditcard_data(sample_size=sample_size)
    X_train, X_test, y_train, y_test = preprocess_features(X, y, n_qubits=n_qubits)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=sampler)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    print("Training QSVM on IBMQ backend...")
    qsvc.fit(X_train, y_train)
    print("Training completed!")
    y_pred = qsvc.predict(X_test)
    y_score = qsvc.decision_function(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_score)
    }
    print("Metrics:", metrics)
    # Save metrics
    script_dir = os.path.dirname(os.path.realpath(__file__))
    metrics_path = os.path.join(script_dir, "metrics_data", "creditcard_qsvm_ibmq.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate_ibmq_qsvm(sample_size=1000, n_qubits=6) 