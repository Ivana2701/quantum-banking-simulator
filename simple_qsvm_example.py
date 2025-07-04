"""
Simple Quantum SVM Example
==========================

This is the essence of Quantum Support Vector Machine (QSVM) code,
extracted from the complex fraud detection implementations.

Core Components:
1. Quantum Feature Map (encodes classical data into quantum states)
2. Quantum Kernel (computes similarity between quantum states)
3. QSVC (Quantum Support Vector Classifier)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel

# Set random seed for reproducibility
np.random.seed(42)

def generate_simple_data():
    """Generate simple 2D classification data"""
    X, y = make_classification(
        n_samples=100,      # Small dataset for quick demonstration
        n_features=2,       # 2 features (2 qubits)
        n_informative=2,    # Both features are informative
        n_redundant=0,      # No redundant features
        n_classes=2,        # Binary classification
        random_state=42
    )
    return X, y

def preprocess_data(X, y):
    """Preprocess data: split, scale"""
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features (important for quantum algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def create_quantum_feature_map(feature_dim):
    """
    Create quantum feature map - this is where classical data becomes quantum
    
    Args:
        feature_dim: Number of features (will be number of qubits)
    """
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dim,  # Each feature becomes a qubit
        reps=2,                         # Number of repetitions
        entanglement='linear'           # How qubits are connected
    )
    
    print(f"Created quantum feature map:")
    print(f"  - Qubits: {feature_map.num_qubits}")
    print(f"  - Repetitions: {2}")
    print(f"  - Entanglement: linear")
    
    return feature_map

def train_quantum_svm(X_train, y_train):
    """
    Train Quantum SVM - the core of QSVM
    
    Args:
        X_train: Training features
        y_train: Training labels
    """
    # Step 1: Create quantum feature map
    feature_map = create_quantum_feature_map(X_train.shape[1])
    
    # Step 2: Create quantum kernel
    quantum_kernel = QuantumKernel(feature_map=feature_map)
    
    # Step 3: Create and train QSVC
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    
    print("Training Quantum SVM...")
    qsvc.fit(X_train, y_train)
    print("Training completed!")
    
    return qsvc

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    
    return accuracy

def main():
    """Main function - complete QSVM pipeline"""
    print("🚀 Simple Quantum SVM Example")
    print("=" * 40)
    
    # Step 1: Generate data
    print("1. Generating data...")
    X, y = generate_simple_data()
    print(f"   Dataset shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Step 3: Train quantum SVM
    print("\n3. Training Quantum SVM...")
    qsvm_model = train_quantum_svm(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\n4. Evaluating model...")
    accuracy = evaluate_model(qsvm_model, X_test, y_test)
    
    print(f"\n✅ QSVM training and evaluation completed!")
    print(f"Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 