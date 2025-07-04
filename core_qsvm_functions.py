"""
Core QSVM Functions - The Essential Building Blocks
==================================================

These are the most important functions extracted from the complex QSVM implementations.
Everything else is just data preprocessing, evaluation, and file handling.
"""

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel

def create_quantum_feature_map(feature_dim, feature_map_type='zz', reps=2, entanglement='linear'):
    """
    Create quantum feature map - encodes classical data into quantum states
    
    Args:
        feature_dim: Number of features (becomes number of qubits)
        feature_map_type: 'zz', 'efficient_su2', 'pauli', 'two_local'
        reps: Number of repetitions in the circuit
        entanglement: How qubits are connected ('linear', 'circular', 'full')
    
    Returns:
        Quantum feature map circuit
    """
    if feature_map_type == 'zz':
        return ZZFeatureMap(
            feature_dimension=feature_dim,
            reps=reps,
            entanglement=entanglement
        )
    elif feature_map_type == 'efficient_su2':
        return EfficientSU2(
            num_qubits=feature_dim,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=True
        )
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")

def create_quantum_kernel(feature_map):
    """
    Create quantum kernel - computes similarity between quantum states
    
    Args:
        feature_map: Quantum feature map circuit
    
    Returns:
        Quantum kernel
    """
    return QuantumKernel(feature_map=feature_map)

def train_quantum_svm(X_train, y_train, feature_dim, feature_map_type='zz', reps=2, entanglement='linear'):
    """
    Train Quantum SVM - the complete training function
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_dim: Number of features
        feature_map_type: Type of quantum feature map
        reps: Number of repetitions
        entanglement: Entanglement pattern
    
    Returns:
        Trained QSVC model
    """
    # Step 1: Create quantum feature map
    feature_map = create_quantum_feature_map(feature_dim, feature_map_type, reps, entanglement)
    
    # Step 2: Create quantum kernel
    quantum_kernel = create_quantum_kernel(feature_map)
    
    # Step 3: Create and train QSVC
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    
    return qsvc

def predict_with_quantum_svm(model, X_test):
    """
    Make predictions with trained quantum SVM
    
    Args:
        model: Trained QSVC model
        X_test: Test features
    
    Returns:
        Predictions and decision scores
    """
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    return predictions, scores

# Example usage:
# qsvm_model = train_quantum_svm(X_train, y_train, feature_dim=4)
# predictions, scores = predict_with_quantum_svm(qsvm_model, X_test) 