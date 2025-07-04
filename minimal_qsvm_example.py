"""
Minimal Quantum SVM Example
===========================

The absolute essence of QSVM in just 3 core lines:

1. Create quantum feature map
2. Create quantum kernel  
3. Train QSVC

This is what all the complex fraud detection code boils down to.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel

# Generate simple data
X, y = make_classification(n_samples=50, n_features=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🚀 Minimal QSVM - The Core Essence")
print("=" * 40)

# THE 3 ESSENTIAL LINES OF QSVM:
print("\n1️⃣ Create quantum feature map:")
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
print(f"   ✓ Feature map created with {feature_map.num_qubits} qubits")

print("\n2️⃣ Create quantum kernel:")
quantum_kernel = QuantumKernel(feature_map=feature_map)
print("   ✓ Quantum kernel created")

print("\n3️⃣ Train QSVC:")
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train_scaled, y_train)
print("   ✓ QSVC trained!")

# Test the model
accuracy = qsvc.score(X_test_scaled, y_test)
print(f"\n✅ Final accuracy: {accuracy:.4f}")

print("\n🎯 That's it! This is the essence of Quantum SVM.")
print("   Everything else in the complex files is just:")
print("   - Data preprocessing")
print("   - Feature engineering") 
print("   - Model evaluation")
print("   - Saving results") 