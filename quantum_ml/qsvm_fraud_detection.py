from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate balanced synthetic data clearly
X, y = make_classification(n_samples=2000,
                           n_features=4,
                           n_informative=3,
                           n_redundant=1,
                           n_classes=2,
                           weights=[0.5, 0.5],
                           flip_y=0.02,
                           class_sep=0.9,
                           random_state=42)

# Split dataset practically and clearly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data practically
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quantum SVM clearly
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2, entanglement='linear')
backend = Aer.get_backend('statevector_simulator')
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)
y_pred_quantum = qsvc.predict(X_test)

# Classical SVM clearly
svc = SVC(kernel='rbf', probability=True, random_state=42)
svc.fit(X_train, y_train)
y_pred_classical = svc.predict(X_test)

# Confusion Matrices practically and clearly
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_quantum), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Quantum SVM Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_classical), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Classical SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC Curves clearly
plt.figure(figsize=(8, 6))

# Quantum ROC practically
y_score_quantum = qsvc.decision_function(X_test)
fpr_q, tpr_q, _ = roc_curve(y_test, y_score_quantum)
roc_auc_q = auc(fpr_q, tpr_q)
plt.plot(fpr_q, tpr_q, label=f'Quantum SVM (AUC = {roc_auc_q:.2f})')

# Classical ROC practically
y_score_classical = svc.decision_function(X_test)
fpr_c, tpr_c, _ = roc_curve(y_test, y_score_classical)
roc_auc_c = auc(fpr_c, tpr_c)
plt.plot(fpr_c, tpr_c, label=f'Classical SVM (AUC = {roc_auc_c:.2f})')

# Diagonal for reference practically
plt.plot([0, 1], [0, 1], 'k--')

# Set titles practically and clearly
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')

plt.show()

# Classification reports practically and clearly
print("✅ Quantum SVM Results:")
print(classification_report(y_test, y_pred_quantum, target_names=["Legitimate", "Fraudulent"]))

print("\n✅ Classical SVM Results:")
print(classification_report(y_test, y_pred_classical, target_names=["Legitimate", "Fraudulent"]))
