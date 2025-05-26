import psycopg2
import os
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import COBYLA

from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
df = pd.read_sql('SELECT amount, EXTRACT(hour FROM transaction_time) as hour, location, is_fraud FROM transactions_qml', engine)


# Encode location practically using numeric encoding
df['location'] = pd.factorize(df['location'])[0]

# Features and labels
X = df[['amount', 'hour', 'location']]
y = df['is_fraud'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data practically
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Ensure reproducibility
algorithm_globals.random_seed = 42

# Setup Quantum Feature Map and Ansatz
feature_dim = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
ansatz = RealAmplitudes(feature_dim, reps=2)

# Backend and optimizer
backend = AerSimulator()
optimizer = COBYLA(maxiter=100)

# Create the Hybrid Quantum-Classical VQC model
sampler = AerSampler()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

# Fit the VQC model (explicitly converting y_train to numpy array)
vqc.fit(X_train, y_train.to_numpy())

# Predict and evaluate
y_pred_vqc = vqc.predict(X_test)

print("\n✅ Hybrid Quantum-Classical VQC Results:")
print(classification_report(y_test, y_pred_vqc, target_names=["Legitimate", "Fraudulent"]))

# Classical SVM comparison
from sklearn.svm import SVC

svc = SVC(kernel='rbf', probability=True, random_state=42)
svc.fit(X_train, y_train)
y_pred_classical = svc.predict(X_test)

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_vqc), annot=True, fmt='d', cmap='Purples', ax=axes[0])
axes[0].set_title('Hybrid Quantum-Classical VQC')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_classical), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Classical SVM')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC Curve comparison
plt.figure(figsize=(8, 6))

# Quantum-Classical VQC ROC
y_score_vqc = vqc.predict_proba(X_test)[:, 1]
fpr_q, tpr_q, _ = roc_curve(y_test, y_score_vqc)
roc_auc_q = auc(fpr_q, tpr_q)
plt.plot(fpr_q, tpr_q, label=f'Hybrid Quantum VQC (AUC = {roc_auc_q:.2f})')

# Classical ROC
y_score_classical = svc.predict_proba(X_test)[:, 1]
fpr_c, tpr_c, _ = roc_curve(y_test, y_score_classical)
roc_auc_c = auc(fpr_c, tpr_c)
plt.plot(fpr_c, tpr_c, label=f'Classical SVM (AUC = {roc_auc_c:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Hybrid Quantum vs Classical')
plt.legend(loc='lower right')

plt.show()
