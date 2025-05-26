import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

# Load environment variables
load_dotenv()

# Database connection
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Fetch data from database
df = pd.read_sql('SELECT amount, EXTRACT(hour FROM transaction_time) as hour, location, is_fraud FROM transactions_qml', engine)
df['location'] = pd.factorize(df['location'])[0]

# Features and labels
X = df[['amount', 'hour', 'location']]
y = df['is_fraud'].astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Balance training data with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Reduce dimensionality for quantum processing
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test)

# Quantum Circuit configuration
feature_dim = 2
feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=3, paulis=['Z', 'X', 'ZZ'])
ansatz = RealAmplitudes(num_qubits=feature_dim, reps=3)

# Quantum backend, optimizer, and sampler
backend = AerSimulator()
optimizer = COBYLA(maxiter=250)
sampler = AerSampler()

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

# Initialize and train the VQC model
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)
vqc.fit(X_train_pca, y_train_balanced.to_numpy())

# Predict and evaluate Quantum VQC model
y_pred_vqc = vqc.predict(X_test_pca)
print("\n✅ Quantum VQC (PCA + SMOTE) Results:\n", 
      classification_report(y_test, y_pred_vqc, target_names=["Legitimate", "Fraudulent"]))

# Classical SVM comparison (trained on the same PCA-transformed data)
svc = SVC(kernel='rbf', probability=True, random_state=42)
svc.fit(X_train_pca, y_train_balanced)
y_pred_classical = svc.predict(X_test_pca)

print("\n✅ Classical SVM (PCA + SMOTE) Results:\n",
      classification_report(y_test, y_pred_classical, target_names=["Legitimate", "Fraudulent"]))

# Visualize Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_vqc), annot=True, fmt='d', cmap='Purples', ax=axes[0])
axes[0].set_title('Quantum VQC (PCA + SMOTE)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_classical), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Classical SVM (PCA + SMOTE)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC Curve comparison
plt.figure(figsize=(8, 6))

# Quantum VQC ROC
y_score_vqc = vqc.predict_proba(X_test_pca)[:, 1]
fpr_q, tpr_q, _ = roc_curve(y_test, y_score_vqc)
roc_auc_q = auc(fpr_q, tpr_q)
plt.plot(fpr_q, tpr_q, color='purple', lw=2, label=f'Quantum VQC (AUC = {roc_auc_q:.2f})')

# Classical SVM ROC
y_score_classical = svc.predict_proba(X_test_pca)[:, 1]
fpr_c, tpr_c, _ = roc_curve(y_test, y_score_classical)
roc_auc_c = auc(fpr_c, tpr_c)
plt.plot(fpr_c, tpr_c, color='green', lw=2, label=f'Classical SVM (AUC = {roc_auc_c:.2f})')

# Chance-level ROC
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# ROC plot adjustments
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Quantum VQC vs Classical SVM')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
