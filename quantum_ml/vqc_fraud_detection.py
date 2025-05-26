#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score
)
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.utils import algorithm_globals
from qiskit.algorithms.optimizers import COBYLA

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load config & fetch data
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql(
    "SELECT amount, EXTRACT(hour FROM transaction_time) AS hour, "
    "location, is_fraud FROM transactions_qml",
    engine
)
df['location'] = pd.factorize(df['location'])[0]

# **Immediately convert labels and features to NumPy arrays**
X = df[['amount', 'hour', 'location']].values
y = df['is_fraud'].astype(int).values

# ─────────────────────────────────────────────────────────────────────────────
# 2) Preprocess: scale, split, SMOTE, PCA
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
# ensure NumPy
y_train_bal = np.array(y_train_bal)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca  = pca.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Configure hybrid VQC
# ─────────────────────────────────────────────────────────────────────────────
algorithm_globals.random_seed = 42

feature_map = PauliFeatureMap(feature_dimension=2, reps=3, paulis=['Z','X','ZZ'])
ansatz      = RealAmplitudes(num_qubits=2, reps=3)

backend   = AerSimulator()
optimizer = COBYLA(maxiter=250)
sampler   = AerSampler()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Train VQC & classical SVM
# ─────────────────────────────────────────────────────────────────────────────
vqc.fit(X_train_pca, y_train_bal)
svc = SVC(kernel='rbf', probability=True, random_state=42)
svc.fit(X_train_pca, y_train_bal)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Evaluate on hold‐out set
# ─────────────────────────────────────────────────────────────────────────────
y_pred_vqc  = vqc.predict(X_test_pca)
y_pred_svm  = svc.predict(X_test_pca)
y_score_svm = svc.predict_proba(X_test_pca)[:,1]

print("\n✅ Quantum VQC (PCA + SMOTE) Results:\n",
      classification_report(y_test, y_pred_vqc,
                            target_names=["Legitimate","Fraudulent"]))
print("\n✅ Classical SVM (PCA + SMOTE) Results:\n",
      classification_report(y_test, y_pred_svm,
                            target_names=["Legitimate","Fraudulent"]))

# confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_vqc), annot=True, fmt='d',
            cmap='Purples', ax=axes[0])
axes[0].set(title='Quantum VQC', xlabel='Pred', ylabel='Actual')
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d',
            cmap='Greens', ax=axes[1])
axes[1].set(title='Classical SVM', xlabel='Pred', ylabel='Actual')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6) ROC curves (only classical + attempt quantum)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
fpr_s, tpr_s, _ = roc_curve(y_test, y_score_svm)
plt.plot(fpr_s, tpr_s, label=f'SVM AUC={auc(fpr_s,tpr_s):.2f}', color='green')

try:
    # if your VQC ever implements predict_proba, this will work
    y_score_vqc = vqc.predict_proba(X_test_pca)[:,1]
    fpr_v, tpr_v, _ = roc_curve(y_test, y_score_vqc)
    plt.plot(fpr_v, tpr_v, label=f'VQC AUC={auc(fpr_v,tpr_v):.2f}', color='purple')
except AttributeError:
    print("⚠️  VQC.predict_proba() not available – skipping quantum ROC curve.")

plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(loc='lower right')
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7) 5‐Fold CV + paired t‐test on Acc & AUC
# ─────────────────────────────────────────────────────────────────────────────
vqc_accs, svm_accs = [], []
vqc_aucs, svm_aucs = [], []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== 5-Fold CV ===")
for i, (tr, te) in enumerate(kf.split(X_scaled, y), 1):
    Xtr, Xte = X_scaled[tr], X_scaled[te]
    ytr, yte = y[tr],      y[te]

    Xtr_bal, ytr_bal = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    ytr_bal = np.array(ytr_bal)

    pca_cv = PCA(n_components=2)
    Xtr_pca = pca_cv.fit_transform(Xtr_bal)
    Xte_pca = pca_cv.transform(Xte)

    algorithm_globals.random_seed = 42
    vqc_cv = VQC(feature_map, ansatz, optimizer, sampler)
    vqc_cv.fit(Xtr_pca, ytr_bal)
    ypv = vqc_cv.predict(Xte_pca)
    try:
        ysv = vqc_cv.predict_proba(Xte_pca)[:,1]
    except AttributeError:
        ysv = (ypv == 1).astype(float)  # fallback: 0/1

    svc_cv = SVC(kernel='rbf', probability=True, random_state=42)
    svc_cv.fit(Xtr_pca, ytr_bal)
    yps = svc_cv.predict(Xte_pca)
    yss = svc_cv.predict_proba(Xte_pca)[:,1]

    acc_v = (ypv == yte).mean()
    acc_s = (yps == yte).mean()
    auc_v = auc(*roc_curve(yte, ysv)[:2])
    auc_s = auc(*roc_curve(yte, yss)[:2])

    vqc_accs.append(acc_v); svm_accs.append(acc_s)
    vqc_aucs.append(auc_v); svm_aucs.append(auc_s)

    print(f" Fold {i}: VQC acc={acc_v:.2f}, SVM acc={acc_s:.2f}, "
          f"VQC AUC={auc_v:.2f}, SVM AUC={auc_s:.2f}")

print("\n=== CV Summary ===")
print(f"VQC Acc:  {np.mean(vqc_accs):.3f} ± {np.std(vqc_accs):.3f}")
print(f"SVM Acc:  {np.mean(svm_accs):.3f} ± {np.std(svm_accs):.3f}")
print(f"VQC AUC:  {np.mean(vqc_aucs):.3f} ± {np.std(vqc_aucs):.3f}")
print(f"SVM AUC:  {np.mean(svm_aucs):.3f} ± {np.std(svm_aucs):.3f}")

t_acc, p_acc = ttest_rel(vqc_accs, svm_accs)
t_auc, p_auc = ttest_rel(vqc_aucs, svm_aucs)
print(f"\nPaired t-test Acc: t={t_acc:.2f}, p={p_acc:.3f}")
print(f"Paired t-test AUC: t={t_auc:.2f}, p={p_auc:.3f}")
