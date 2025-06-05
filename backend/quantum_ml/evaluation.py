# quantum_ml/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

def holdout_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = (model.predict_proba(X_test)[:,1]
               if hasattr(model, 'predict_proba')
               else None)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1'      : f1_score(y_test, y_pred),
        'auc'     : (roc_auc_score(y_test, y_score) if y_score is not None else np.nan)
    }

def cv_compare(make_model_fn, X, y, n_splits=5, **make_kwargs):
    """Return per‐fold acc & auc for two models: vqc vs svc."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    vqc_accs, svc_accs = [], []
    vqc_aucs, svc_aucs = [], []

    for train_idx, test_idx in kf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # make balanced & PCA inside each fold if desired...
        model_vqc = make_model_fn(reps=make_kwargs['reps'], paulis=make_kwargs['paulis'])
        model_svc = make_model_fn is None and None  # skip

        # fit & score VQC:
        model_vqc.fit(Xtr, ytr)
        pv = model_vqc.predict(Xte)
        sv = (model_vqc.predict_proba(Xte)[:,1]
              if hasattr(model_vqc, 'predict_proba') else pv)

        # fit & score SVC
        from sklearn.svm import SVC
        model_svc = SVC(kernel='rbf', probability=True, random_state=42)
        model_svc.fit(Xtr, ytr)
        ps = model_svc.predict(Xte)
        ss = model_svc.predict_proba(Xte)[:,1]

        vqc_accs.append(accuracy_score(yte, pv))
        svc_accs.append(accuracy_score(yte, ps))
        vqc_aucs.append(roc_auc_score(yte, sv))
        svc_aucs.append(roc_auc_score(yte, ss))

    return {
        'vqc_acc': np.array(vqc_accs),
        'svc_acc': np.array(svc_accs),
        'vqc_auc': np.array(vqc_aucs),
        'svc_auc': np.array(svc_aucs),
    }
