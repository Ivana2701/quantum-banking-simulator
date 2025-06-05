#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

from data import load_and_preprocess
from models import make_vqc, make_svm
from evaluation import holdout_metrics

# 1) load & preprocess
Xtr_pca, Xte_pca, ytr_bal, yte = load_and_preprocess()

# 2) grid of hyper-configs
grid = [
    {"reps":1, "paulis":['Z','X']},
    {"reps":2, "paulis":['Z','X','ZZ']},
    {"reps":3, "paulis":['Z','Y','X','ZZ']},
]

results = []
for cfg in grid:
    # build models
    vqc = make_vqc(
        reps=cfg["reps"],
        paulis=cfg["paulis"],
        feature_dim=Xtr_pca.shape[1],
        backend_name='ibm_brisbane'
    )
    svm = make_svm()

    # train models
    vqc.fit(Xtr_pca, ytr_bal)
    svm.fit(Xtr_pca, ytr_bal)

    # evaluate on the same hold-out
    metrics_vqc = holdout_metrics(vqc, Xtr_pca, ytr_bal, Xte_pca, yte)
    metrics_svm = holdout_metrics(svm, Xtr_pca, ytr_bal, Xte_pca, yte)

    # merge into one row
    row = {
      **cfg,
      **{f"vqc_{k}":v for k,v in metrics_vqc.items()},
      **{f"svm_{k}":v for k,v in metrics_svm.items()}
    }
    results.append(row)

# 3) summarise
df = pd.DataFrame(results)
df = df.sort_values("vqc_f1", ascending=False)
print("Top 2 configs:\n", df.head(2))

# plot F1 vs reps
plt.plot(df["reps"], df["vqc_f1"], 'o-')
plt.xlabel("Circuit reps")
plt.ylabel("VQC F1-score")
plt.title("Hold-out F1 by circuit depth")
plt.show()
