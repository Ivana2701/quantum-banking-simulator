import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

def save_metrics(y_true, y_pred, y_prob, algorithm_name):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": roc_curve(y_true, y_prob)[0].tolist(),
            "tpr": roc_curve(y_true, y_prob)[1].tolist(),
            "auc": auc(*roc_curve(y_true, y_prob)[:2])
        }
    }

    with open(f"../metrics_data/{algorithm_name}.json", "w") as file:
        json.dump(metrics, file, indent=4)
