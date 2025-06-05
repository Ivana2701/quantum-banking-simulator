import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel


# Reproducibility
np.random.seed(42)

def generate_and_prepare_data():
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.5, 0.5],
        flip_y=0.02,
        class_sep=0.9,
        random_state=42
    )
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_quantum_svm(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
    quantum_kernel = QuantumKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    return qsvc


def train_classical_svm(X_train, y_train):
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train, y_train)
    return svc

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)

    print(f"\n✅ {model_name} Results:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if 'Quantum' in model_name else 'Greens')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # plt.show() //not here, but in FE

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return y_pred, y_score, fpr, tpr, roc_auc

def save_metrics(y_test, y_pred, fpr, tpr, roc_auc, filename):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc
        }
    }
    with open(filename, "w") as file:
        json.dump(metrics, file, indent=4)

# Main execution
X_train, X_test, y_train, y_test = generate_and_prepare_data()

# Train models
qsvc = train_quantum_svm(X_train, y_train)
svc = train_classical_svm(X_train, y_train)

# Evaluate models
quantum_results = evaluate_model(qsvc, X_test, y_test, "Quantum SVM")
classical_results = evaluate_model(svc, X_test, y_test, "Classical SVM")

# ROC curves comparison
plt.figure(figsize=(8, 6))
plt.plot(quantum_results[2], quantum_results[3], label=f'Quantum SVM (AUC = {quantum_results[4]:.2f})')
plt.plot(classical_results[2], classical_results[3], label=f'Classical SVM (AUC = {classical_results[4]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
# plt.show()

# Save metrics
script_dir = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(script_dir, "metrics_data", "quantum_qsvm.json")

print("Saving metrics to:", json_path)

save_metrics(y_test, quantum_results[0], quantum_results[2], quantum_results[3], quantum_results[4], json_path)

def run_and_save_qsvm_metrics():
    X_train, X_test, y_train, y_test = generate_and_prepare_data()
    qsvc = train_quantum_svm(X_train, y_train)
    y_pred, _, fpr, tpr, roc_auc = evaluate_model(qsvc, X_test, y_test, "Quantum SVM")
    save_metrics(y_test, y_pred, fpr, tpr, roc_auc, "/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/metrics_data/quantum_qsvm.json")


def load_data():
    load_dotenv()
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    df = pd.read_sql(
        "SELECT amount, EXTRACT(hour FROM transaction_time) AS hour, location, is_fraud FROM transactions_qml",
        engine
    )
    df['location'] = pd.factorize(df['location'])[0]
    return df

def preprocess_data(df):
    X = df[['amount', 'hour', 'location']].values
    y = df['is_fraud'].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, y_train, y_test

def train_qsvm(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
    quantum_kernel = QuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    return qsvc

def detect_fraud_qsvm(transaction_id: int):
    df = load_data()
    transaction = df.iloc[transaction_id]
    location_encoded = transaction['location']
    features = np.array([[transaction['amount'], transaction['hour'], location_encoded]])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # In production, load a pre-trained model instead!
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_qsvm(X_train, y_train)

    prediction = model.predict(features_pca)
    return int(prediction[0])
