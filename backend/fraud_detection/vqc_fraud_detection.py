# backend/fraud_detection/vqc_fraud_detection.py
import os
import numpy as np
import pandas as pd
import json

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA

def load_data():
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
    return df

def preprocess_data(df):
    X = df[['amount', 'hour', 'location']].values
    y = df['is_fraud'].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_bal)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, y_train_bal, y_test

def train_vqc(X_train, y_train):
    algorithm_globals.random_seed = 42
    feature_map = PauliFeatureMap(feature_dimension=2, reps=3, paulis=['Z','X','ZZ'])
    ansatz = RealAmplitudes(num_qubits=2, reps=3)
    optimizer = COBYLA(maxiter=250)
    sampler = AerSampler()
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)
    vqc.fit(X_train, y_train)
    return vqc

def evaluate_and_save_metrics(model, X_test, y_test, filepath="./metrics_data/quantum_vqc.json"):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": roc_curve(y_test, y_pred)[0].tolist(),
            "tpr": roc_curve(y_test, y_pred)[1].tolist(),
            "auc": auc(*roc_curve(y_test, y_pred)[:2])
        }
    }
    script_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(script_dir, "metrics_data", "quantum_vqc.json")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    print("Saving metrics to:", json_path)
    with open(json_path, "w") as file:
        json.dump(metrics, file, indent=4)

def detect_fraud_vqc(transaction_id: int):
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Proper scaling and PCA fitting on training data
    scaler = StandardScaler().fit(df[['amount', 'hour', 'location']])
    pca = PCA(n_components=2).fit(scaler.transform(df[['amount', 'hour', 'location']]))

    # Extract transaction and transform properly
    transaction = df.iloc[transaction_id]
    location_encoded = pd.factorize(df['location'])[0][transaction_id]
    features = np.array([[transaction['amount'], transaction['hour'], location_encoded]])
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    model = train_vqc(X_train, y_train)
    prediction = model.predict(features_pca)
    # return int(prediction[0])
    return int(prediction)
    # return int(np.array(prediction).item()) # explicitly handle scalar predictions with NumPy:

def run_and_save_vqc_metrics():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    vqc = train_vqc(X_train, y_train)
    evaluate_and_save_metrics(vqc, X_test, y_test)

if __name__ == "__main__":
    run_and_save_vqc_metrics()
