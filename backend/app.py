import json
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from backend.encryption.encrypt_data import encrypt
from backend.encryption.decrypt_data import decrypt
from backend.quantum_encryption.pqc_encrypt_data import pqc_encrypt
from backend.quantum_encryption.pqc_decrypt_data import pqc_decrypt
from backend.fraud_detection.qsvm_fraud_detection import detect_fraud_qsvm, run_and_save_qsvm_metrics

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Quantum Banking Backend is running"}

@app.post("/encrypt/")
def encrypt_endpoint(data: str):
    encrypted_data = encrypt(data)
    return {"encrypted": encrypted_data.decode()}

@app.post("/decrypt/")
def decrypt_endpoint(encrypted_data: str):
    decrypted_data = decrypt(encrypted_data.encode())
    return {"decrypted": decrypted_data}

@app.get("/transactions/results/{transaction_id}")
def get_transaction_result(transaction_id: int):
    result_path = f"./transaction_results/{transaction_id}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Transaction result not found.")

@app.post("/transactions/verify/{transaction_id}")
def verify_and_save_transaction(transaction_id: int):
    # Now clearly uses your trained QSVM model:
    is_fraudulent = detect_fraud_qsvm(transaction_id)
    
    result_data = {
        "transaction_id": transaction_id,
        "is_fraudulent": bool(is_fraudulent),
        "verified_at": datetime.now().isoformat()
    }
    
    save_dir = "./transaction_results"
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"{transaction_id}.json")
    
    with open(result_path, "w") as f:
        json.dump(result_data, f)
    
    return result_data

@app.get("/run_metrics/{algorithm}")
def run_metrics(algorithm: str, background_tasks: BackgroundTasks):
    if algorithm == "quantum_qsvm":
        background_tasks.add_task(run_and_save_qsvm_metrics)
    else:
        return {"error": "Invalid algorithm"}
    return {"message": "Metrics are being generated."}

@app.get("/model_metrics/{algorithm}")
def get_model_metrics(algorithm: str):
    filepath = f"./backend/fraud_detection/metrics_data/{algorithm}.json"
    try:
        with open(filepath, "r") as file:
            metrics = json.load(file)
    except FileNotFoundError:
        return {"error": "Metrics file not found. Please run metrics first."}
    return metrics
