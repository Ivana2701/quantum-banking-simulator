from fastapi import FastAPI
from backend.encryption.encrypt_data import encrypt
from backend.encryption.decrypt_data import decrypt
from backend.quantum_encryption.pqc_encrypt_data import pqc_encrypt
from backend.quantum_encryption.pqc_decrypt_data import pqc_decrypt
from backend.fraud_detection.vqc_fraud_detection import detect_fraud_vqc
from backend.fraud_detection.qsvm_fraud_detection import detect_fraud_qsvm
from backend.fraud_detection.qsvm_fraud_detection import run_and_save_qsvm_metrics
from backend.fraud_detection.vqc_fraud_detection import run_and_save_vqc_metrics
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
import subprocess

app = FastAPI()
router = APIRouter()

# Map algorithm to script paths
ALGO_SCRIPT_MAP = {
    "quantum_vqc": "backend/fraud_detection/vqc_fraud_detection.py",
    "quantum_qsvm": "backend/fraud_detection/qsvm_fraud_detection.py",
    # Add more if needed
}

@router.post("/run_model/{algorithm}")
async def run_model(algorithm: str):
    script_path = ALGO_SCRIPT_MAP.get(algorithm)
    if not script_path:
        raise HTTPException(status_code=404, detail="Algorithm script not found")
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return {"status": "success", "stdout": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

@app.get("/verify_transaction/{transaction_id}")
def verify_transaction(transaction_id: int):
    result_vqc = detect_fraud_vqc(transaction_id)
    result_qsvm = detect_fraud_qsvm(transaction_id)
    return {"vqc_result": result_vqc, "qsvm_result": result_qsvm}

@app.get("/run_metrics/{algorithm}")
def run_metrics(algorithm: str, background_tasks: BackgroundTasks):
    if algorithm == "quantum_qsvm":
        background_tasks.add_task(run_and_save_qsvm_metrics)
    elif algorithm == "quantum_vqc":
        background_tasks.add_task(run_and_save_vqc_metrics)
    else:
        return {"error": "Invalid algorithm"}
    return {"message": "Metrics are being generated."}

@app.get("/model_metrics/{algorithm}")
def get_model_metrics(algorithm: str):
    valid_algorithms = {
        "quantum_vqc",
        "quantum_zfeaturemap",
        "quantum_qsvm",
        "quantum_qnn"
    }

    if algorithm not in valid_algorithms:
        return {"error": "Invalid algorithm selected."}

    filepath = f".backend/fraud_detection/metrics_data/{algorithm}.json"

    try:
        with open(filepath, "r") as file:
            metrics = json.load(file)
    except FileNotFoundError:
        return {"error": "Metrics file not found. Please run metrics first."}

    return metrics

app.include_router(router)
