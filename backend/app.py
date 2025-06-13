import json
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from backend.encryption.encrypt_data import encrypt
from backend.encryption.decrypt_data import decrypt
from backend.quantum_encryption.pqc_encrypt_data import pqc_encrypt
from backend.quantum_encryption.pqc_decrypt_data import pqc_decrypt
from backend.quantum_encryption.bb84 import generate_bb84_key, encrypt_with_bb84, decrypt_with_bb84
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

@app.post("/transactions/verify/{transaction_id}")
def verify_and_save_transaction(transaction_id: int):
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

# DEMO ONLY: Temporarily stores keys for visualization.
@app.post("/encrypt_transaction_demo/")
def save_demo_transaction(data: dict):
    encrypted_transaction = data["encrypted_data"]
    sender_customer_id = data["sender_customer_id"]
    recipient_customer_id = data["recipient_customer_id"]
    amount = data["amount"]
    bb84_key = data["bb84_key"]

    transaction_id = datetime.now().strftime("%Y%m%d%H%M%S")

    transaction_meta = {
        "transaction_id": transaction_id,
        "sender_customer_id": sender_customer_id,
        "recipient_customer_id": recipient_customer_id,
        "amount": amount,
        "encrypted_transaction": encrypted_transaction,
        "bb84_key": bb84_key,  # Demo only, not secure!
        "timestamp": datetime.now().isoformat()
    }

    save_dir = "./transaction_results/demo"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{transaction_id}.json"

    with open(save_path, "w") as f:
        json.dump(transaction_meta, f)

    return {
        "transaction_id": transaction_id,
        "status": "saved (demo only)",
        "warning": "Educational demo storage ONLY. Not secure for production!"
    }

# REAL MODE: Secure, no storage of keys.
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

@app.post("/encrypt_transaction_real/")
def save_real_transaction(data: dict):
    encrypted_data = data["encrypted_data"]
    sender_customer_id = data["sender_customer_id"]
    recipient_customer_id = data["recipient_customer_id"]
    amount = data["amount"]

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO transactions_encrypted (
                customer_id, recipient_account, amount, encrypted_data, timestamp
            ) VALUES (%s, %s, %s, %s, %s)
            RETURNING transaction_id;
        """, (
            sender_customer_id,
            recipient_customer_id,
            amount,
            encrypted_data,
            datetime.now()
        ))

        transaction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return {
            "transaction_id": transaction_id,
            "status": "securely stored in DB",
            "note": "BB84 key was discarded after encryption."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save transaction: {e}")

@app.get("/transactions/results/{transaction_id}")
def get_transaction_result(transaction_id: str):
    result_dirs = ["./transaction_results/demo", "./transaction_results/real", "./transaction_results"]
    for result_dir in result_dirs:
        result_path = os.path.join(result_dir, f"{transaction_id}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail="Transaction result not found.")

@app.get("/transactions/all")
def get_all_transactions():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                transaction_id, 
                customer_id, 
                recipient_account, 
                amount, 
                encrypted_data, 
                timestamp
            FROM transactions_encrypted;
        """)

        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        transactions = [dict(zip(columns, row)) for row in rows]

        cur.close()
        conn.close()
        return transactions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transactions: {e}")

@app.get("/generate_bb84_key/")
def generate_bb84_key_api(length: int = 32):  # <- Renamed function to avoid name conflict
    try:
        alice_bits, alice_bases, bob_results = generate_bb84_key(length)  # This now calls the imported one
        return {
            "alice_bits": alice_bits.tolist(),
            "alice_bases": alice_bases.tolist(),
            "bob_results": bob_results.tolist()
        }
    except Exception as e:
        print(f"Error in generate_bb84_key: {e}, Type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Key generation failed: {e}")

@app.post("/encrypt_with_bb84/")
def api_encrypt_with_bb84(data: dict):
    plaintext = data["plaintext"]
    bb84_key_hex = data["bb84_key"]
    bb84_key = np.frombuffer(bytes.fromhex(bb84_key_hex), dtype=np.uint8)
    encrypted = encrypt_with_bb84(plaintext, bb84_key)
    return {"encrypted_data": encrypted.decode()}

@app.post("/decrypt_with_bb84/")
def api_decrypt_with_bb84(data: dict):
    encrypted_data = data["encrypted_data"]
    bb84_key_hex = data["bb84_key"]
    bb84_key = np.frombuffer(bytes.fromhex(bb84_key_hex), dtype=np.uint8)
    decrypted = decrypt_with_bb84(encrypted_data.encode(), bb84_key)
    return {"decrypted_data": decrypted}

@app.get("/account_balance/{customer_id}")
def get_account_balance(customer_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT account_balance FROM customers WHERE customer_id = %s;", (customer_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            return {"balance": float(result[0])}
        else:
            raise HTTPException(status_code=404, detail="Customer not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {e}")
