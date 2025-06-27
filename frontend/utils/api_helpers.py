import requests
import time
import os

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")

def fetch_balance_with_retry(customer_id, retries=3, delay=2):
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/account_balance/{customer_id}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return float(response.json()["balance"])
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

def send_demo_transaction(sender_id: int,
                          recipient_id: int,
                          amount: float,
                          encrypted_data: str,
                          bb84_key: str,
                          retries=3,
                          delay=2):
    """
    Sends to the /encrypt_transaction_demo/ endpoint.
    Returns the JSON response.
    """
    url = f"{BASE_URL}/encrypt_transaction_demo/"
    payload = {
        "encrypted_data": encrypted_data,
        "sender_customer_id": sender_id,
        "recipient_customer_id": recipient_id,
        "amount": amount,
        "bb84_key": bb84_key,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

def send_real_transaction(sender_id: int,
                          recipient_account_id: int,
                          amount: float,
                          encrypted_data: str,
                          retries=3,
                          delay=2):
    """
    Sends to the /encrypt_transaction_real/ endpoint.
    Returns the JSON response.
    """
    url = f"{BASE_URL}/encrypt_transaction_real/"
    payload = {
        "encrypted_data": encrypted_data,
        "sender_customer_id": sender_id,
        "recipient_account_id": recipient_account_id,
        "amount": amount,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
