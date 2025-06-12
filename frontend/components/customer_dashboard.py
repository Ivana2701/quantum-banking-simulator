# frontend/components/customer_dashboard.py
import streamlit as st
import requests

def customer_dashboard():
    st.title("Customer Dashboard")
    st.metric("Account Balance", "$12,500")

    recipient = st.text_input("Recipient Customer ID")
    amount = st.number_input("Amount", min_value=0.01, format="%.2f")

    mode = st.radio("Select Transaction Mode:", ["Demo Mode (educational)", "Real Mode (secure)"])

    if st.button("Send Transaction"):
        transaction_data = f"Send ${amount:.2f} to customer {recipient}"

        # Generate key from backend
        key_response = requests.get("http://127.0.0.1:8000/generate_bb84_key/")
        shared_key = key_response.json()["bob_results"]
        bb84_key_bytes = bytes(shared_key)

        # Encrypt via backend
        encrypt_response = requests.post("http://127.0.0.1:8000/encrypt_with_bb84/", json={
            "plaintext": transaction_data,
            "bb84_key": bb84_key_bytes.hex()
        })

        encrypted = encrypt_response.json()["encrypted_data"]

        endpoint = "encrypt_transaction/demo" if mode.startswith("Demo") else "encrypt_transaction/real"

        response = requests.post(f"http://127.0.0.1:8000/{endpoint}/", json={
            "encrypted_data": encrypted,
            "sender_customer_id": st.session_state["customer_id"],
            "recipient_customer_id": recipient,
            "amount": amount,
            "bb84_key": bb84_key_bytes.hex() if mode.startswith("Demo") else None
        })

        if response.status_code == 200:
            st.success(f"Transaction successfully sent ({mode})!")
        else:
            st.error("Transaction failed.")
