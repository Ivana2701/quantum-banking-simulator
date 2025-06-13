# frontend/components/customer_dashboard.py
import streamlit as st
import requests
from utils.api_helpers import fetch_balance_with_retry


def customer_dashboard():
    st.title("Customer Dashboard")

    # Get logged-in customer ID from session
    customer_id = st.session_state.get("customer_id", None)

    if not customer_id:
        st.error("Unauthorized. Please log in as a customer.")
        st.stop()

    # Fetch real account balance from backend
    try:
        balance = fetch_balance_with_retry(customer_id)
        balance_float = float(str(balance).replace(',', ''))  # Remove commas, ensure float
        st.metric("Account Balance", f"${balance_float:,.2f}")
    except Exception as e:
        st.warning(f"Could not load account balance: {e}")
        st.metric("Account Balance", "$0.00")

    recipient = st.text_input("Recipient Customer ID")
    amount = st.number_input("Amount", min_value=0.01, format="%.2f")
    mode = st.radio("Select Transaction Mode:", ["Demo Mode (educational)", "Real Mode (secure)"])

    if st.button("Send Transaction"):
        if not recipient:
            st.error("Recipient Customer ID is required.")
            return

        with st.spinner("Processing transaction..."):
            try:
                # 1. Generate BB84 Key
                key_response = requests.get("http://127.0.0.1:8000/generate_bb84_key/", timeout=10)
                key_response.raise_for_status()
                key_data = key_response.json()
                shared_key = key_data["bob_results"]
                bb84_key_bytes = bytes(shared_key)

                # 2. Encrypt with BB84 Key
                encrypt_response = requests.post("http://127.0.0.1:8000/encrypt_with_bb84/", json={
                    "plaintext": f"Send ${amount:.2f} to customer {recipient}",
                    "bb84_key": bb84_key_bytes.hex()
                }, timeout=10)
                encrypt_response.raise_for_status()
                encrypted = encrypt_response.json()["encrypted_data"]

                # 3. Choose Endpoint
                endpoint = "encrypt_transaction_demo" if mode.startswith("Demo") else "encrypt_transaction_real"

                # 4. Save Encrypted Transaction
                payload = {
                    "encrypted_data": encrypted,
                    "sender_customer_id": st.session_state.get("customer_id", 1),  # fallback for testing
                    "recipient_customer_id": recipient,
                    "amount": amount
                }
                if mode.startswith("Demo"):
                    payload["bb84_key"] = bb84_key_bytes.hex()

                response = requests.post(f"http://127.0.0.1:8000/{endpoint}/", json=payload, timeout=10)
                response.raise_for_status()

                st.success(f"Transaction successfully sent ({mode})!")

            except requests.exceptions.ConnectTimeout:
                st.error("Backend is not responding (connection timeout). Is the server running?")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error: {http_err}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
