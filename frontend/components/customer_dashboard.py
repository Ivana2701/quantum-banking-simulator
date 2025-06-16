import streamlit as st
import requests
from utils.api_helpers import fetch_balance_with_retry
from animation.quantum_safe_banking import quantum_security_simulation  # Adjust import as needed

def customer_dashboard():
    st.title("Customer Dashboard")

    customer_id = st.session_state.get("customer_id")

    if not customer_id:
        st.error("Unauthorized. Please log in as a customer.")
        st.stop()

    # Fetch real account balance
    try:
        balance = fetch_balance_with_retry(customer_id)
        balance_float = float(str(balance).replace(',', ''))
        st.metric("Account Balance", f"${balance_float:,.2f}")
    except Exception as e:
        st.warning(f"Could not load account balance: {e}")
        st.metric("Account Balance", "$0.00")

    recipient = st.text_input("Recipient Account ID")
    amount = st.number_input("Amount", min_value=0.01, format="%.2f")
    mode = st.radio("Select Transaction Mode:", ["Demo Mode (educational)", "Real Mode (secure)"])

    if st.button("Send Transaction"):
        if not recipient:
            st.error("Recipient Account ID is required.")
            return
        
        if int(recipient) == customer_id:
            st.error("You cannot send money to your own account.")
            return

        # First Pop-up
        st.info("🔐 Quantum-Safe Banking Security Demonstration")
        if st.button("OK", key="quantum_security_intro"):
            # Second Pop-up
            demo_interest = st.radio("Would you like to see a live quantum security demo?", ["Yes", "No"], key="demo_interest")
            if demo_interest == "Yes":
                quantum_security_simulation()

            # Proceed with transaction after demo (or without demo)
            with st.spinner("Processing transaction..."):
                try:
                    key_response = requests.get("http://127.0.0.1:8000/generate_bb84_key/", timeout=10)
                    key_response.raise_for_status()
                    key_data = key_response.json()
                    shared_key = key_data["bob_results"]
                    bb84_key_bytes = bytes(shared_key)

                    encrypt_response = requests.post("http://127.0.0.1:8000/encrypt_with_bb84/", json={
                        "plaintext": f"Send ${amount:.2f} to account {recipient}",
                        "bb84_key": bb84_key_bytes.hex()
                    }, timeout=10)
                    encrypt_response.raise_for_status()
                    encrypted = encrypt_response.json()["encrypted_data"]

                    endpoint = "encrypt_transaction_demo" if mode.startswith("Demo") else "encrypt_transaction_real"

                    payload = {
                        "encrypted_data": encrypted,
                        "sender_customer_id": customer_id,
                        "recipient_account_id": int(recipient),
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
