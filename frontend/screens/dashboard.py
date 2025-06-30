# frontend/screens/dashboard.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_customer_dashboard():
    st.title("🏦 Customer Dashboard")
    hdr = {"Authorization": f"Bearer {st.session_state.token}"}

    # balance
    bal = requests.get(f"{API_URL}/customer/balance", headers=hdr)
    if bal.status_code == 200:
        st.metric("Balance", bal.json().get("encrypted_balance"))
    else:
        st.error("Could not fetch balance.")

    # send money
    st.subheader("Send Money")
    to_id = st.text_input("Recipient Account ID")
    amt = st.number_input("Amount", min_value=0.01, step=0.01)
    if st.button("Send"):
        r = requests.post(
            f"{API_URL}/customer/transfer",
            headers=hdr,
            json={"to_account_id": int(to_id), "amount": amt}
        )
        if r.status_code == 200:
            st.success("Sent!")
        else:
            st.error(f"Failed: {r.text}")

    # transactions
    st.subheader("Transactions")
    tx = requests.get(f"{API_URL}/customer/transactions", headers=hdr)
    if tx.status_code == 200:
        st.write(tx.json())
    else:
        st.error("Could not fetch transactions.")

def show_employee_dashboard():
    st.title("👩‍💼 Employee Dashboard")
    hdr = {"Authorization": f"Bearer {st.session_state.token}"}
    r = requests.get(f"{API_URL}/employee/transactions/all", headers=hdr)
    if r.status_code == 200:
        st.write(r.json())
    else:
        st.error("Could not fetch all transactions.")
