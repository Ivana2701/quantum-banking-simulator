# frontend/components/transactions.py
import streamlit as st
import requests

def show_transactions():
    st.subheader("Transactions")

    transaction_id = st.number_input("Enter Transaction ID", min_value=1)

    if st.button("Verify Quantumly"):
        res = requests.get(f"http://localhost:8000/verify_transaction/{transaction_id}")
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("Transaction verification failed.")
