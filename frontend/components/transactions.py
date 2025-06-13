# frontend/components/transactions.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_transactions():
    st.subheader("Transactions")

    transaction_id = st.number_input("Enter Transaction ID", min_value=1)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Fetch Saved Result"):
            response = requests.get(f"{API_URL}/transactions/results/{transaction_id}")
            if response.status_code == 200:
                st.success(f"Fetched result for Transaction ID {transaction_id}.")
                st.json(response.json())
            else:
                st.error("No results found.")

    with col2:
        if st.button("Run Quantum Verification"):
            with st.spinner("Running quantum verification..."):
                response = requests.post(f"{API_URL}/transactions/verify/{transaction_id}")
                if response.status_code == 200:
                    st.success(f"Transaction ID {transaction_id} verified successfully.")
                    st.json(response.json())
                else:
                    st.error(f"Verification error: {response.text}")



# import streamlit as st
# import requests
# import os
# import json

# API_URL = "http://localhost:8000"

# def show_transactions():
#     st.subheader("Transactions")

#     transaction_id = st.number_input("Enter Transaction ID", min_value=1)

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("Fetch Saved Result"):
#             res = requests.get(f"{API_URL}/transactions/results/{transaction_id}")
#             if res.status_code == 200:
#                 result = res.json()
#                 st.success(f"Transaction ID {transaction_id} verification result fetched.")
#                 st.json(result)
#             else:
#                 st.error("No saved results found for this transaction.")

#     with col2:
#         if st.button("Run Quantum Verification"):
#             with st.spinner("Running quantum verification..."):
#                 res = requests.post(f"{API_URL}/transactions/verify/{transaction_id}")
#                 try:
#                     result = res.json()
#                     if res.status_code == 200:
#                         st.success(f"Transaction ID {transaction_id} verified successfully.")
#                         st.json(result)
#                     else:
#                         st.error(f"Verification failed: {result.get('detail', 'Unknown error')}")
#                 except requests.JSONDecodeError:
#                     st.error("Invalid response from the server. Please check backend logs.")
