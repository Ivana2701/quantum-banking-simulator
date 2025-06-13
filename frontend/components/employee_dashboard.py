# components/employee_dashboard.py
import streamlit as st
import requests
import pandas as pd
from components.auth import assert_employee
from components.dashboard import generate_bb84_key, decrypt_with_bb84

def employee_dashboard():
    assert_employee()
    st.title("Employee Dashboard")

    if st.button("Load All Transactions"):
        response = requests.get("http://127.0.0.1:8000/transactions/all")
        if response.status_code == 200:
            transactions = response.json()
            df = pd.DataFrame(transactions)

            st.write("📑 All Transactions:")
            expected_cols = ["transaction_id", "account_id", "encrypted_amount", "timestamp"]

            if df.empty:
                st.info("No transactions available yet.")
                df = pd.DataFrame(columns=expected_cols)
                st.dataframe(df)
            else:
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Some columns are missing: {missing_cols}")
                    st.dataframe(df)
                else:
                    st.dataframe(df[expected_cols])

                # Only try to decrypt if transaction_id column is present and has values
                if "transaction_id" in df.columns and not df["transaction_id"].empty:
                    selected_id = st.selectbox("Select Transaction ID to Decrypt", df["transaction_id"].tolist())
                    if st.button("Decrypt Selected Transaction"):
                        try:
                            selected_encrypted = df[df["transaction_id"] == selected_id]["encrypted_transaction"].iloc[0]
                            _, _, shared_key = generate_bb84_key(length=16)
                            decrypted = decrypt_with_bb84(selected_encrypted.encode(), shared_key)
                            st.write("🔓 **Decrypted Transaction:**", decrypted)
                        except Exception as e:
                            st.error(f"Decryption failed: {e}")
        else:
            st.error("Failed to load transactions.")
