# components/employee_dashboard.py
import streamlit as st
import requests
import pandas as pd
from components.auth import assert_employee
from components.dashboard import generate_bb84_key, decrypt_with_bb84
from animation.qkd_animation import bb84_qkd_simulation  # Importing the new quantum simulation

def employee_dashboard():
    assert_employee()
    st.title("Employee Dashboard")

    menu_choice = st.sidebar.selectbox("Employee Menu", ["Transactions", "Quantum Security Demo"])

    if menu_choice == "Transactions":
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

    elif menu_choice == "Quantum Security Demo":
        bb84_qkd_simulation()
