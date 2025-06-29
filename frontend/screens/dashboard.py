# frontend/screens/dashboard.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_customer_dashboard(token):
    st.title("Customer Dashboard")
    headers = {"Authorization": f"Bearer {token}"}
    acct = requests.get(f"{API_URL}/accounts/me", headers=headers).json()
    txns = requests.get(f"{API_URL}/customer/transactions", headers=headers).json()
    st.subheader(f"Hello, {acct['full_name']}")
    st.write("Balance:", acct["encrypted_balance"])
    st.write("Your transactions:", txns)

def show_employee_dashboard(token):
    st.title("Employee Dashboard")
    headers = {"Authorization": f"Bearer {token}"}
    txns = requests.get(f"{API_URL}/employee/transactions/all", headers=headers).json()
    st.write("All transactions:", txns)
