import sys
import os

# Ensure root of the project is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from components.login import show_login
from components.logout import show_logout

st.set_page_config(page_title="Quantum Banking App")

# --- Initialize session state variables ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None

# --- ROUTING ---
if not st.session_state.logged_in:
    show_login()
else:
    show_logout()  # Optional: Add a logout button after login

    # Role-based UI routing
    if st.session_state.role == "Customer":
        from components.customer_dashboard import customer_dashboard
        from components.transactions import show_transactions

        menu_choice = st.sidebar.selectbox("Customer Menu", ["Dashboard", "Transactions"])

        if menu_choice == "Dashboard":
            if st.session_state.customer_id:
                customer_dashboard()
            else:
                st.error("Login incomplete. Customer ID missing.")
        elif menu_choice == "Transactions":
            show_transactions()

    elif st.session_state.role == "Employee":
        from components.employee_dashboard import employee_dashboard
        from components.transactions import show_transactions
        from components.model_evaluation import show_model_evaluation

        menu_choice = st.sidebar.selectbox("Employee Menu", ["Dashboard", "Transactions", "Model Evaluation"])

        if menu_choice == "Dashboard":
            employee_dashboard()
        elif menu_choice == "Transactions":
            show_transactions()
        elif menu_choice == "Model Evaluation":
            show_model_evaluation()
