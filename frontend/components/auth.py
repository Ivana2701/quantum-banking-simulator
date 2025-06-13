# frontend/components/auth.py
import sys
import os
import streamlit as st

# Ensure backend is accessible from frontend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.authentication.authentication_service import authenticate_user

def is_logged_in():
    return st.session_state.get("logged_in", False)

def get_user_role():
    return st.session_state.get("role", None)

def get_customer_id():
    return st.session_state.get("customer_id", None)

def get_employee_id():
    return st.session_state.get("employee_id", None)

def assert_customer():
    if not is_logged_in() or get_user_role() != "Customer" or not get_customer_id():
        st.error("Unauthorized access. Please log in as a customer.")
        st.stop()

def assert_employee():
    if not is_logged_in() or get_user_role() != "Employee":
        st.error("Unauthorized access. Please log in as an employee.")
        st.stop()

def login_user(username, password, role):
    account_id = authenticate_user(username, password, role)

    if account_id:
        st.session_state["logged_in"] = True
        st.session_state["role"] = role.capitalize()

        if role.lower() == "customer":
            st.session_state["customer_id"] = account_id
        elif role.lower() == "employee":
            st.session_state["employee_id"] = account_id

        return True
    return False

def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["role"] = None
    st.session_state["customer_id"] = None
    st.session_state["employee_id"] = None
