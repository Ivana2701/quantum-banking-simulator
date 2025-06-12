# components/login.py
import streamlit as st
from components.auth import login_user

def show_login():
    st.title("QBank Login")

    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Login as", ["customer", "employee"])

    if st.button("Login"):
        if login_user(user, password, role):
            st.success(f"Successfully logged in as {role.capitalize()}!")
            st.rerun()
        else:
            st.error("Invalid login credentials.")
