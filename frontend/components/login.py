# frontend/components/login.py
import streamlit as st
from components.auth import login_user

def show_login():
    st.title("QBank Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role     = st.selectbox("Login as", ["customer","employee"])

    if st.button("Login"):
        if login_user(username, password, role):
            st.success(f"Logged in as {role.capitalize()}!")
            st.experimental_rerun()
        else:
            st.error("Invalid login credentials.")
