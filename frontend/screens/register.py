# frontend/screens/register.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_register():
    st.title("Register")
    full_name = st.text_input("Full Name")
    username  = st.text_input("Username")
    email     = st.text_input("Email")
    password  = st.text_input("Password", type="password")
    confirm   = st.text_input("Confirm Password", type="password")
    role      = st.selectbox("Register as", ["customer","employee"])
    if st.button("Sign Up"):
        if password != confirm:
            st.error("Passwords must match")
            return
        payload = {
            "username": username,
            "full_name": full_name,
            "email": email,
            "password": password,
            "account_type": role,
        }
        r = requests.post(f"{API_URL}/auth/register", json=payload)
        if r.status_code == 201:
            st.success("Registered! Please log in.")
        else:
            st.error(f"Error ({r.status_code}): {r.text}")
