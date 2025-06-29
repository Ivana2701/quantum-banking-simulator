# frontend/screens/login.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        form_data = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": "",
            "client_id": "",
            "client_secret": "",
        }
        try:
            resp = requests.post(f"{API_URL}/auth/token", data=form_data)
        except Exception as e:
            st.error(f"Server error: {e}")
            return
        if resp.status_code != 200:
            st.error(f"Login failed ({resp.status_code}): {resp.text}")
            return
        try:
            token = resp.json().get("access_token")
        except ValueError:
            st.error(f"Bad response JSON: {resp.text}")
            return
        if not token:
            st.error(f"No token in response: {resp.text}")
            return

        st.session_state.token = token
        st.session_state.account_type = requests.get(
            f"{API_URL}/accounts/me", 
            headers={"Authorization": f"Bearer {token}"}
        ).json()["account_type"]
        st.success("Logged in!")
        st.experimental_rerun()
