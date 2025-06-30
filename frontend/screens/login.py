# frontend/screens/login.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # OAuth2PasswordRequestForm expects form-data, not JSON
        form = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": "",
            "client_id": "",
            "client_secret": "",
        }
        try:
            r = requests.post(f"{API_URL}/auth/token", data=form)
        except Exception as e:
            st.error(f"Cannot reach server: {e}")
            return

        if r.status_code != 200:
            st.error(f"Login failed ({r.status_code}): {r.text}")
            return

        token = r.json().get("access_token")
        if not token:
            st.error("No token returned.")
            return

        st.session_state.token = token

        # fetch the user's account to learn their role
        profile = requests.get(
            f"{API_URL}/accounts/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        if profile.status_code == 200:
            st.session_state.account_type = profile.json()["account_type"]
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Could not fetch account info.")
