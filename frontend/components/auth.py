# frontend/components/auth.py
import os, requests, streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")
_session = requests.Session()

def login_user(username: str, password: str, role: str) -> bool:
    # 1) fetch CSRF token (cookie + raw token)
    resp = _session.get(f"{API_URL}/auth/csrf-token", timeout=5)
    resp.raise_for_status()
    csrf_token = resp.json()["csrf_token"]

    # 2) login with header + cookie
    payload = {"username": username, "password": password, "role": role.capitalize()}
    headers = {"X-CSRF-Token": csrf_token}
    login_resp = _session.post(
        f"{API_URL}/auth/login",
        json=payload,
        headers=headers,
        timeout=5
    )
    if not login_resp.ok:
        st.error(f"Login failed: {login_resp.text}")
        return False

    data = login_resp.json()
    st.session_state["logged_in"] = True
    st.session_state["role"]      = data["role"]
    if role.lower() == "customer":
        st.session_state["customer_id"] = data["account_id"]
    else:
        st.session_state["employee_id"] = data["account_id"]
    return True
