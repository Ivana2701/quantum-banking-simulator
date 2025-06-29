import streamlit as st
import requests

API_URL = "http://localhost:8000"   # ← make sure this points at your FastAPI server

st.set_page_config(page_title="QBank", layout="centered")

# initialize session state
if "token" not in st.session_state:
    st.session_state.token = None
if "account_type" not in st.session_state:
    st.session_state.account_type = None

def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # OAuth2PasswordRequestForm expects form-data, not JSON
        form = {
            "username": username,
            "password": password,
            "grant_type": "",
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
        else:
            st.error("Could not fetch account info.")

def register_page():
    st.title("📝 Register")
    full_name = st.text_input("Full Name")
    username  = st.text_input("Username")
    email     = st.text_input("Email")
    password  = st.text_input("Password", type="password")
    password2 = st.text_input("Confirm Password", type="password")
    account_type = st.selectbox("Register as", ["customer", "employee"])
    if st.button("Sign Up"):
        if password != password2:
            st.error("Passwords do not match.")
            return

        payload = {
            "full_name": full_name,
            "username":  username,
            "email":     email,
            "password":  password,
            "account_type": account_type
        }
        try:
            r = requests.post(f"{API_URL}/auth/register", json=payload)
        except Exception as e:
            st.error(f"Cannot reach server: {e}")
            return

        if r.status_code == 201:
            st.success("Account created! You can now log in.")
        else:
            st.error(f"Registration failed ({r.status_code}): {r.text}")

def customer_dashboard():
    st.title("🏦 Customer Dashboard")
    hdr = {"Authorization": f"Bearer {st.session_state.token}"}

    # balance
    bal = requests.get(f"{API_URL}/customer/balance", headers=hdr)
    if bal.status_code == 200:
        st.metric("Balance", bal.json().get("encrypted_balance"))
    else:
        st.error("Could not fetch balance.")

    # send money
    st.subheader("Send Money")
    to_id = st.text_input("Recipient Account ID")
    amt   = st.number_input("Amount", min_value=0.01, step=0.01)
    if st.button("Send"):
        r = requests.post(
            f"{API_URL}/customer/transfer",
            headers=hdr,
            json={"to_account_id": int(to_id), "amount": amt}
        )
        if r.status_code == 200:
            st.success("Sent!")
        else:
            st.error(f"Failed: {r.text}")

    # transactions
    st.subheader("Transactions")
    tx = requests.get(f"{API_URL}/customer/transactions", headers=hdr)
    if tx.status_code == 200:
        st.write(tx.json())
    else:
        st.error("Could not fetch transactions.")

def employee_dashboard():
    st.title("👩‍💼 Employee Dashboard")
    hdr = {"Authorization": f"Bearer {st.session_state.token}"}
    r = requests.get(f"{API_URL}/employee/transactions/all", headers=hdr)
    if r.status_code == 200:
        st.write(r.json())
    else:
        st.error("Could not fetch all transactions.")

# sidebar navigation
if st.session_state.token is None:
    page = st.sidebar.radio("Go to", ["Login", "Register"])
    if page == "Login":
        login_page()
    else:
        register_page()
else:
    page = st.sidebar.radio("Dashboard", ["Customer", "Employee", "Logout"])
    if page == "Customer":
        customer_dashboard()
    elif page == "Employee":
        employee_dashboard()
    else:
        # logout
        st.session_state.token = None
        st.session_state.account_type = None
        st.experimental_rerun()
