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

def register_page():
    st.title("📝 Register")
    
    # Basic information
    st.subheader("Account Information")
    
    # Use the label parameter with HTML styling for required fields
    full_name = st.text_input("Full Name *", key="full_name", placeholder="Enter your full name", help="Required field")
    
    username = st.text_input("Username *", key="username", placeholder="Choose a username (minimum 3 characters)", help="Required field")
    
    email = st.text_input("Email *", key="email", placeholder="Enter your email address", help="Required field")
    
    password = st.text_input("Password *", type="password", key="password", placeholder="Enter password (minimum 8 characters)", help="Required field")
    
    password2 = st.text_input("Confirm Password *", type="password", key="password2", placeholder="Confirm your password", help="Required field")
    
    account_type = st.selectbox("Account Type *", ["customer", "employee"], key="account_type", help="Required field")
    
    # Optional contact information
    st.subheader("Contact Information (Optional)")
    with st.expander("Address Information", expanded=False):
        street = st.text_input("Street Address", placeholder="123 Main Street")
        city = st.text_input("City", placeholder="Sofia")
        state = st.text_input("State/Province", placeholder="Sofia Province")
        country = st.text_input("Country", placeholder="Bulgaria")
        postal_code = st.text_input("Postal Code", placeholder="1000")
    
    with st.expander("Phone Information", expanded=False):
        phone_number = st.text_input("Phone Number", placeholder="+359888123456")
        phone_type = st.selectbox("Phone Type", ["mobile", "work", "home"])
    
    with st.expander("Device Information", expanded=False):
        device_name = st.text_input("Device Name", placeholder="My Laptop")
        
    if st.button("Sign Up", type="primary"):
        # Client-side validation
        errors = []
        
        if not full_name or len(full_name.strip()) == 0:
            errors.append("Full Name is required")
        
        if not username or len(username.strip()) < 3:
            errors.append("Username is required and must be at least 3 characters long")
        
        if not email or len(email.strip()) == 0:
            errors.append("Email is required")
        elif "@" not in email or "." not in email:
            errors.append("Please enter a valid email address")
        
        if not password or len(password) < 8:
            errors.append("Password is required and must be at least 8 characters long")
        
        if not password2:
            errors.append("Password confirmation is required")
        elif password != password2:
            errors.append("Passwords do not match")
        
        if errors:
            st.error("Please fix the following issues:")
            for error in errors:
                st.error(f"• {error}")
            return

        payload = {
            "full_name": full_name.strip(),
            "username": username.strip(),
            "email": email.strip(),
            "password": password,
            "account_type": account_type
        }
        
        # Add optional fields if provided
        if street and street.strip():
            payload["street"] = street.strip()
        if city and city.strip():
            payload["city"] = city.strip()
        if state and state.strip():
            payload["state"] = state.strip()
        if country and country.strip():
            payload["country"] = country.strip()
        if postal_code and postal_code.strip():
            payload["postal_code"] = postal_code.strip()
        if phone_number and phone_number.strip():
            payload["phone_number"] = phone_number.strip()
            payload["phone_type"] = phone_type
        if device_name and device_name.strip():
            payload["device_name"] = device_name.strip()
            payload["device_fingerprint"] = f"web_{username}_{device_name}".replace(" ", "_")
        
        try:
            r = requests.post(f"{API_URL}/accounts/createAccount", json=payload)
        except requests.exceptions.ConnectionError:
            st.error("❌ **Cannot connect to server**")
            st.error("Please make sure the backend server is running and try again.")
            return
        except requests.exceptions.Timeout:
            st.error("❌ **Request timed out**")
            st.error("The server is taking too long to respond. Please try again.")
            return
        except Exception as e:
            st.error(f"❌ **Network error**: {str(e)}")
            return

        if r.status_code == 201:
            st.success("🎉 **Account created successfully!**")
            st.success("You can now log in with your credentials.")
        elif r.status_code == 422:
            # Handle validation errors from the server
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error("❌ **Validation Error**")
                    if isinstance(error_detail["detail"], list):
                        for error in error_detail["detail"]:
                            field = error.get("loc", ["unknown"])[-1]
                            message = error.get("msg", "Invalid value")
                            st.error(f"• **{field}**: {message}")
                    else:
                        st.error(f"• {error_detail['detail']}")
                else:
                    st.error("❌ **Please check your input and try again**")
            except:
                st.error("❌ **Invalid input format**")
                st.error("Please check all fields and try again.")
        elif r.status_code == 400:
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error("❌ **Registration Failed**")
                    st.error(f"• {error_detail['detail']}")
                else:
                    st.error("❌ **Bad request**")
                    st.error("Please check your input and try again.")
            except:
                st.error("❌ **Username might already be taken**")
                st.error("Please try a different username.")
        elif r.status_code == 500:
            st.error("❌ **Server Error**")
            st.error("Something went wrong on our end. Please try again later.")
        else:
            st.error(f"❌ **Registration failed**")
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error(f"• {error_detail['detail']}")
                else:
                    st.error(f"• Status code: {r.status_code}")
            except:
                st.error(f"• Status code: {r.status_code}")

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
    # Show only relevant dashboard based on account type
    if st.session_state.account_type == "customer":
        dashboard_pages = ["Home"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.session_state.account_type = None
            st.rerun()
        if page == "Home":
            customer_dashboard()
    elif st.session_state.account_type == "employee":
        dashboard_pages = ["Home"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.session_state.account_type = None
            st.rerun()
        if page == "Home":
            employee_dashboard()
    else:
        # logout if account type is unknown
        st.session_state.token = None
        st.session_state.account_type = None
        st.rerun()
