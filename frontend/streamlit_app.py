import streamlit as st
import time
from screens.login import show_login
from screens.register import show_register
from screens.transactions import show_customer_transactions, show_employee_transactions
from screens.dashboard import show_employee_dashboard
from screens.admin_dashboard import show_admin_dashboard
from screens.qb_learn import show_qb_learn
from screens.quantum_security import show_quantum_security_dashboard, show_quantum_demo
from utils.auth_manager import auth_manager
from utils.quantum_session_manager import quantum_session_manager

st.set_page_config(page_title="QBank", layout="centered")

# Enhanced session initialization with persistent authentication
def initialize_session():
    """Initialize session state with persistent authentication check"""
    if "token" not in st.session_state:
        st.session_state.token = None
    if "account_type" not in st.session_state:
        st.session_state.account_type = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "full_name" not in st.session_state:
        st.session_state.full_name = None
    if "account_id" not in st.session_state:
        st.session_state.account_id = None
    
    # Debug: Show current session state (only for admin users)
    if auth_manager.is_debug_enabled():
        st.write("🔍 DEBUG - Authentication Settings:")
        st.write(f"Token expiration: {auth_manager.token_expire_minutes} minutes")
        st.write(f"Session file: {auth_manager.session_file}")
        st.write("🔍 DEBUG - Current session state:")
        st.write(f"Token: {st.session_state.token[:20] + '...' if st.session_state.token else 'None'}")
        st.write(f"Account type: {st.session_state.account_type}")
        st.write(f"Auth manager session: {auth_manager._get_stored_session() is not None}")
        
        # Show session age if available
        session_data = auth_manager._get_stored_session()
        if session_data:
            session_age_minutes = (time.time() - session_data.get("timestamp", 0)) / 60
            st.write(f"Session age: {session_age_minutes:.1f} minutes")
            st.write(f"Session expires in: {auth_manager.token_expire_minutes - session_age_minutes:.1f} minutes")
    
    # Check for persistent session on every page load
    if st.session_state.token is None:
        # Try to restore session from persistent storage
        try:
            if auth_manager.validate_session():
                user_data = auth_manager.get_current_user()
                if user_data:
                    st.session_state.token = auth_manager.get_token()
                    st.session_state.account_type = user_data.get("account_type")
                    st.session_state.username = user_data.get("username")
                    st.session_state.full_name = user_data.get("full_name")
                    st.session_state.account_id = user_data.get("account_id")
                    
                    # Debug: Show successful restoration (only for admin users)
                    if auth_manager.is_debug_enabled():
                        st.success(f"✅ Session restored for {user_data.get('username')}")
            else:
                # Debug: Show why validation failed (only for admin users)
                if auth_manager.is_debug_enabled():
                    st.warning("❌ Session validation failed")
        except Exception as e:
            # Debug: Show any errors (only for admin users)
            if auth_manager.is_debug_enabled():
                st.error(f"🚨 Session restoration error: {e}")

# Initialize session state
initialize_session()

# Add debug toggle in sidebar (only for admin users)
if st.session_state.token and st.session_state.account_type == "admin":
    if st.sidebar.checkbox("Enable Auth Debug", key="debug_auth"):
        st.sidebar.write("🔍 Debug mode enabled")

# sidebar navigation
if st.session_state.token is None:
    page = st.sidebar.radio("Go to", ["Login", "Register", "QB-Learn"])
    if page == "Login":
        show_login()
    elif page == "Register":
        show_register()
    elif page == "QB-Learn":
        show_qb_learn()
else:
    # Show only relevant dashboard based on account type
    if st.session_state.account_type == "customer":
        dashboard_pages = ["Home", "Transactions", "Quantum Security"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            # Clean up quantum session before logout
            if st.session_state.token:
                quantum_session_manager.cleanup_session(st.session_state.token)
            auth_manager.logout()  # Use auth_manager for proper logout
        if page == "Transactions":
            show_customer_transactions()
        elif page == "Quantum Security":
            show_quantum_security_dashboard()
    elif st.session_state.account_type == "employee":
        dashboard_pages = ["Home", "Customers", "Transactions"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            # Clean up quantum session before logout
            if st.session_state.token:
                quantum_session_manager.cleanup_session(st.session_state.token)
            auth_manager.logout()  # Use auth_manager for proper logout
        if page == "Home":
            show_employee_dashboard()
        elif page == "Customers":
            from screens.employee_customers import show_employee_customers
            show_employee_customers()
        elif page == "Transactions":
            show_employee_transactions()
    elif st.session_state.account_type == "admin":
        dashboard_pages = ["Home", "Quantum Demo"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            # Clean up quantum session before logout
            if st.session_state.token:
                quantum_session_manager.cleanup_session(st.session_state.token)
            auth_manager.logout()  # Use auth_manager for proper logout
        if page == "Home":
            show_admin_dashboard()
        elif page == "Quantum Demo":
            show_quantum_demo()
    else:
        # logout if account type is unknown
        auth_manager.logout()  # Use auth_manager for proper logout
