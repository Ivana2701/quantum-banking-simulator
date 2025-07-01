# frontend/screens/login.py
import streamlit as st
from utils.auth_manager import auth_manager

def show_login():
    st.title("🔐 Quantum Banking Login")
    
    # Check if already authenticated
    if auth_manager.is_authenticated():
        user_data = auth_manager.get_current_user()
        st.success(f"Already logged in as {user_data.get('full_name', 'User')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue to Dashboard"):
                st.rerun()
        with col2:
            if st.button("Logout"):
                auth_manager.logout()
        return
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
                return
            
            with st.spinner("Authenticating..."):
                if auth_manager.login(username, password):
                    st.rerun()  # Refresh page after successful login
    
    # Additional info
    st.markdown("---")
    st.markdown("""
    **🛡️ Security Features:**
    - JWT-based authentication with role information
    - Persistent sessions across page refreshes
    - Role-based access control
    - Post-quantum cryptography for data protection
    """)
