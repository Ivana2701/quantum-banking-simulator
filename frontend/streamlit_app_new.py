# frontend/streamlit_app_new.py
import streamlit as st
from utils.auth_manager import auth_manager
from screens.login_new import show_login
from screens.register import show_register
from screens.dashboard import show_customer_dashboard, show_employee_dashboard
from screens.admin_dashboard import show_admin_dashboard

# Page configuration
st.set_page_config(
    page_title="QBank - Quantum Secure Banking",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

def main():
    """Main application with enhanced authentication"""
    
    # Initialize session state if needed
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    
    # Check authentication status
    is_authenticated = auth_manager.is_authenticated()
    
    if not is_authenticated:
        # Show login/register pages
        st.sidebar.title("🏦 QBank")
        page = st.sidebar.radio("Navigation", ["Login", "Register"])
        
        if page == "Login":
            show_login()
        else:
            show_register()
    else:
        # User is authenticated, show appropriate dashboard
        user_data = auth_manager.get_current_user()
        account_type = user_data.get("account_type")
        full_name = user_data.get("full_name", "User")
        
        # Sidebar with user info
        st.sidebar.title("🏦 QBank")
        st.sidebar.success(f"Welcome, {full_name}!")
        st.sidebar.info(f"Role: {account_type.title()}")
        
        # Navigation based on role
        if account_type == "customer":
            show_customer_interface()
        elif account_type == "employee":
            show_employee_interface()
        elif account_type == "admin":
            show_admin_interface()
        else:
            st.error("Unknown account type. Please contact administrator.")
            if st.button("Logout"):
                auth_manager.logout()

def show_customer_interface():
    """Customer interface with role-specific navigation"""
    pages = ["Dashboard", "Transactions", "Profile"]
    page = st.sidebar.radio("Navigation", pages)
    
    if st.sidebar.button("Logout", key="customer_logout"):
        auth_manager.logout()
    
    if page == "Dashboard":
        show_customer_dashboard()
    elif page == "Transactions":
        from screens import transactions
        transactions.show_transactions()
    elif page == "Profile":
        show_profile_page()

def show_employee_interface():
    """Employee interface with role-specific navigation"""
    pages = ["Dashboard", "Customers", "Transactions", "Reports"]
    page = st.sidebar.radio("Navigation", pages)
    
    if st.sidebar.button("Logout", key="employee_logout"):
        auth_manager.logout()
    
    if page == "Dashboard":
        show_employee_dashboard()
    elif page == "Customers":
        from screens.employee_customers import show_employee_customers
        show_employee_customers()
    elif page == "Transactions":
        show_employee_transactions()
    elif page == "Reports":
        show_employee_reports()

def show_admin_interface():
    """Admin interface with full access"""
    pages = ["Dashboard", "User Management", "System Settings", "Reports", "Security"]
    page = st.sidebar.radio("Navigation", pages)
    
    if st.sidebar.button("Logout", key="admin_logout"):
        auth_manager.logout()
    
    if page == "Dashboard":
        show_admin_dashboard()
    elif page == "User Management":
        show_user_management()
    elif page == "System Settings":
        show_system_settings()
    elif page == "Reports":
        show_admin_reports()
    elif page == "Security":
        show_security_settings()

def show_profile_page():
    """User profile page"""
    st.title("👤 Profile")
    user_data = auth_manager.get_current_user()
    
    if user_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Username:** {user_data.get('username')}")
            st.info(f"**Full Name:** {user_data.get('full_name')}")
            st.info(f"**Account Type:** {user_data.get('account_type', '').title()}")
        
        with col2:
            st.info(f"**Account ID:** {user_data.get('account_id')}")
            st.info(f"**Status:** {user_data.get('status', 'Active')}")
            st.info(f"**Created:** {user_data.get('created_at', 'N/A')}")
    
    if st.button("Refresh Profile"):
        if auth_manager.validate_session():
            st.success("Profile refreshed successfully!")
            st.rerun()
        else:
            st.error("Session expired. Please log in again.")

def show_employee_transactions():
    """Employee transaction management"""
    st.title("💳 Transaction Management")
    st.info("Employee transaction management interface coming soon...")

def show_employee_reports():
    """Employee reports"""
    st.title("📊 Reports")
    st.info("Employee reporting interface coming soon...")

def show_user_management():
    """Admin user management"""
    st.title("👥 User Management")
    
    # Add role-based access check
    if not auth_manager.has_role("admin"):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.info("User management interface with role updates coming soon...")

def show_system_settings():
    """Admin system settings"""
    st.title("⚙️ System Settings")
    
    if not auth_manager.has_role("admin"):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.info("System settings interface coming soon...")

def show_admin_reports():
    """Admin reports"""
    st.title("📊 Administrative Reports")
    
    if not auth_manager.has_role("admin"):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.info("Administrative reporting interface coming soon...")

def show_security_settings():
    """Admin security settings"""
    st.title("🔒 Security Settings")
    
    if not auth_manager.has_role("admin"):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.info("Security settings interface coming soon...")

if __name__ == "__main__":
    main()
