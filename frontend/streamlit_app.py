import streamlit as st
from screens.login import show_login
from screens.register import show_register
from screens.dashboard import show_customer_dashboard, show_employee_dashboard
from screens.admin_dashboard import show_admin_dashboard

st.set_page_config(page_title="QBank", layout="centered")

# initialize session state
if "token" not in st.session_state:
    st.session_state.token = None
if "account_type" not in st.session_state:
    st.session_state.account_type = None

# sidebar navigation
if st.session_state.token is None:
    page = st.sidebar.radio("Go to", ["Login", "Register"])
    if page == "Login":
        show_login()
    else:
        show_register()
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
            show_customer_dashboard()
    elif st.session_state.account_type == "employee":
        dashboard_pages = ["Home"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.session_state.account_type = None
            st.rerun()
        if page == "Home":
            show_employee_dashboard()
    elif st.session_state.account_type == "admin":
        dashboard_pages = ["Home"]
        page = st.sidebar.radio("Dashboard", dashboard_pages)
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.session_state.account_type = None
            st.rerun()
        if page == "Home":
            show_admin_dashboard()
    else:
        # logout if account type is unknown
        st.session_state.token = None
        st.session_state.account_type = None
        st.rerun()
