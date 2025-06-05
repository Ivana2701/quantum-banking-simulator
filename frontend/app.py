import streamlit as st
from components.login import show_login
from components.dashboard import show_dashboard
from components.model_evaluation import show_model_evaluation
from components.transactions import show_transactions

st.set_page_config(page_title="Quantum Banking App")

# Session state setup for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login()
else:
    # Explicit menu selection
    menu_choice = st.sidebar.selectbox("Menu", ["Overview", "Transactions", "Model Evaluation"])

    if menu_choice == "Overview":
        show_dashboard()
    elif menu_choice == "Transactions":
        show_transactions()
    elif menu_choice == "Model Evaluation":
        show_model_evaluation()
