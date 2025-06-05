import streamlit as st

def show_dashboard():
    st.title("Quantum Secure Dashboard")

    st.subheader("Account Overview")
    st.metric("Balance", "$15,000")
