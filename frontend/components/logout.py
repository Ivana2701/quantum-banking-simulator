# components/logout.py
import streamlit as st

def show_logout():
    """
    Renders a logout button in the sidebar. When clicked, it clears
    the relevant session state and reruns the app so the login screen
    reappears.
    """
    if st.sidebar.button("Logout"):
        # Clear only the keys we used for authentication
        for key in ("logged_in", "role"):
            if key in st.session_state:
                del st.session_state[key]
        # Restart the script from the top
        st.rerun()

