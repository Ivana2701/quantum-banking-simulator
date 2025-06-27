import streamlit as st
import requests
from utils.api_helpers import fetch_balance_with_retry, encrypt_demo_transaction
import streamlit as st
import os
from utils.api_helpers import fetch_balance_with_retry

def load_custom_css(file_path):
    abs_path = os.path.join(os.path.dirname(__file__), "..", file_path)
    with open(abs_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def customer_dashboard():
    load_custom_css("styles/popup_modal.css")

    if "show_popup" not in st.session_state:
        st.session_state["show_popup"] = True

    # Manage sidebar visibility based on popup state
    if st.session_state["show_popup"]:
        st.markdown(
            "<style>[data-testid='stSidebar'] { visibility: hidden; }</style>",
            unsafe_allow_html=True
        )

        # Popup overlay and content
        st.markdown("""
        <div class="overlay"></div>
        <div class="modal">
            <h2>🛡️ Quantum Security Overview</h2>
            <p>This demo shows how we use:</p>
            <ul>
                <li><strong>Quantum Key Distribution (BB84)</strong></li>
                <li><strong>Post-Quantum Cryptography (Kyber)</strong></li>
                <li><strong>AES Encryption</strong></li>
            </ul>
            <p>To protect your financial data from quantum threats and modern fraud.</p>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit buttons BELOW the modal clearly
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❌ Close", key="close_modal"):
                st.session_state["show_popup"] = False
                st.rerun()

        with col2:
            if st.button("🚀 Try Live DEMO", key="live_demo_modal"):
                st.session_state["show_popup"] = False
                st.switch_page("pages/1_Quantum_Safe_Banking_Simulation.py")

        return  # Do NOT continue rendering below until popup is closed

    # Reveal sidebar after popup closes
    st.markdown(
        "<style>[data-testid='stSidebar'] { visibility: visible; }</style>",
        unsafe_allow_html=True
    )
    # Main dashboard content
    st.title("Customer Dashboard")

    customer_id = st.session_state.get("customer_id")
    if not customer_id:
        st.error("Unauthorized. Please log in.")
        st.stop()
        try:
        balance = fetch_balance_with_retry(customer_id)
        st.metric("Account Balance", f"${float(balance):,.2f}")
    except Exception as e:
        st.warning(f"Could not load account balance: {e}")
        st.metric("Account Balance", "$0.00")
    recipient = st.text_input("Recipient Account ID")
    amount    = st.number_input("Amount", min_value=0.01, format="%.2f")
    mode      = st.radio("Select Transaction Mode:", ["Demo Mode", "Real Mode"])

    if st.button("Send Transaction"):
        # 1) basic validation
        if not recipient:
            st.error("Recipient Account ID is required.")
            st.stop()
        if int(recipient) == customer_id:
            st.error("You cannot send money to yourself.")
            st.stop()

        with st.spinner("Processing…"):
            try:
                if mode == "Demo Mode":
                    # (a) locally encrypt and save to JSON
                    encrypted_data, bb84_key = encrypt_demo_transaction(
                        sender_id=customer_id,
                        recipient_id=int(recipient),
                        amount=float(amount),
                    )
                    resp = requests.post(
                        f"{API_URL}/encrypt_transaction_demo/",
                        json={
                            "encrypted_data": encrypted_data,
                            "sender_customer_id": customer_id,
                            "recipient_customer_id": int(recipient),
                            "amount": float(amount),
                            "bb84_key": bb84_key,
                        },
                        timeout=10
                    )
                else:
                    # (b) real‐mode: encrypt & POST to real endpoint
                    # assume you have an AES/Kyber wrapper, or call your encrypt() here:
                    ciphertext = encrypt_demo_transaction(  # or your real encrypt function
                        sender_id=customer_id,
                        recipient_id=int(recipient),
                        amount=float(amount),
                    )[0]

                    resp = requests.post(
                        f"{API_URL}/encrypt_transaction_real/",
                        json={
                            "encrypted_data": ciphertext,
                            "sender_customer_id": customer_id,
                            "recipient_account_id": int(recipient),
                            "amount": float(amount),
                        },
                        timeout=10
                    )

                # check the HTTP response
                if resp.ok:
                    data = resp.json()
                    st.success(
                        mode == "Demo Mode"
                        and "Transaction sent successfully! (demo only) 🎉"
                        or f"✅ Stored in DB as ID {data['transaction_id']}"
                    )
                else:
                    st.error(f"🚨 Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Unexpected error: {e}")
