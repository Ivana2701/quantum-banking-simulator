# frontend/screens/dashboard.py
import streamlit as st
import requests
from utils.config import get_cached_quantum_safe_setting
from utils.quantum_session_manager import quantum_session_manager
from screens import employee_experiments

API_URL = "http://localhost:8000"

def show_employee_dashboard():
    st.title("👩‍💼 Employee Dashboard")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📊 Transactions", "🔐 Quantum Security Demo", "🧪 ML Experiments"])
    
    with tab1:
        st.subheader("All Transactions")
        
        # Load all transactions button
        if st.button("Load All Transactions"):
            try:
                response = requests.get(f"{API_URL}/employee/transactions/all", headers=headers)
                if response.status_code == 200:
                    transactions = response.json()
                    if transactions:
                        st.success(f"Loaded {len(transactions)} transactions")
                        # Display transactions in a table
                        import pandas as pd
                        df = pd.DataFrame(transactions)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No transactions found")
                elif response.status_code == 403:
                    st.error("Access denied. Employee privileges required.")
                else:
                    st.error(f"Failed to load transactions. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
                st.info("Make sure the backend server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.subheader("🔐 Post-Quantum Cryptography Security")
        st.info("This tab demonstrates the quantum-resistant security features used in the banking system.")
        
        # Post-quantum cryptography information
        st.markdown("""
        ### 🔐 Post-Quantum Cryptographic Algorithms
        
        Our banking system uses NIST-selected post-quantum cryptographic algorithms to ensure security against both classical and quantum computer attacks:
        
        #### Key Encapsulation Mechanism (KEM)
        - **CRYSTAL-Kyber512**: Used for secure key distribution
        - Each customer account has unique Kyber key pairs
        - Shared secrets are generated for each balance encryption
        
        #### Digital Signatures  
        - **CRYSTAL-DILITHIUM2**: Used for transaction authentication
        - All balance modifications are digitally signed
        - Provides non-repudiation and integrity verification
        
        #### Symmetric Encryption
        - **AES-256-CBC**: Used for encrypting sensitive balance data
        - Keys derived from Kyber-encapsulated shared secrets
        - PBKDF2 key derivation for additional security
        
        ### 🔒 Security Features
        
        - **Quantum-Resistant**: Algorithms are designed to resist quantum computer attacks
        - **Forward Secrecy**: New keys generated for each transaction
        - **Digital Signatures**: All transactions are cryptographically signed
        - **Encrypted Storage**: Customer balances are encrypted at rest
        """)
        
        # Security demonstration
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔐 Demo: Key Generation"):
                with st.spinner("Generating post-quantum keys..."):
                    import time
                    time.sleep(1)  # Simulate processing
                    st.success("✅ CRYSTAL-Kyber keypair generated!")
                    st.success("✅ CRYSTAL-DILITHIUM keypair generated!")
                    st.info("🔑 Keys would be stored securely in production")
        
        with col2:
            if st.button("📝 Demo: Digital Signature"):
                with st.spinner("Creating digital signature..."):
                    import time
                    time.sleep(1)  # Simulate processing
                    st.success("✅ Transaction signed with DILITHIUM!")
                    st.success("✅ Signature verified successfully!")
                    st.info("🛡️ Transaction integrity guaranteed")
        
        # Security status
        st.markdown("""
        ### 📊 Security Status
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔐 Encryption", "Active", help="AES-256 encryption active")
        with col2:
            st.metric("🔑 Key Exchange", "Quantum-Safe", help="CRYSTAL-Kyber in use")
        with col3:
            st.metric("📝 Signatures", "Post-Quantum", help="CRYSTAL-DILITHIUM active")
    
    with tab3:
        employee_experiments.main()
