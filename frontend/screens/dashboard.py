# frontend/screens/dashboard.py
import streamlit as st
import requests
from utils.config import get_cached_quantum_safe_setting
from utils.quantum_session_manager import quantum_session_manager
from screens import employee_experiments

API_URL = "http://localhost:8000"

def show_employee_dashboard():
    st.title("Employee Dashboard")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Transactions", "Quantum Security Demo", "ML Experiments"])
    
    with tab1:
        st.subheader("All System Transactions")
        st.info("Employee Access: Transaction amounts are decrypted for monitoring purposes")
        
        # Date filter
        from datetime import datetime, timedelta
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input(
                "From Date", 
                value=datetime.now().date() - timedelta(days=7),
                key="dash_from_date"
            )
        with col2:
            to_date = st.date_input(
                "To Date", 
                value=datetime.now().date(),
                key="dash_to_date"
            )
        
        if st.button("Load Transactions", use_container_width=True):
            st.rerun()
        
        # Load all transactions with decrypted amounts
        try:
            params = {}
            if from_date:
                params["from_date"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to_date"] = to_date.strftime("%Y-%m-%d")
            
            response = requests.get(f"{API_URL}/employee/transactions/all", headers=headers, params=params)
            if response.status_code == 200:
                transactions = response.json()
                if transactions:
                    # Create DataFrame with better column names
                    import pandas as pd
                    df = pd.DataFrame(transactions)
                    # Convert to datetime for proper sorting, then sort by latest first
                    df["created_at"] = pd.to_datetime(df["created_at"])
                    df = df.sort_values("created_at", ascending=False)
                    # Convert back to string for display
                    df["created_at"] = df["created_at"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Rename columns for better display
                    display_df = df.rename(columns={
                        "transaction_id": "ID",
                        "from_account_id": "From Account",
                        "to_account_id": "To Account",
                        "amount": "Amount ($)",
                        "created_at": "Date & Time",
                        "is_fraud": "Fraud Alert"
                    })
                    
                    # Add Created By column if account_id exists, otherwise use from_account_id as fallback
                    if "account_id" in df.columns:
                        display_df["Created By"] = df["account_id"]
                    else:
                        display_df["Created By"] = df["from_account_id"]  # Fallback to sender account
                    
                    # Format amount column
                    if "Amount ($)" in display_df.columns:
                        display_df["Amount ($)"] = display_df["Amount ($)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Encrypted")
                    
                    st.dataframe(
                        display_df[["ID", "From Account", "To Account", "Amount ($)", "Created By", "Date & Time", "Fraud Alert"]], 
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("### Transaction Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(transactions))
                    
                    with col2:
                        fraud_count = len([tx for tx in transactions if tx.get("is_fraud", False)])
                        st.metric("Fraud Alerts", fraud_count)
                    
                    with col3:
                        amounts = [tx.get("amount", 0) for tx in transactions if tx.get("amount") is not None]
                        if amounts:
                            total_volume = sum(amounts)
                            st.metric("Total Volume", f"${total_volume:.2f}")
                        else:
                            st.metric("Total Volume", "N/A")
                    
                    with col4:
                        if amounts:
                            avg_amount = sum(amounts) / len(amounts)
                            st.metric("Average Amount", f"${avg_amount:.2f}")
                        else:
                            st.metric("Average Amount", "N/A")
                    
                else:
                    st.info("No transactions found for the selected date range")
                    
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
        st.subheader("Post-Quantum Cryptography Security")
        st.info("This tab demonstrates the quantum-resistant security features used in the banking system.")
        
        # Post-quantum cryptography information
        st.markdown("""
        ### Post-Quantum Cryptographic Algorithms
        
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
        
        ### Security Features
        
        - **Quantum-Resistant**: Algorithms are designed to resist quantum computer attacks
        - **Forward Secrecy**: New keys generated for each transaction
        - **Digital Signatures**: All transactions are cryptographically signed
        - **Encrypted Storage**: Customer balances are encrypted at rest
        """)
        
        if st.button("Demo: Key Generation", use_container_width=True):
            with st.spinner("Generating post-quantum keys..."):
                try:
                    response = requests.post(f"{API_URL}/transactions/quantum/demo-key-generation", timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("Post-quantum keys generated successfully!")
                            
                            # Display key generation results
                            kyber_info = result["algorithms"]["kyber"]
                            dilithium_info = result["algorithms"]["dilithium"]
                            
                            with st.expander("**Key Generation Details**"):
                                st.markdown("### CRYSTALS-Kyber (Key Encapsulation)")
                                col_k1, col_k2 = st.columns(2)
                                with col_k1:
                                    st.metric("Public Key Size", f"{kyber_info['public_key_size']:,} bytes")
                                    st.metric("Private Key Size", f"{kyber_info['private_key_size']:,} bytes")
                                with col_k2:
                                    st.metric("Security Level", kyber_info['security_level'])
                                    st.metric("Generation Time", f"{result['performance']['total_time']}s")
                                
                                st.markdown("### CRYSTALS-Dilithium (Digital Signatures)")
                                col_d1, col_d2 = st.columns(2)
                                with col_d1:
                                    st.metric("Public Key Size", f"{dilithium_info['public_key_size']:,} bytes")
                                    st.metric("Private Key Size", f"{dilithium_info['private_key_size']:,} bytes")
                                with col_d2:
                                    st.metric("Security Level", dilithium_info['security_level'])
                                    st.metric("Library", result['implementation_details']['library'])
                                
                                st.markdown("### Security Features")
                                st.info(f"**Quantum Resistance**: {result['implementation_details']['quantum_resistance']}")
                                st.info(f"**Standards**: {result['implementation_details']['standardization']}")
                                
                                # Show features
                                col_feat1, col_feat2 = st.columns(2)
                                with col_feat1:
                                    st.markdown("**Kyber Features:**")
                                    for feature in result['performance']['kyber_features']:
                                        st.markdown(f"• {feature}")
                                with col_feat2:
                                    st.markdown("**Dilithium Features:**")
                                    for feature in result['performance']['dilithium_features']:
                                        st.markdown(f"• {feature}")
                        else:
                            st.error(f"Key generation failed: {result.get('message', 'Unknown error')}")
                    else:
                        st.error(f"Server error: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.button("Demo: Digital Signature", use_container_width=True):
            with st.spinner("Creating and verifying digital signature..."):
                try:
                    response = requests.post(f"{API_URL}/transactions/quantum/demo-digital-signature", timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("Digital signature demo completed!")
                            
                            # Display signature results
                            sig_details = result["signature_details"]
                            transaction = result["transaction_data"]
                            
                            with st.expander("**Digital Signature Details**"):
                                st.markdown("### Transaction Data Signed")
                                st.json(transaction)
                                
                                st.markdown("### Signature Information")
                                col_s1, col_s2 = st.columns(2)
                                with col_s1:
                                    st.metric("Algorithm", sig_details['algorithm'])
                                    st.metric("Signature Size", f"{sig_details['signature_size']:,} bytes")
                                with col_s2:
                                    st.metric("Verification", sig_details['verification_result'])
                                    st.metric("Total Time", f"{result['performance']['total_time']}s")
                                
                                st.markdown("### Security Properties")
                                security = result["security_properties"]
                                for prop, desc in security.items():
                                    st.markdown(f"**{prop.replace('_', ' ').title()}**: {desc}")
                                
                                st.markdown("### Real-World Applications")
                                for app in result["real_world_applications"]:
                                    st.markdown(f"• {app}")
                        else:
                            st.error(f"Signature demo failed: {result.get('message', 'Unknown error')}")
                    else:
                        st.error(f"Server error: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Security status
        st.markdown("""
        ### Security Status
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Encryption", "Active", help="AES-256 encryption active")
        with col2:
            st.metric("Key Exchange", "Quantum-Safe", help="CRYSTAL-Kyber in use")
        with col3:
            st.metric("Signatures", "Post-Quantum", help="CRYSTAL-DILITHIUM active")
    
    with tab3:
        employee_experiments.main()
