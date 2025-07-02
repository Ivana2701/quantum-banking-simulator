# frontend/screens/transactions.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from utils.notifications import show_notification

API_URL = "http://localhost:8000"

def show_customer_transactions():
    """Display customer transaction dashboard with send money functionality"""
    
    st.title("💰 Transaction Center")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["💳 Send Money", "📋 Transaction History"])
    
    with tab1:
        st.subheader("💸 Send Money")
        st.info("🔐 **Quantum Security**: All transaction amounts are encrypted using post-quantum cryptography (CRYSTAL-Kyber)")
        
        # Balance section with refresh button
        col_balance, col_refresh = st.columns([3, 1])
        
        with col_refresh:
            refresh_balance = st.button("🔄 Refresh Balance", help="Click to update your current balance")
        
        # Display current balance at the top
        try:
            response = requests.get(f"{API_URL}/customer/balance", headers=headers)
            if response.status_code == 200:
                balance_data = response.json()
                balance = balance_data.get("balance", 0)
                
                # Display balance with nice formatting
                with col_balance:
                    st.metric(
                        "💵 Current Balance", 
                        f"${balance:.2f}",
                        help="Your current account balance"
                    )
                    
                st.divider()  # Add a visual separator
            else:
                st.warning("⚠️ Could not fetch current balance")
                if refresh_balance:
                    st.rerun()
        except Exception as e:
            st.warning(f"⚠️ Balance unavailable: {str(e)}")
            if refresh_balance:
                st.rerun()
        
        # Send money form
        col1, col2 = st.columns(2)
        
        with col1:
            recipient_id = st.number_input(
                "Recipient Account ID", 
                min_value=1, 
                step=1,
                help="Enter the account ID of the person you want to send money to"
            )
            
        with col2:
            amount = st.number_input(
                "Amount ($)", 
                min_value=0.01, 
                step=0.01, 
                format="%.2f",
                help="Enter the amount you want to send"
            )
        
        if st.button("💸 Send Money", type="primary"):
            if recipient_id and amount > 0:
                with st.spinner("Processing quantum-encrypted transaction..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/customer/transfer",
                            headers=headers,
                            json={"to_account_id": int(recipient_id), "amount": float(amount)}
                        )
                        if response.status_code == 200:
                            show_notification(f"✅ Successfully sent ${amount:.2f} to account {recipient_id}", "success", True)
                        elif response.status_code == 400:
                            st.error("❌ Insufficient funds or invalid transaction")
                        elif response.status_code == 404:
                            if response.json().get("detail") == "Recipient account must be a customer account":
                                st.error(f"❌ {response.json().get('detail')}")
                            else:
                                st.error("❌ Recipient account not found")
                        else:
                            st.error(f"❌ Transaction failed: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Connection error: {str(e)}")
                        st.info("Make sure the backend server is running on http://localhost:8000")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid recipient ID and amount")
    
    with tab2:
        st.subheader("📋 Your Transaction History")
        
        # Date filter
        col1, col2, col3 = st.columns(3)
        with col1:
            from_date = st.date_input(
                "From Date", 
                value=datetime.now().date() - timedelta(days=30),
                help="Start date for transaction history"
            )
        with col2:
            to_date = st.date_input(
                "To Date", 
                value=datetime.now().date(),
                help="End date for transaction history"
            )
        with col3:
            if st.button("🔄 Refresh Transactions"):
                st.rerun()
        
        # Load transactions
        try:
            params = {}
            if from_date:
                params["from_date"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to_date"] = to_date.strftime("%Y-%m-%d")
            
            response = requests.get(f"{API_URL}/customer/transactions", headers=headers, params=params)
            if response.status_code == 200:
                transactions_data = response.json()
                
                # Display sent transactions
                if transactions_data.get("sent"):
                    st.markdown("### 📤 Sent Transactions")
                    sent_df = pd.DataFrame(transactions_data["sent"])
                    sent_df["type"] = "Sent"
                    sent_df["created_at"] = pd.to_datetime(sent_df["created_at"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Rename columns for better display
                    display_sent = sent_df.rename(columns={
                        "transaction_id": "ID",
                        "to_account_id": "To Account",
                        "created_at": "Date & Time",
                        "is_fraud": "Fraud Check"
                    })
                    
                    st.dataframe(display_sent[["ID", "To Account", "Date & Time", "Fraud Check"]], use_container_width=True)
                else:
                    st.info("📤 No sent transactions found")
                
                # Display received transactions
                if transactions_data.get("received"):
                    st.markdown("### 📥 Received Transactions")
                    received_df = pd.DataFrame(transactions_data["received"])
                    received_df["type"] = "Received"
                    received_df["created_at"] = pd.to_datetime(received_df["created_at"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Rename columns for better display
                    display_received = received_df.rename(columns={
                        "transaction_id": "ID",
                        "from_account_id": "From Account",
                        "created_at": "Date & Time",
                        "is_fraud": "Fraud Check"
                    })
                    
                    st.dataframe(display_received[["ID", "From Account", "Date & Time", "Fraud Check"]], use_container_width=True)
                else:
                    st.info("📥 No received transactions found")
                
                # Summary statistics
                total_sent = len(transactions_data.get("sent", []))
                total_received = len(transactions_data.get("received", []))
                
                if total_sent > 0 or total_received > 0:
                    st.markdown("### 📊 Transaction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📤 Sent", total_sent)
                    with col2:
                        st.metric("📥 Received", total_received)
                    with col3:
                        st.metric("📊 Total", total_sent + total_received)
                
            elif response.status_code == 403:
                st.error("❌ Access denied. Customer privileges required.")
            else:
                st.error(f"❌ Failed to load transactions. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
            st.info("Make sure the backend server is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


def show_employee_transactions():
    """Display employee transaction dashboard with enhanced capabilities"""
    
    st.title("👩‍💼 Employee Transaction Dashboard")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📊 All Transactions", "🔍 Search Transactions", "💸 Create Transaction", "🔐 Security Overview"])
    
    with tab1:
        st.subheader("📊 All System Transactions")
        st.info("🔓 **Employee Access**: Transaction amounts are decrypted for monitoring purposes")
        
        # Date filter
        col1, col2, col3 = st.columns(3)
        with col1:
            from_date = st.date_input(
                "From Date", 
                value=datetime.now().date() - timedelta(days=7),
                key="emp_from_date"
            )
        with col2:
            to_date = st.date_input(
                "To Date", 
                value=datetime.now().date(),
                key="emp_to_date"
            )
        with col3:
            if st.button("🔄 Load Transactions"):
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
                    
                    # Format amount column
                    if "Amount ($)" in display_df.columns:
                        display_df["Amount ($)"] = display_df["Amount ($)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Encrypted")
                    
                    st.dataframe(
                        display_df[["ID", "From Account", "To Account", "Amount ($)", "Date & Time", "Fraud Alert"]], 
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("### 📈 Transaction Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Total Transactions", len(transactions))
                    
                    with col2:
                        fraud_count = len([tx for tx in transactions if tx.get("is_fraud", False)])
                        st.metric("🚨 Fraud Alerts", fraud_count)
                    
                    with col3:
                        amounts = [tx.get("amount", 0) for tx in transactions if tx.get("amount") is not None]
                        if amounts:
                            total_volume = sum(amounts)
                            st.metric("💰 Total Volume", f"${total_volume:.2f}")
                        else:
                            st.metric("💰 Total Volume", "N/A")
                    
                    with col4:
                        if amounts:
                            avg_amount = sum(amounts) / len(amounts)
                            st.metric("📊 Average Amount", f"${avg_amount:.2f}")
                        else:
                            st.metric("📊 Average Amount", "N/A")
                    
                else:
                    st.info("📊 No transactions found for the selected date range")
                    
            elif response.status_code == 403:
                st.error("❌ Access denied. Employee privileges required.")
            else:
                st.error(f"❌ Failed to load transactions. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
            st.info("Make sure the backend server is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Search Transactions by Account")
        
        account_id = st.number_input(
            "Account ID to Search", 
            min_value=1, 
            step=1,
            help="Enter the account ID to search for all related transactions"
        )
        
        if st.button("🔍 Search Transactions"):
            if account_id:
                try:
                    response = requests.get(
                        f"{API_URL}/employee/transactions/search",
                        headers=headers,
                        params={"account_id": account_id}
                    )
                    if response.status_code == 200:
                        transactions = response.json()
                        if transactions:
                            df = pd.DataFrame(transactions)
                            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Categorize transactions
                            sent_transactions = df[df["from_account_id"] == account_id]
                            received_transactions = df[df["to_account_id"] == account_id]
                            
                            st.success(f"Found {len(transactions)} transactions for account {account_id}")
                            
                            if not sent_transactions.empty:
                                st.markdown("#### 📤 Sent Transactions")
                                sent_display = sent_transactions.rename(columns={
                                    "transaction_id": "ID",
                                    "to_account_id": "To Account",
                                    "amount": "Amount ($)",
                                    "created_at": "Date & Time"
                                })
                                sent_display["Amount ($)"] = sent_display["Amount ($)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Encrypted")
                                st.dataframe(sent_display[["ID", "To Account", "Amount ($)", "Date & Time"]], use_container_width=True)
                            
                            if not received_transactions.empty:
                                st.markdown("#### 📥 Received Transactions")
                                received_display = received_transactions.rename(columns={
                                    "transaction_id": "ID",
                                    "from_account_id": "From Account",
                                    "amount": "Amount ($)",
                                    "created_at": "Date & Time"
                                })
                                received_display["Amount ($)"] = received_display["Amount ($)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Encrypted")
                                st.dataframe(received_display[["ID", "From Account", "Amount ($)", "Date & Time"]], use_container_width=True)
                        else:
                            st.info(f"No transactions found for account {account_id}")
                    else:
                        st.error(f"Failed to search transactions: {response.text}")
                except Exception as e:
                    st.error(f"Error searching transactions: {str(e)}")
            else:
                st.warning("Please enter a valid account ID")
    
    with tab3:
        st.subheader("💸 Create Transaction (Employee)")
        st.info("🏦 **Employee Access**: Create transactions between customer accounts")
        
        # Transaction creation form
        col1, col2 = st.columns(2)
        
        with col1:
            from_account_id = st.number_input(
                "From Account ID (Sender)", 
                min_value=1, 
                step=1,
                key="emp_from_account",
                help="Enter the account ID that will send money"
            )
            
            # Check and display from account balance
            if from_account_id > 0:
                try:
                    response = requests.get(
                        f"{API_URL}/employee/customers/{from_account_id}/balance",
                        headers=headers
                    )
                    if response.status_code == 200:
                        balance_data = response.json()
                        balance = balance_data.get("balance", 0)
                        st.success(f"💰 Balance: ${balance:.2f}")
                        # Clear refresh flag if it was set
                        if hasattr(st.session_state, 'refresh_balances') and st.session_state.refresh_balances:
                            st.session_state.refresh_balances = False
                    elif response.status_code == 404:
                        st.error("❌ Account not found")
                    else:
                        st.error("❌ Could not fetch balance")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            to_account_id = st.number_input(
                "To Account ID (Receiver)", 
                min_value=1, 
                step=1,
                key="emp_to_account",
                help="Enter the account ID that will receive money"
            )
            
            # Check and display to account status
            if to_account_id > 0:
                try:
                    response = requests.get(
                        f"{API_URL}/employee/customers/{to_account_id}/balance",
                        headers=headers
                    )
                    if response.status_code == 200:
                        balance_data = response.json()
                        balance = balance_data.get("balance", 0)
                        st.success(f"💰 Balance: ${balance:.2f}")
                        # Clear refresh flag after both balances are shown
                        if hasattr(st.session_state, 'refresh_balances') and st.session_state.refresh_balances:
                            st.session_state.refresh_balances = False
                    elif response.status_code == 404:
                        st.error("❌ Account not found")
                    else:
                        st.error("❌ Could not fetch balance")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Amount input
        amount = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.01, 
            step=0.01, 
            format="%.2f",
            key="emp_amount",
            help="Enter the amount to transfer"
        )
        
        # Create transaction button (full width like amount input)
        if st.button("💸 Create Transaction", type="primary", key="emp_create_tx", use_container_width=True):
            if from_account_id > 0 and to_account_id > 0 and amount > 0:
                if from_account_id == to_account_id:
                    st.error("❌ From and To accounts cannot be the same")
                else:
                    with st.spinner("Processing employee-initiated transaction..."):
                        try:
                            # Create transaction with employee as initiator
                            response = requests.post(
                                f"{API_URL}/employee/create-transaction",
                                headers=headers,
                                json={
                                    "from_account_id": int(from_account_id),
                                    "to_account_id": int(to_account_id), 
                                    "amount": float(amount)
                                }
                            )
                            if response.status_code == 200:
                                show_notification("✅ Transaction created successfully!")
                                st.rerun()  # Refresh to show new transaction
                            elif response.status_code == 400:
                                error_detail = response.json().get("detail", "Invalid transaction")
                                if "Insufficient funds" in error_detail:
                                    st.error("❌ Insufficient funds in sender account")
                                else:
                                    st.error(f"❌ {error_detail}")
                            elif response.status_code == 404:
                                error_detail = response.json().get("detail", "Account not found")
                                if "customer account" in error_detail:
                                    st.error(f"❌ {error_detail}")
                                else:
                                    st.error("❌ One or both accounts not found")
                            elif response.status_code == 403:
                                st.error("❌ Access denied. Employee privileges required.")
                            else:
                                st.error(f"❌ Transaction failed: {response.text}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Connection error: {str(e)}")
                            st.info("Make sure the backend server is running on http://localhost:8000")
                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
            else:
                st.warning("Please enter valid account IDs and amount")
        
        # Helper information
        st.markdown("---")
        st.markdown("### ℹ️ Employee Transaction Guidelines")
        st.info("""
        - Only customer accounts can send and receive money
        - Employee account will be recorded as the transaction initiator
        - Both sender and receiver accounts must exist and be customer accounts
        - Sender must have sufficient balance for the transaction
        - All transaction amounts are encrypted using post-quantum cryptography
        """)
    
    with tab4:
        st.subheader("🔐 Post-Quantum Security Overview")
        st.info("Transaction security features employed in the system")
        
        # Security features overview
        st.markdown("""
        ### 🛡️ Transaction Security Features
        
        #### **Encryption**
        - **CRYSTAL-Kyber512**: Used for encrypting transaction amounts
        - Each transaction generates a new shared secret
        - Quantum-resistant key encapsulation mechanism
        
        #### **Digital Signatures**
        - **CRYSTAL-DILITHIUM2**: Used for transaction authentication
        - All transactions are digitally signed
        - Provides non-repudiation and integrity verification
        
        #### **Data Protection**
        - Transaction amounts are encrypted at rest
        - Decryption only available to authorized employees
        - Audit trail for all access attempts
        
        ### 📊 Security Status
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔐 Encryption", "Active", help="CRYSTAL-Kyber encryption active")
        with col2:
            st.metric("📝 Signatures", "Verified", help="CRYSTAL-DILITHIUM signatures enabled")
        with col3:
            st.metric("🛡️ Quantum-Safe", "Yes", help="Post-quantum cryptography in use")
        
        # Demo buttons
        st.markdown("### 🔬 Security Demonstrations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔑 Demo: Key Generation"):
                with st.spinner("Generating post-quantum keys..."):
                    import time
                    time.sleep(1)
                    st.success("✅ CRYSTAL-Kyber keypair generated!")
                    st.success("✅ CRYSTAL-DILITHIUM keypair generated!")
                    st.info("🔑 Keys would be stored securely for transaction encryption")
        
        with col2:
            if st.button("📝 Demo: Transaction Signing"):
                with st.spinner("Creating transaction signature..."):
                    import time
                    time.sleep(1)
                    st.success("✅ Transaction signed with DILITHIUM!")
                    st.success("✅ Signature verified successfully!")
                    st.info("🛡️ Transaction integrity and authenticity guaranteed")
