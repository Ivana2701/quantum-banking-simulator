# frontend/screens/employee_customers.py
import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

def show_employee_customers():
    st.title("👥 Customer Management")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Fetch all customers
    try:
        response = requests.get(f"{API_URL}/employee/customers", headers=headers)
        if response.status_code == 200:
            customers = response.json()
            
            if customers:
                st.subheader("📋 All Customers")
                
                # Convert to DataFrame for better display
                customer_data = []
                for customer in customers:
                    # For display purposes, we'll show a placeholder balance
                    # since the actual balance is encrypted with post-quantum cryptography
                    # In a real system, the employee might have limited access to decrypted balances
                    
                    # Try to get balance via a separate API call (if available)
                    try:
                        balance_response = requests.get(
                            f"{API_URL}/employee/customers/{customer['account_id']}/balance",
                            headers=headers
                        )
                        if balance_response.status_code == 200:
                            balance_data = balance_response.json()
                            balance_display = f"${balance_data['balance']:,.2f}"
                        else:
                            balance_display = "**Encrypted**"
                    except:
                        balance_display = "**Encrypted**"
                    
                    customer_data.append({
                        "ID": customer["account_id"],
                        "Username": customer["username"],
                        "Full Name": customer["full_name"],
                        "Balance": balance_display,
                        "Status": customer["status"],
                        "Created": customer["created_at"][:10] if customer["created_at"] else "N/A"
                    })
                
                df = pd.DataFrame(customer_data)
                
                # Display customers table
                st.dataframe(df, use_container_width=True)
                
                st.divider()
                
                # Money addition section
                st.subheader("💰 Add Money to Customer Account")
                
                # Select customer to add money to
                selected_customer = st.selectbox(
                    "Select customer to add money:",
                    options=[(customer["account_id"], f"{customer['username']} ({customer['full_name']})") for customer in customers],
                    format_func=lambda x: x[1]
                )
                
                if selected_customer:
                    customer_id = selected_customer[0]
                    current_customer = next(customer for customer in customers if customer["account_id"] == customer_id)
                    
                    # Show current customer info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Customer:** {current_customer['full_name']}")
                        st.write(f"**Username:** {current_customer['username']}")
                    with col2:
                        # Display current balance
                        current_balance_data = next((c for c in customer_data if c["ID"] == customer_id), None)
                        if current_balance_data:
                            st.write(f"**Current Balance:** {current_balance_data['Balance']}")
                    
                    # Amount input
                    amount_to_add = st.number_input(
                        "Amount to add:",
                        min_value=0.01,
                        step=0.01,
                        format="%.2f"
                    )
                    
                    # Add money button
                    if st.button(f"Add ${amount_to_add:.2f} to {current_customer['username']}'s account"):
                        if amount_to_add > 0:
                            try:
                                add_response = requests.post(
                                    f"{API_URL}/employee/customers/{customer_id}/add-money",
                                    headers=headers,
                                    json={"amount": amount_to_add}
                                )
                                
                                if add_response.status_code == 200:
                                    result = add_response.json()
                                    st.success(f"✅ {result['message']}")
                                    st.info(f"New balance: ${result['new_balance']:,.2f}")
                                    st.rerun()  # Refresh the page to show updated data
                                else:
                                    error_msg = add_response.json().get("detail", "Unknown error")
                                    st.error(f"Failed to add money: {error_msg}")
                            except Exception as e:
                                st.error(f"Error adding money: {str(e)}")
                        else:
                            st.warning("Please enter an amount greater than 0")
                
                # Statistics section
                st.divider()
                st.subheader("📊 Customer Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_customers = len(customers)
                active_customers = len([c for c in customers if c["status"] == "active"])
                
                # Calculate total balance across all customers (where available)
                total_balance = 0
                customers_with_balance = 0
                for customer_info in customer_data:
                    try:
                        if customer_info["Balance"] != "**Encrypted**":
                            balance_str = customer_info["Balance"].replace("$", "").replace(",", "")
                            total_balance += float(balance_str)
                            customers_with_balance += 1
                    except:
                        pass
                
                with col1:
                    st.metric("Total Customers", total_customers)
                with col2:
                    st.metric("Active Customers", active_customers)
                with col3:
                    if customers_with_balance > 0:
                        st.metric("Total Balance", f"${total_balance:,.2f}")
                    else:
                        st.metric("Balances", "🔐 Encrypted")
                with col4:
                    if customers_with_balance > 0:
                        avg_balance = total_balance / customers_with_balance
                        st.metric("Average Balance", f"${avg_balance:,.2f}")
                    else:
                        st.metric("Encryption", "✅ Active")
                
                # Security information
                st.divider()
                st.subheader("🔐 Security Information")
                
                st.info("""
                **Post-Quantum Cryptography Protection**
                
                Customer balances are protected using:
                - 🔑 **CRYSTAL-Kyber**: Quantum-resistant key encapsulation
                - 📝 **CRYSTAL-DILITHIUM**: Post-quantum digital signatures  
                - 🔒 **AES-256**: Military-grade symmetric encryption
                
                All balance modifications are cryptographically signed and encrypted for maximum security.
                """)
                
            else:
                st.info("No customers found in the system")
                
        elif response.status_code == 403:
            st.error("Access denied. Employee privileges required.")
        else:
            st.error(f"Failed to fetch customers. Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
