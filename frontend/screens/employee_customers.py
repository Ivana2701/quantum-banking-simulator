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
                
                # Username filter
                username_filter = st.text_input("🔍 Filter by username:", placeholder="Enter username to filter...")
                
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
                
                # Apply username filter if provided
                if username_filter:
                    df_filtered = df[df["Username"].str.contains(username_filter, case=False, na=False)]
                else:
                    df_filtered = df
                
                # Pagination logic
                customers_per_page = 20
                total_pages = (len(df_filtered) + customers_per_page - 1) // customers_per_page if len(df_filtered) > 0 else 1
                
                if total_pages > 1:
                    # Initialize page number in session state
                    if "customer_current_page" not in st.session_state:
                        st.session_state.customer_current_page = 1
                    
                    # Reset to page 1 if filter changed
                    if "last_username_filter" not in st.session_state:
                        st.session_state.last_username_filter = ""
                    
                    if st.session_state.last_username_filter != username_filter:
                        st.session_state.customer_current_page = 1
                        st.session_state.last_username_filter = username_filter
                    
                    # Ensure current page is within bounds
                    st.session_state.customer_current_page = min(st.session_state.customer_current_page, total_pages)
                    
                    # Page navigation
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                    
                    with col1:
                        if st.button("⏮️ First", key="customers_first") and st.session_state.customer_current_page > 1:
                            st.session_state.customer_current_page = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("◀️ Previous", key="customers_prev") and st.session_state.customer_current_page > 1:
                            st.session_state.customer_current_page -= 1
                            st.rerun()
                    
                    with col3:
                        st.write(f"**Page {st.session_state.customer_current_page} of {total_pages}** ({len(df_filtered)} customers)")
                    
                    with col4:
                        if st.button("Next ▶️", key="customers_next") and st.session_state.customer_current_page < total_pages:
                            st.session_state.customer_current_page += 1
                            st.rerun()
                    
                    with col5:
                        if st.button("Last ⏭️", key="customers_last") and st.session_state.customer_current_page < total_pages:
                            st.session_state.customer_current_page = total_pages
                            st.rerun()
                    
                    # Calculate start and end indices for current page
                    start_idx = (st.session_state.customer_current_page - 1) * customers_per_page
                    end_idx = min(start_idx + customers_per_page, len(df_filtered))
                    
                    # Display the current page of data
                    if len(df_filtered) > 0:
                        st.dataframe(df_filtered.iloc[start_idx:end_idx], use_container_width=True)
                        st.caption(f"Showing customers {start_idx + 1}-{end_idx} of {len(df_filtered)}")
                    else:
                        st.info("No customers match the current filter.")
                else:
                    # If only one page or no results, show all filtered data
                    if len(df_filtered) > 0:
                        st.dataframe(df_filtered, use_container_width=True)
                        if username_filter:
                            st.caption(f"Showing {len(df_filtered)} customers matching '{username_filter}'")
                        else:
                            st.caption(f"Showing all {len(df_filtered)} customers")
                    else:
                        st.info("No customers match the current filter.")
                
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
