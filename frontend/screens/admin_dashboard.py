# frontend/screens/admin_dashboard.py
import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

def show_admin_dashboard():
    st.title("👑 Admin Dashboard")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Fetch all users
    try:
        response = requests.get(f"{API_URL}/admin/users", headers=headers)
        if response.status_code == 200:
            users = response.json()
            
            if users:
                # Role management section (moved to top)
                st.subheader("🔧 Role Management")
                
                # Select user to modify
                selected_user = st.selectbox(
                    "Select user to modify role:",
                    options=[(user["account_id"], f"{user['username']} ({user['full_name']})") for user in users],
                    format_func=lambda x: x[1]
                )
                
                if selected_user:
                    user_id = selected_user[0]
                    current_user = next(user for user in users if user["account_id"] == user_id)
                    
                    # Show current role
                    st.write(f"**Current Role:** {current_user['account_type']} (Role ID: {current_user['role_id']})")
                    
                    # Role selection
                    role_options = ["customer", "employee", "admin"]
                    current_role_index = role_options.index(current_user["account_type"]) if current_user["account_type"] in role_options else 0
                    
                    new_role = st.selectbox(
                        "New Role:",
                        options=role_options,
                        index=current_role_index
                    )
                    
                    # Update role button
                    if st.button(f"Update {current_user['username']}'s role to {new_role}"):
                        if new_role != current_user["account_type"]:
                            try:
                                update_response = requests.patch(
                                    f"{API_URL}/admin/users/{user_id}/role",
                                    headers=headers,
                                    json={"new_role": new_role}
                                )
                                
                                if update_response.status_code == 200:
                                    st.success(f"Successfully updated {current_user['username']}'s role to {new_role}")
                                    st.rerun()  # Refresh the page to show updated data
                                else:
                                    error_msg = update_response.json().get("detail", "Unknown error")
                                    st.error(f"Failed to update role: {error_msg}")
                            except Exception as e:
                                st.error(f"Error updating role: {str(e)}")
                        else:
                            st.info("No change needed - user already has this role")
                
                st.divider()  # Add visual separator
                
                # All Users section with pagination
                st.subheader("📋 All Users")
                
                # Convert to DataFrame for better display
                user_data = []
                for user in users:
                    user_data.append({
                        "ID": user["account_id"],
                        "Username": user["username"],
                        "Full Name": user["full_name"],
                        "Account Type": user["account_type"],
                        "Role ID": user["role_id"],
                        "Balance": user["encrypted_balance"],
                        "Status": user["status"],
                        "Created": user["created_at"][:10] if user["created_at"] else "N/A"
                    })
                
                df = pd.DataFrame(user_data)
                
                # Pagination logic
                users_per_page = 20
                total_pages = (len(df) + users_per_page - 1) // users_per_page  # Ceiling division
                
                if total_pages > 1:
                    # Initialize page number in session state
                    if "current_page" not in st.session_state:
                        st.session_state.current_page = 1
                    
                    # Page navigation
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                    
                    with col1:
                        if st.button("⏮️ First") and st.session_state.current_page > 1:
                            st.session_state.current_page = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("◀️ Previous") and st.session_state.current_page > 1:
                            st.session_state.current_page -= 1
                            st.rerun()
                    
                    with col3:
                        st.write(f"**Page {st.session_state.current_page} of {total_pages}** ({len(df)} total users)")
                    
                    with col4:
                        if st.button("Next ▶️") and st.session_state.current_page < total_pages:
                            st.session_state.current_page += 1
                            st.rerun()
                    
                    with col5:
                        if st.button("Last ⏭️") and st.session_state.current_page < total_pages:
                            st.session_state.current_page = total_pages
                            st.rerun()
                    
                    # Calculate start and end indices for current page
                    start_idx = (st.session_state.current_page - 1) * users_per_page
                    end_idx = min(start_idx + users_per_page, len(df))
                    
                    # Display the current page of data
                    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
                    
                    # Show page info at bottom
                    st.caption(f"Showing users {start_idx + 1}-{end_idx} of {len(df)}")
                else:
                    # If only one page, show all data
                    st.dataframe(df, use_container_width=True)
                
                # Statistics section
                st.subheader("📊 System Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_users = len(users)
                customers = len([u for u in users if u["account_type"] == "customer"])
                employees = len([u for u in users if u["account_type"] == "employee"])
                admins = len([u for u in users if u["account_type"] == "admin"])
                
                with col1:
                    st.metric("Total Users", total_users)
                with col2:
                    st.metric("Customers", customers)
                with col3:
                    st.metric("Employees", employees)
                with col4:
                    st.metric("Admins", admins)
                
            else:
                st.info("No users found in the system")
                
        elif response.status_code == 403:
            st.error("Access denied. Admin privileges required.")
        else:
            st.error(f"Failed to fetch users. Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
