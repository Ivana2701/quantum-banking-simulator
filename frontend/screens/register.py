# frontend/screens/register.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def show_register():
    st.title("📝 Register")
    
    # Basic information
    st.subheader("Account Information")
    
    # Use the label parameter with HTML styling for required fields
    full_name = st.text_input("Full Name *", key="full_name", placeholder="Enter your full name", help="Required field")
    
    username = st.text_input("Username *", key="username", placeholder="Choose a username (minimum 3 characters)", help="Required field")
        
    password = st.text_input("Password *", type="password", key="password", placeholder="Enter password (minimum 8 characters)", help="Required field")
    
    password2 = st.text_input("Confirm Password *", type="password", key="password2", placeholder="Confirm your password", help="Required field")
    
    account_type = st.selectbox("Account Type *", ["customer", "employee"], key="account_type", help="Required field")
    
    # Optional contact information
    st.subheader("Contact Information (Optional)")
    with st.expander("Address Information", expanded=False):
        street = st.text_input("Street Address", placeholder="123 Main Street")
        city = st.text_input("City", placeholder="Sofia")
        state = st.text_input("State/Province", placeholder="Sofia Province")
        country = st.text_input("Country", placeholder="Bulgaria")
        postal_code = st.text_input("Postal Code", placeholder="1000")
    
    with st.expander("Phone Information", expanded=False):
        phone_number = st.text_input("Phone Number", placeholder="+359888123456")
        phone_type = st.selectbox("Phone Type", ["mobile", "work", "home"])
    
    with st.expander("Device Information", expanded=False):
        device_name = st.text_input("Device Name", placeholder="My Laptop")
    
    # Geolocation section
    with st.expander("Location Services (Optional)", expanded=False):
        st.info("📍 Location information helps us provide better security for your account")
        
        # Initialize session state for location data
        if "user_location" not in st.session_state:
            st.session_state.user_location = None
        if "location_requested" not in st.session_state:
            st.session_state.location_requested = False
        if "location_found" not in st.session_state:
            st.session_state.location_found = False
        if "user_ip" not in st.session_state:
            st.session_state.user_ip = None
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📍 Get My Location", help="Uses your browser's location services"):
                st.session_state.location_requested = True
                st.session_state.location_found = False
                
                # JavaScript to get geolocation and display results
                geolocation_js = f"""
                <script>
                let locationAttempted = false;
                
                function getLocation() {{
                    if (locationAttempted) return;
                    locationAttempted = true;
                    
                    if (navigator.geolocation) {{
                        navigator.geolocation.getCurrentPosition(
                            function(position) {{
                                const lat = position.coords.latitude;
                                const lng = position.coords.longitude;
                                
                                // Clear any existing content and show success
                                document.body.innerHTML = '';
                                
                                // Display the coordinates immediately
                                const coordsDiv = document.createElement('div');
                                coordsDiv.innerHTML = `
                                    <div style="
                                        background: #e8f5e8; 
                                        border: 1px solid #4caf50; 
                                        border-radius: 5px; 
                                        padding: 10px; 
                                        margin: 10px 0;
                                        font-family: monospace;
                                    ">
                                        <strong>📍 Location Found:</strong><br>
                                        <strong>Latitude:</strong> ${{lat.toFixed(6)}}<br>
                                        <strong>Longitude:</strong> ${{lng.toFixed(6)}}
                                    </div>
                                `;
                                document.body.appendChild(coordsDiv);
                                
                                // Signal that location was found
                                window.parent.postMessage({{
                                    type: 'geolocation_success',
                                    latitude: lat,
                                    longitude: lng
                                }}, '*');
                            }},
                            function(error) {{
                                const errorDiv = document.createElement('div');
                                errorDiv.innerHTML = `
                                    <div style="
                                        background: #ffebee; 
                                        border: 1px solid #f44336; 
                                        border-radius: 5px; 
                                        padding: 10px; 
                                        margin: 10px 0;
                                    ">
                                        <strong>❌ Location Error:</strong><br>
                                        ${{error.message}}
                                    </div>
                                `;
                                document.body.appendChild(errorDiv);
                                
                                window.parent.postMessage({{
                                    type: 'geolocation_error',
                                    error: error.message
                                }}, '*');
                            }}
                        );
                    }} else {{
                        const errorDiv = document.createElement('div');
                        errorDiv.innerHTML = `
                            <div style="
                                background: #ffebee; 
                                border: 1px solid #f44336; 
                                border-radius: 5px; 
                                padding: 10px; 
                                margin: 10px 0;
                            ">
                                <strong>❌ Error:</strong><br>
                                Geolocation is not supported by this browser.
                            </div>
                        `;
                        document.body.appendChild(errorDiv);
                        
                        window.parent.postMessage({{
                            type: 'geolocation_error',
                            error: 'Geolocation is not supported by this browser.'
                        }}, '*');
                    }}
                }}
                getLocation();
                </script>
                """
                st.components.v1.html(geolocation_js, height=150)
            
            # Display current IP (informational)
            if st.button("🌐 Check My IP", help="Shows your current IP address"):
                try:
                    ip_response = requests.get("https://api.ipify.org?format=json", timeout=5)
                    if ip_response.status_code == 200:
                        current_ip = ip_response.json().get("ip")
                        st.session_state.user_ip = current_ip  # Store IP in session state
                        st.success(f"Your current IP address: {current_ip}")
                        st.info("✅ This IP will be saved with your account.")
                    else:
                        st.warning("Could not retrieve IP address")
                except Exception as e:
                    st.warning("Could not retrieve IP address")
        
        with col2:
            # Manual location entry
            manual_lat = st.number_input("Latitude", value=None, placeholder="42.6977", format="%.6f")
            manual_lng = st.number_input("Longitude", value=None, placeholder="23.3219", format="%.6f")
            location_desc = st.text_input("Location Description", placeholder="Sofia, Bulgaria")
        
    if st.button("Sign Up", type="primary"):
        # Client-side validation
        errors = []
        
        if not full_name or len(full_name.strip()) == 0:
            errors.append("Full Name is required")
        
        if not username or len(username.strip()) < 3:
            errors.append("Username is required and must be at least 3 characters long")

        if not password or len(password) < 8:
            errors.append("Password is required and must be at least 8 characters long")
        
        if not password2:
            errors.append("Password confirmation is required")
        elif password != password2:
            errors.append("Passwords do not match")
        
        if errors:
            st.error("Please fix the following issues:")
            for error in errors:
                st.error(f"• {error}")
            return

        payload = {
            "full_name": full_name.strip(),
            "username": username.strip(),
            "password": password,
            "account_type": account_type
        }
        
        # Add optional fields if provided
        if street and street.strip():
            payload["street"] = street.strip()
        if city and city.strip():
            payload["city"] = city.strip()
        if state and state.strip():
            payload["state"] = state.strip()
        if country and country.strip():
            payload["country"] = country.strip()
        if postal_code and postal_code.strip():
            payload["postal_code"] = postal_code.strip()
        if phone_number and phone_number.strip():
            payload["phone_number"] = phone_number.strip()
            payload["phone_type"] = phone_type
        if device_name and device_name.strip():
            payload["device_name"] = device_name.strip()
            payload["device_fingerprint"] = f"web_{username}_{device_name}".replace(" ", "_")
        
        # Add location data if provided
        if manual_lat is not None and manual_lng is not None:
            payload["latitude"] = manual_lat
            payload["longitude"] = manual_lng
        if location_desc and location_desc.strip():
            payload["geo_description"] = location_desc.strip()
        
        # Add IP address if captured
        if st.session_state.user_ip:
            payload["ip_address"] = st.session_state.user_ip
        
        try:
            r = requests.post(f"{API_URL}/accounts/createAccount", json=payload)
        except requests.exceptions.ConnectionError:
            st.error("❌ **Cannot connect to server**")
            st.error("Please make sure the backend server is running and try again.")
            return
        except requests.exceptions.Timeout:
            st.error("❌ **Request timed out**")
            st.error("The server is taking too long to respond. Please try again.")
            return
        except Exception as e:
            st.error(f"❌ **Network error**: {str(e)}")
            return

        if r.status_code == 201:
            st.success("🎉 **Account created successfully!**")
            st.success("You can now log in with your credentials.")
        elif r.status_code == 422:
            # Handle validation errors from the server
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error("❌ **Validation Error**")
                    if isinstance(error_detail["detail"], list):
                        for error in error_detail["detail"]:
                            field = error.get("loc", ["unknown"])[-1]
                            message = error.get("msg", "Invalid value")
                            st.error(f"• **{field}**: {message}")
                    else:
                        st.error(f"• {error_detail['detail']}")
                else:
                    st.error("❌ **Please check your input and try again**")
            except:
                st.error("❌ **Invalid input format**")
                st.error("Please check all fields and try again.")
        elif r.status_code == 400:
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error("❌ **Registration Failed**")
                    st.error(f"• {error_detail['detail']}")
                else:
                    st.error("❌ **Bad request**")
                    st.error("Please check your input and try again.")
            except:
                st.error("❌ **Username might already be taken**")
                st.error("Please try a different username.")
        elif r.status_code == 500:
            st.error("❌ **Server Error**")
            st.error("Something went wrong on our end. Please try again later.")
        else:
            st.error(f"❌ **Registration failed**")
            try:
                error_detail = r.json()
                if "detail" in error_detail:
                    st.error(f"• {error_detail['detail']}")
                else:
                    st.error(f"• Status code: {r.status_code}")
            except:
                st.error(f"• Status code: {r.status_code}")
