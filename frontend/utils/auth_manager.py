# frontend/utils/auth_manager.py
import streamlit as st
import requests
import json
import time
import os
import tempfile
import hashlib
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from backend .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

API_URL = "http://localhost:8000"

class AuthManager:
    """Manages authentication and session persistence for the frontend"""
    
    def __init__(self):
        self.session_key = "qbank_session"
        self.token_key = "access_token"
        self.user_key = "user_data"
        
        # Get token expiration from environment (same as backend)
        self.token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        
        # Create a unique session file based on the user's machine
        self.session_file = os.path.join(tempfile.gettempdir(), "qbank_session.json")
        
    def _get_stored_session(self) -> Optional[Dict[str, Any]]:
        """Get stored session data from Streamlit session state or file"""
        # First try session state (for current session)
        session_data = st.session_state.get(self.session_key)
        if session_data:
            return session_data
            
        # If not in session state, try to load from file (for persistence across refreshes)
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    file_data = json.load(f)
                    # Check if the session is not expired based on token expiration time
                    session_age_minutes = (time.time() - file_data.get("timestamp", 0)) / 60
                    if session_age_minutes < self.token_expire_minutes:
                        # Restore to session state
                        st.session_state[self.session_key] = file_data
                        return file_data
                    else:
                        # Session expired, remove file
                        os.remove(self.session_file)
        except Exception:
            # If file is corrupted or can't be read, ignore
            pass
            
        return None
    
    def _store_session(self, token: str, user_data: Dict[str, Any]):
        """Store session data in Streamlit session state and persistent file"""
        session_data = {
            self.token_key: token,
            self.user_key: user_data,
            "timestamp": time.time(),
            "persistent": True  # Mark as persistent session
        }
        
        # Store in session state for immediate use
        st.session_state[self.session_key] = session_data
        
        # Store in legacy format for backward compatibility
        st.session_state.token = token
        st.session_state.account_type = user_data.get("account_type")
        st.session_state.account_id = user_data.get("account_id")
        st.session_state.username = user_data.get("username")
        st.session_state.full_name = user_data.get("full_name")
        
        # Additional persistence markers
        st.session_state._auth_timestamp = time.time()
        st.session_state._auth_persistent = True
        
        # Store in file for persistence across refreshes
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
        except Exception as e:
            # If we can't write to file, that's okay, just use session state
            pass
    
    def _clear_session(self):
        """Clear all session data"""
        if self.session_key in st.session_state:
            del st.session_state[self.session_key]
        
        # Clear legacy session state
        for key in ["token", "account_type", "account_id", "username", "full_name", "_auth_timestamp", "_auth_persistent"]:
            if key in st.session_state:
                del st.session_state[key]
                
        # Remove persistent file
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
        except Exception:
            pass
    
    def login(self, username: str, password: str) -> bool:
        """
        Authenticate user and establish session
        """
        try:
            # Prepare OAuth2 form data
            form_data = {
                "username": username,
                "password": password,
                "grant_type": "password",
                "scope": "",
                "client_id": "",
                "client_secret": "",
            }
            
            # Make login request
            response = requests.post(
                f"{API_URL}/auth/token",
                data=form_data,
                timeout=10
            )
            
            if response.status_code != 200:
                st.error(f"Login failed: {response.text}")
                return False
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                st.error("No access token received")
                return False
            
            # Get user profile
            user_data = self._get_user_profile(access_token)
            if not user_data:
                st.error("Failed to get user profile")
                return False
            
            # Store session
            self._store_session(access_token, user_data)
            
            st.success(f"Welcome, {user_data.get('full_name', username)}!")
            return True
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return False
        except Exception as e:
            st.error(f"Login error: {e}")
            return False
    
    def _get_user_profile(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user profile data using the access token"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{API_URL}/accounts/me", headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Error fetching user profile: {e}")
            return None
    
    def validate_session(self) -> bool:
        """
        Validate current session and refresh if necessary
        Enhanced with better persistence handling
        """
        session_data = self._get_stored_session()
        if not session_data:
            # Check if we have legacy session data that can be recovered
            if hasattr(st.session_state, 'token') and st.session_state.token:
                # Try to validate the legacy token
                token = st.session_state.token
                try:
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(f"{API_URL}/auth/validate", headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        # Token is valid, restore full session data
                        user_data = response.json()
                        self._store_session(token, user_data)
                        return True
                except Exception:
                    pass
            return False
        
        token = session_data.get(self.token_key)
        if not token:
            return False
        
        # Check if token is still valid
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{API_URL}/auth/validate", headers=headers, timeout=5)
            
            if response.status_code == 200:
                # Token is valid, update user data
                user_data = response.json()
                self._store_session(token, user_data)
                return True
            else:
                # Token is invalid, clear session
                self._clear_session()
                return False
                
        except Exception as e:
            # On network errors, be more forgiving for persistent sessions
            if session_data.get("persistent", False):
                # For persistent sessions, assume validity on network errors
                # but check if the session is not expired based on token expiration time
                session_age_minutes = (time.time() - session_data.get("timestamp", 0)) / 60
                if session_age_minutes < self.token_expire_minutes:
                    return True
            
            # Clear session if expired or not persistent
            self._clear_session()
            return False
    
    def logout(self):
        """Clear session and logout user"""
        self._clear_session()
        st.success("Logged out successfully")
        st.rerun()
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return self.validate_session()
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current user data"""
        session_data = self._get_stored_session()
        if session_data:
            return session_data.get(self.user_key)
        return None
    
    def get_token(self) -> Optional[str]:
        """Get current access token"""
        session_data = self._get_stored_session()
        if session_data:
            return session_data.get(self.token_key)
        return None
    
    def has_role(self, role: str) -> bool:
        """Check if current user has specific role"""
        user_data = self.get_current_user()
        if user_data:
            return user_data.get("account_type") == role
        return False
    
    def has_any_role(self, roles: list) -> bool:
        """Check if current user has any of the specified roles"""
        user_data = self.get_current_user()
        if user_data:
            return user_data.get("account_type") in roles
        return False
    
    def can_access_debug(self) -> bool:
        """Check if current user can access debug features (admin only)"""
        return self.has_role("admin")
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is currently enabled and user has permission"""
        return (self.can_access_debug() and 
                st.session_state.get("debug_auth", False))

# Global auth manager instance
auth_manager = AuthManager()
