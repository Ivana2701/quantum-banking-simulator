"""
Quantum-Safe Session Manager for Frontend
Handles secure session establishment using the hybrid post-quantum protocol
"""
import streamlit as st
import requests
import json
import time
import os
import tempfile
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

API_URL = "http://localhost:8000"

@dataclass
class SecureSession:
    """Represents a quantum-safe secure session"""
    session_id: str
    kyber_public_key: str
    dilithium_public_key: str
    bb84_key: str
    aes_key: str
    established_at: float
    expires_at: float
    
    def is_expired(self) -> bool:
        """Check if the session has expired"""
        return time.time() > self.expires_at
    
    def time_until_expiry(self) -> float:
        """Get time until session expires in seconds"""
        return max(0, self.expires_at - time.time())

class QuantumSessionManager:
    """Manages quantum-safe sessions for secure transactions"""
    
    def __init__(self):
        self.session_key = "quantum_secure_session"
        self.session_duration = 1800  # 30 minutes
        
        # Create a unique session file for quantum sessions
        self.session_file = os.path.join(tempfile.gettempdir(), "qbank_quantum_session.json")
        
    def _get_stored_session(self) -> Optional[Dict[str, Any]]:
        """Get stored quantum session data from Streamlit session state or file"""
        # First try session state (for current session)
        session_data = st.session_state.get(self.session_key)
        if session_data:
            return session_data
            
        # If not in session state, try to load from file (for persistence across refreshes)
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    file_data = json.load(f)
                    # Check if the session is not expired
                    current_time = time.time()
                    if current_time < file_data.get("expires_at", 0):
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
    
    def _store_session(self, session: 'SecureSession'):
        """Store quantum session data in Streamlit session state and persistent file"""
        session_data = {
            "session_id": session.session_id,
            "kyber_public_key": session.kyber_public_key,
            "dilithium_public_key": session.dilithium_public_key,
            "bb84_key": session.bb84_key,
            "aes_key": session.aes_key,
            "established_at": session.established_at,
            "expires_at": session.expires_at
        }
        
        # Store in session state for immediate use
        st.session_state[self.session_key] = session_data
        
        # Store in file for persistence across refreshes
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
        except Exception as e:
            # If we can't write to file, that's okay, just use session state
            pass
        
    def get_current_session(self) -> Optional[SecureSession]:
        """Get the current secure session if valid"""
        session_data = self._get_stored_session()
        if not session_data:
            return None
            
        session = SecureSession(**session_data)
        
        if session.is_expired():
            self.clear_session()
            return None
            
        return session
    
    def establish_session(self, auth_token: str) -> Tuple[bool, Optional[str]]:
        """Establish a new quantum-safe session"""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            # Call the backend to establish a secure session
            response = requests.post(
                f"{API_URL}/transactions/establish-session",
                headers=headers,
                json={"client_id": "frontend_client"},
                timeout=30
            )
            
            if response.status_code != 200:
                return False, f"Session establishment failed: {response.text}"
            
            session_data = response.json()
            
            # Create session object
            session = SecureSession(
                session_id=session_data["session_id"],
                kyber_public_key=session_data["kyber_public_key"],
                dilithium_public_key=session_data["bank_public_key"],
                bb84_key=session_data["bb84_parameters"].get("shared_key", ""),
                aes_key="derived",  # Placeholder - key is derived on backend
                established_at=time.time(),
                expires_at=time.time() + self.session_duration
            )
            
            # Store in session state
            self._store_session(session)
            
            return True, None
            
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Session establishment error: {str(e)}"
    
    def establish_session_manual(self, auth_token: str) -> Tuple[bool, Optional[str]]:
        """Manually establish a new quantum-safe session (for UI button)"""
        return self.establish_session(auth_token)
    
    def send_secure_transaction(
        self, 
        auth_token: str,
        to_account_id: int, 
        amount: float,
        description: str = ""
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Send a secure transaction using the quantum-safe protocol with proper two-step process"""
        
        # Get current session
        session = self.get_current_session()
        if not session:
            # Try to establish a new session
            success, error = self.establish_session(auth_token)
            if not success:
                return False, f"Could not establish secure session: {error}", None
            session = self.get_current_session()
        
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            # Step 1: Create encrypted and signed transaction package
            # Prepare transaction data for encryption
            transaction_create_data = {
                "to_account_id": to_account_id,
                "amount": amount
            }
            
            # Add description if provided (note: backend may not handle this field yet)
            if description:
                transaction_create_data["description"] = description
            
            # Call /create-secure-transaction to encrypt and sign the transaction
            create_response = requests.post(
                f"{API_URL}/transactions/create-secure-transaction",
                headers=headers,
                params={"session_id": session.session_id},
                json=transaction_create_data,
                timeout=30
            )
            
            if create_response.status_code != 200:
                return False, f"Failed to create secure transaction package: {create_response.text}", None
            
            encrypted_transaction_package = create_response.json()
            
            # Step 2: Submit the encrypted transaction package for execution
            execute_response = requests.post(
                f"{API_URL}/transactions/secure-transaction",
                headers=headers,
                json=encrypted_transaction_package,
                timeout=30
            )
            
            if execute_response.status_code == 200:
                result = execute_response.json()
                return True, None, result
            else:
                return False, f"Transaction execution failed: {execute_response.text}", None
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}", None
        except Exception as e:
            return False, f"Transaction error: {str(e)}", None
    
    def send_secure_transaction_debug(
        self, 
        auth_token: str,
        to_account_id: int, 
        amount: float,
        description: str = ""
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Debug version that shows each step of the quantum-safe transaction process"""
        
        # Get current session
        session = self.get_current_session()
        if not session:
            # Try to establish a new session
            success, error = self.establish_session(auth_token)
            if not success:
                return False, f"Could not establish secure session: {error}", None
            session = self.get_current_session()
        
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            print(f"🔐 DEBUG: Starting quantum secure transaction")
            print(f"📍 Session ID: {session.session_id}")
            print(f"💰 Amount: ${amount}")
            print(f"🎯 To Account: {to_account_id}")
            
            # Step 1: Create encrypted and signed transaction package
            transaction_create_data = {
                "to_account_id": to_account_id,
                "amount": amount
            }
            
            print(f"🔒 DEBUG: Creating encrypted transaction package...")
            create_response = requests.post(
                f"{API_URL}/transactions/create-secure-transaction",
                headers=headers,
                params={"session_id": session.session_id},
                json=transaction_create_data,
                timeout=30
            )
            
            print(f"📦 DEBUG: Encryption response status: {create_response.status_code}")
            
            if create_response.status_code != 200:
                error_msg = f"Failed to create secure transaction package: {create_response.text}"
                print(f"❌ DEBUG: {error_msg}")
                return False, error_msg, None
            
            encrypted_transaction_package = create_response.json()
            print(f"✅ DEBUG: Transaction encrypted and signed successfully")
            print(f"📋 DEBUG: Package keys: {list(encrypted_transaction_package.keys())}")
            
            # Step 2: Submit the encrypted transaction package for execution
            print(f"🚀 DEBUG: Executing encrypted transaction...")
            execute_response = requests.post(
                f"{API_URL}/transactions/secure-transaction",
                headers=headers,
                json=encrypted_transaction_package,
                timeout=30
            )
            
            print(f"⚡ DEBUG: Execution response status: {execute_response.status_code}")
            
            if execute_response.status_code == 200:
                result = execute_response.json()
                print(f"🎉 DEBUG: Transaction executed successfully!")
                print(f"🆔 DEBUG: Transaction ID: {result.get('transaction_id', 'N/A')}")
                return True, None, result
            else:
                error_msg = f"Transaction execution failed: {execute_response.text}"
                print(f"❌ DEBUG: {error_msg}")
                return False, error_msg, None
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"🌐 DEBUG: {error_msg}")
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Transaction error: {str(e)}"
            print(f"💥 DEBUG: {error_msg}")
            return False, error_msg, None
    
    def verify_transaction(
        self,
        auth_token: str,
        transaction_id: str
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Verify a transaction using quantum-safe signatures"""
        
        session = self.get_current_session()
        if not session:
            return False, "No secure session available", None
        
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            response = requests.post(
                f"{API_URL}/transactions/verify-transaction",
                headers=headers,
                json={
                    "session_id": session.session_id,
                    "transaction_id": transaction_id
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, None, result
            else:
                return False, f"Verification failed: {response.text}", None
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}", None
        except Exception as e:
            return False, f"Verification error: {str(e)}", None
    
    def cleanup_session(self, auth_token: str) -> bool:
        """Clean up the secure session"""
        session = self.get_current_session()
        if not session:
            return True
        
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            response = requests.post(
                f"{API_URL}/transactions/cleanup-session",
                headers=headers,
                json={"session_id": session.session_id},
                timeout=10
            )
            
            # Clear local session regardless of backend response
            self.clear_session()
            
            return response.status_code == 200
            
        except Exception:
            # Clear local session even if backend call fails
            self.clear_session()
            return False
    
    def clear_session(self):
        """Clear the current session from memory and persistent storage"""
        if self.session_key in st.session_state:
            del st.session_state[self.session_key]
            
        # Remove persistent file
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
        except Exception:
            pass
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed session information for debugging"""
        session = self.get_current_session()
        if not session:
            return None
            
        return {
            "session_id": session.session_id,
            "established_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.established_at)),
            "expires_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.expires_at)),
            "time_remaining_seconds": session.time_until_expiry(),
            "time_remaining_minutes": int(session.time_until_expiry() // 60),
            "protocols": ["BB84 QKD", "CRYSTALS-Kyber", "CRYSTALS-Dilithium", "AES-256-GCM"],
            "security_level": "Post-Quantum Secure"
        }

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status for UI display"""
        session = self.get_current_session()
        
        if not session:
            return {
                "active": False,
                "session_id": None,
                "time_remaining": 0,
                "security_level": "No Session"
            }
        
        time_remaining = session.time_until_expiry()
        
        return {
            "active": True,
            "session_id": session.session_id[:8] + "...",  # Truncated for display
            "time_remaining": time_remaining,
            "security_level": "Post-Quantum Secure",
            "protocols": ["BB84 QKD", "CRYSTALS-Kyber", "CRYSTALS-Dilithium", "AES-256-GCM"]
        }

# Global instance
quantum_session_manager = QuantumSessionManager()
