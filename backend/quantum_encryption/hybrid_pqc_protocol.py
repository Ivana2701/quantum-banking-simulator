"""
Hybrid Post-Quantum Secure Protocol for Quantum Banking Simulator

This module implements a hybrid cryptographic protocol combining:
1. BB84 Quantum Key Distribution (QKD) for quantum key establishment
2. CRYSTALS-Kyber (PQC KEM) for post-quantum key encapsulation
3. CRYSTALS-Dilithium (PQC signatures) for digital signatures
4. AES-256-GCM for symmetric encryption of transaction data
5. HKDF for key derivation and session key establishment

The protocol ensures both quantum and classical security for banking transactions.
"""

import os
import hashlib
import secrets
import base64
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import json

try:
    import oqs
    OQS_AVAILABLE = True
except ImportError:
    OQS_AVAILABLE = False
    logging.warning("liboqs not available. PQC features will be disabled.")

from quantum_encryption.bb84 import BB84Protocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridPQCProtocol:
    """
    Hybrid Post-Quantum Cryptographic Protocol for secure banking transactions.
    
    This class implements a multi-layered security approach:
    - BB84 QKD for quantum-secure key material
    - Kyber KEM for post-quantum key encapsulation
    - Dilithium for post-quantum digital signatures
    - AES-256-GCM for symmetric encryption
    """
    
    def __init__(self):
        self.bb84 = BB84Protocol()
        
        # Initialize PQC algorithms if available
        if OQS_AVAILABLE:
            try:
                # CRYSTALS-Kyber for KEM
                self.kyber_kem = oqs.KeyEncapsulation("Kyber1024")
                
                # CRYSTALS-Dilithium for signatures
                self.dilithium_signer = oqs.Signature("Dilithium5")
                bank_public_key = self.dilithium_signer.generate_keypair()
                
                # Store signing keys for verification
                self.bank_signing_keys = {
                    'public_key': bank_public_key,
                    'private_key': self.dilithium_signer.export_secret_key()
                }
                
                logger.info("PQC algorithms initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize PQC algorithms: {e}")
                raise
        else:
            raise ImportError("liboqs is required for PQC functionality")
    
    def establish_session_keys(self, client_id: str) -> Dict[str, Any]:
        """
        Establish session keys using hybrid BB84 + Kyber approach.
        
        Args:
            client_id: Unique identifier for the client session
            
        Returns:
            Dictionary containing session information and public keys
        """
        logger.info(f"Establishing session keys for client {client_id}")
        
        # Step 1: BB84 Quantum Key Distribution
        bb84_result = self.bb84.run_full_protocol()
        bb84_key = bb84_result['shared_key']
        
        logger.info(f"BB84 key established. Length: {len(bb84_key)} bits")
        
        # Step 2: Kyber Key Encapsulation
        kyber_public_key = self.kyber_kem.generate_keypair()
        
        # For demonstration, we'll simulate the client encapsulating a shared secret
        # In practice, the client would do this step
        encapsulated_secret, shared_secret = self.kyber_kem.encap_secret(kyber_public_key)
        
        logger.info(f"Kyber shared secret established. Length: {len(shared_secret)} bytes")
        
        # Step 3: Combine keys using HKDF
        session_key = self._derive_session_key(bb84_key, shared_secret, client_id)
        
        # Step 4: Generate session metadata
        session_info = {
            'session_id': self._generate_session_id(),
            'client_id': client_id,
            'timestamp': datetime.utcnow().isoformat(),
            'kyber_public_key': base64.b64encode(kyber_public_key).decode('utf-8'),
            'kyber_encapsulated_secret': base64.b64encode(encapsulated_secret).decode('utf-8'),
            'bank_public_key': base64.b64encode(self.bank_signing_keys['public_key']).decode('utf-8'),
            'bb84_parameters': bb84_result['parameters']
        }
        
        # Store session key securely (in production, use secure key management)
        self._store_session_key(session_info['session_id'], session_key)
        
        logger.info(f"Session established: {session_info['session_id']}")
        return session_info
    
    def _derive_session_key(self, bb84_key: str, kyber_secret: bytes, client_id: str) -> bytes:
        """
        Derive session key from BB84 and Kyber key material using HKDF.
        
        Args:
            bb84_key: Binary key from BB84 protocol
            kyber_secret: Shared secret from Kyber KEM
            client_id: Client identifier for key diversification
            
        Returns:
            256-bit AES session key
        """
        # Convert BB84 binary string to bytes
        bb84_bytes = int(bb84_key, 2).to_bytes((len(bb84_key) + 7) // 8, byteorder='big')
        
        # Combine key materials
        combined_material = bb84_bytes + kyber_secret + client_id.encode('utf-8')
        
        # Derive session key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=b'quantum-banking-session-salt',
            info=b'hybrid-pqc-session-key',
            backend=default_backend()
        )
        
        session_key = hkdf.derive(combined_material)
        logger.info("Session key derived using HKDF")
        return session_key
    
    def encrypt_transaction_data(self, session_id: str, transaction_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt transaction data using AES-256-GCM with the session key.
        
        Args:
            session_id: Session identifier
            transaction_data: Transaction data to encrypt
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        session_key = self._get_session_key(session_id)
        if not session_key:
            raise ValueError(f"Session key not found for session {session_id}")
        
        # Serialize transaction data
        plaintext = json.dumps(transaction_data, sort_keys=True).encode('utf-8')
        
        # Generate nonce for AES-GCM
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        # Encrypt using AES-256-GCM
        aesgcm = AESGCM(session_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        encrypted_package = {
            'session_id': session_id,
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Transaction data encrypted for session {session_id}")
        return encrypted_package
    
    def decrypt_transaction_data(self, encrypted_package: Dict[str, str]) -> Dict[str, Any]:
        """
        Decrypt transaction data using AES-256-GCM.
        
        Args:
            encrypted_package: Encrypted transaction package
            
        Returns:
            Decrypted transaction data
        """
        session_id = encrypted_package['session_id']
        session_key = self._get_session_key(session_id)
        
        if not session_key:
            raise ValueError(f"Session key not found for session {session_id}")
        
        # Decode encrypted components
        nonce = base64.b64decode(encrypted_package['nonce'])
        ciphertext = base64.b64decode(encrypted_package['ciphertext'])
        
        # Decrypt using AES-256-GCM
        aesgcm = AESGCM(session_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        # Deserialize transaction data
        transaction_data = json.loads(plaintext.decode('utf-8'))
        
        logger.info(f"Transaction data decrypted for session {session_id}")
        return transaction_data
    
    def sign_transaction(self, transaction_data: Dict[str, Any]) -> str:
        """
        Sign transaction data using Dilithium post-quantum signatures.
        
        Args:
            transaction_data: Transaction data to sign
            
        Returns:
            Base64-encoded signature
        """
        # Serialize transaction data for signing
        message = json.dumps(transaction_data, sort_keys=True).encode('utf-8')
        
        # Use the stored signer instance to sign the message
        signature = self.dilithium_signer.sign(message)
        
        logger.info("Transaction signed with Dilithium")
        return base64.b64encode(signature).decode('utf-8')
    
    def verify_transaction_signature(self, transaction_data: Dict[str, Any], signature_b64: str, public_key_b64: str = None) -> bool:
        """
        Verify transaction signature using Dilithium.
        
        Args:
            transaction_data: Transaction data to verify
            signature_b64: Base64-encoded signature
            public_key_b64: Optional public key (uses bank's key if not provided)
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Serialize transaction data
            message = json.dumps(transaction_data, sort_keys=True).encode('utf-8')
            signature = base64.b64decode(signature_b64)
            
            # Use provided public key or bank's public key
            if public_key_b64:
                public_key = base64.b64decode(public_key_b64)
            else:
                public_key = self.bank_signing_keys['public_key']
            
            # Create verifier and verify signature
            verifier = oqs.Signature("Dilithium5")
            is_valid = verifier.verify(message, signature, public_key)
            
            logger.info(f"Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def create_secure_transaction_request(self, session_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete secure transaction request with encryption and signature.
        
        Args:
            session_id: Session identifier
            transaction_data: Transaction data
            
        Returns:
            Secure transaction request package
        """
        # Add timestamp and transaction ID
        transaction_data['timestamp'] = datetime.utcnow().isoformat()
        transaction_data['transaction_id'] = self._generate_transaction_id()
        
        # Encrypt the transaction data
        encrypted_package = self.encrypt_transaction_data(session_id, transaction_data)
        
        # Sign the encrypted package
        signature = self.sign_transaction(encrypted_package)
        
        # Create complete secure package
        secure_request = {
            'session_id': session_id,  # Add the session_id that's required by the schema
            'encrypted_transaction': encrypted_package,
            'signature': signature,
            'bank_public_key': base64.b64encode(self.bank_signing_keys['public_key']).decode('utf-8'),
            'protocol_version': '1.0',
            'security_level': 'hybrid-pqc'
        }
        
        logger.info(f"Secure transaction request created for session {session_id}")
        return secure_request
    
    def process_secure_transaction_request(self, secure_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate a secure transaction request.
        
        Args:
            secure_request: Secure transaction request package
            
        Returns:
            Decrypted and validated transaction data
        """
        # Verify signature
        encrypted_package = secure_request['encrypted_transaction']
        signature = secure_request['signature']
        
        if not self.verify_transaction_signature(encrypted_package, signature):
            raise ValueError("Invalid transaction signature")
        
        # Decrypt transaction data
        transaction_data = self.decrypt_transaction_data(encrypted_package)
        
        logger.info("Secure transaction request processed successfully")
        return transaction_data
    
    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        return f"session_{secrets.token_hex(16)}"
    
    def _generate_transaction_id(self) -> str:
        """Generate a unique transaction identifier."""
        return f"tx_{secrets.token_hex(12)}"
    
    def _store_session_key(self, session_id: str, session_key: bytes):
        """
        Store session key securely.
        In production, this should use a secure key management system.
        """
        # For demonstration, we'll use a simple in-memory store
        if not hasattr(self, '_session_keys'):
            self._session_keys = {}
        
        self._session_keys[session_id] = {
            'key': session_key,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=24)
        }
    
    def _get_session_key(self, session_id: str) -> Optional[bytes]:
        """
        Retrieve session key securely.
        """
        if not hasattr(self, '_session_keys'):
            return None
        
        session_info = self._session_keys.get(session_id)
        if not session_info:
            return None
        
        # Check if session has expired
        if datetime.utcnow() > session_info['expires_at']:
            del self._session_keys[session_id]
            return None
        
        return session_info['key']
    
    def cleanup_expired_sessions(self):
        """Remove expired session keys."""
        if not hasattr(self, '_session_keys'):
            return
        
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session_info in self._session_keys.items()
            if current_time > session_info['expires_at']
        ]
        
        for session_id in expired_sessions:
            del self._session_keys[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Global protocol instance
_protocol_instance = None

def get_protocol_instance() -> HybridPQCProtocol:
    """Get or create the global protocol instance."""
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = HybridPQCProtocol()
    return _protocol_instance
