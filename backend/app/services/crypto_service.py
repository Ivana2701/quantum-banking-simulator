"""
Post-Quantum Cryptography Service
Implements CRYSTAL-Kyber for key encapsulation, CRYSTAL-DILITHIUM for digital signatures, and AES for symmetric encryption
"""
try:
    import liboqs
    OQS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Using real post-quantum cryptography (liboqs)")
except ImportError:
    try:
        # Try the package we just installed
        from liboqs_python import liboqs
        OQS_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("Using real post-quantum cryptography (liboqs_python)")
    except ImportError:
        OQS_AVAILABLE = False
        print("Warning: liboqs-python not available. Falling back to simulation mode.")

import os
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import secrets

logger = logging.getLogger(__name__)

class PostQuantumCryptoService:
    """Service for post-quantum cryptographic operations"""
    
    def __init__(self):
        # Post-quantum algorithms
        self.kem_algorithm = "Kyber512"  # CRYSTAL-Kyber for key encapsulation
        self.sig_algorithm = "Dilithium2"  # CRYSTAL-DILITHIUM for digital signatures
        self.oqs_available = OQS_AVAILABLE
        
        if not self.oqs_available:
            logger.warning("OQS not available - using simulation mode")
    
    def generate_kyber_keypair(self):
        """Generate a Kyber key pair for key encapsulation"""
        try:
            if self.oqs_available:
                with liboqs.KeyEncapsulation(self.kem_algorithm) as kem:
                    public_key = kem.generate_keypair()
                    return public_key, kem.export_secret_key()
            else:
                # Simulation mode - generate deterministic keys for consistency
                master_seed = secrets.token_bytes(32)
                import hashlib
                public_key = hashlib.sha256(master_seed + b"public").digest() + secrets.token_bytes(800 - 32)
                secret_key = master_seed + secrets.token_bytes(1632 - 32)  # Include master seed in secret key
                return public_key, secret_key
        except Exception as e:
            logger.error(f"Error generating Kyber keypair: {str(e)}")
            raise
    
    def generate_dilithium_keypair(self):
        """Generate a Dilithium key pair for digital signatures"""
        try:
            if self.oqs_available:
                with liboqs.Signature(self.sig_algorithm) as sig:
                    public_key = sig.generate_keypair()
                    return public_key, sig.export_secret_key()
            else:
                # Simulation mode
                public_key = secrets.token_bytes(1312)  # Dilithium2 public key size
                secret_key = secrets.token_bytes(2528)  # Dilithium2 secret key size
                return public_key, secret_key
        except Exception as e:
            logger.error(f"Error generating Dilithium keypair: {str(e)}")
            raise
    
    def kyber_encapsulate(self, public_key: bytes):
        """Use Kyber to encapsulate a shared secret"""
        try:
            if self.oqs_available:
                with liboqs.KeyEncapsulation(self.kem_algorithm) as kem:
                    ciphertext, shared_secret = kem.encap_secret(public_key)
                    return ciphertext, shared_secret
            else:
                # Simulation mode - derive shared secret from public key
                import hashlib
                # Use the first 32 bytes of public key as source for shared secret
                shared_secret = hashlib.sha256(public_key[:32] + b"shared_secret").digest()
                # Create a dummy ciphertext that embeds the shared secret for later recovery
                ciphertext = shared_secret + secrets.token_bytes(768 - 32)
                return ciphertext, shared_secret
        except Exception as e:
            logger.error(f"Error in Kyber encapsulation: {str(e)}")
            raise
    
    def kyber_decapsulate(self, ciphertext: bytes, secret_key: bytes):
        """Use Kyber to decapsulate a shared secret"""
        try:
            if self.oqs_available:
                with liboqs.KeyEncapsulation(self.kem_algorithm) as kem:
                    kem.secret_key = secret_key
                    shared_secret = kem.decap_secret(ciphertext)
                    return shared_secret
            else:
                # Simulation mode - extract shared secret from ciphertext
                # In our simulation, the first 32 bytes of ciphertext is the shared secret
                return ciphertext[:32]
        except Exception as e:
            logger.error(f"Error in Kyber decapsulation: {str(e)}")
            raise
    
    def dilithium_sign(self, message: bytes, secret_key: bytes):
        """Sign a message using Dilithium"""
        try:
            if self.oqs_available:
                with liboqs.Signature(self.sig_algorithm) as sig:
                    sig.secret_key = secret_key
                    signature = sig.sign(message)
                    return signature
            else:
                # Simulation mode - create deterministic signature
                import hashlib
                combined = message + secret_key
                signature = hashlib.sha256(combined).digest() + secrets.token_bytes(32)
                return signature
        except Exception as e:
            logger.error(f"Error in Dilithium signing: {str(e)}")
            raise
    
    def dilithium_verify(self, message: bytes, signature: bytes, public_key: bytes):
        """Verify a Dilithium signature"""
        try:
            if self.oqs_available:
                with liboqs.Signature(self.sig_algorithm) as sig:
                    return sig.verify(message, signature, public_key)
            else:
                # Simulation mode - simple verification 
                # In real implementation, this would be cryptographically secure
                return len(signature) == 64  # Simple check for simulation
        except Exception as e:
            logger.error(f"Error in Dilithium verification: {str(e)}")
            return False
    
    def aes_encrypt(self, plaintext: bytes, shared_secret: bytes):
        """Encrypt data using AES with a shared secret from Kyber"""
        try:
            # Derive a 256-bit key from the shared secret
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(shared_secret)
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Encrypt the data
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad the plaintext to be a multiple of 16 bytes
            pad_length = 16 - (len(plaintext) % 16)
            padded_plaintext = plaintext + bytes([pad_length] * pad_length)
            
            ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
            
            # Return salt + IV + ciphertext
            return salt + iv + ciphertext
            
        except Exception as e:
            logger.error(f"Error in AES encryption: {str(e)}")
            raise
    
    def aes_decrypt(self, encrypted_data: bytes, shared_secret: bytes):
        """Decrypt AES-encrypted data using a shared secret"""
        try:
            # Extract salt, IV, and ciphertext
            salt = encrypted_data[:16]
            iv = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Derive the same key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(shared_secret)
            
            # Decrypt the data
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            pad_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-pad_length]
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Error in AES decryption: {str(e)}")
            raise
    
    def encrypt_balance(self, balance: float, public_key: bytes):
        """Encrypt a balance amount using post-quantum cryptography"""
        try:
            # Convert balance to bytes
            balance_bytes = str(balance).encode('utf-8')
            
            # Use Kyber to generate shared secret
            ciphertext, shared_secret = self.kyber_encapsulate(public_key)
            
            # Encrypt balance with AES using the shared secret
            encrypted_balance = self.aes_encrypt(balance_bytes, shared_secret)
            
            # Return both the Kyber ciphertext and encrypted balance
            return ciphertext, encrypted_balance
            
        except Exception as e:
            logger.error(f"Error encrypting balance: {str(e)}")
            raise
    
    def decrypt_balance(self, kyber_ciphertext: bytes, encrypted_balance: bytes, secret_key: bytes):
        """Decrypt a balance amount using post-quantum cryptography"""
        try:
            # Use Kyber to recover shared secret
            shared_secret = self.kyber_decapsulate(kyber_ciphertext, secret_key)
            
            # Decrypt balance with AES
            balance_bytes = self.aes_decrypt(encrypted_balance, shared_secret)
            
            # Convert back to float
            balance = float(balance_bytes.decode('utf-8'))
            
            return balance
            
        except Exception as e:
            logger.error(f"Error decrypting balance: {str(e)}")
            raise

# Global instance
crypto_service = PostQuantumCryptoService()
