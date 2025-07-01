#!/usr/bin/env python3
"""
Test script for post-quantum cryptography implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.crypto_service import crypto_service

def test_post_quantum_crypto():
    """Test the post-quantum cryptography implementation"""
    print("🔐 Testing Post-Quantum Cryptography Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Kyber Key Generation
        print("\n1. Testing CRYSTAL-Kyber Key Generation...")
        kyber_public_key, kyber_secret_key = crypto_service.generate_kyber_keypair()
        print(f"   ✅ Kyber public key generated: {len(kyber_public_key)} bytes")
        print(f"   ✅ Kyber secret key generated: {len(kyber_secret_key)} bytes")
        
        # Test 2: Dilithium Key Generation
        print("\n2. Testing CRYSTAL-DILITHIUM Key Generation...")
        dilithium_public_key, dilithium_secret_key = crypto_service.generate_dilithium_keypair()
        print(f"   ✅ Dilithium public key generated: {len(dilithium_public_key)} bytes")
        print(f"   ✅ Dilithium secret key generated: {len(dilithium_secret_key)} bytes")
        
        # Test 3: Balance Encryption/Decryption
        print("\n3. Testing Balance Encryption/Decryption...")
        test_balance = 1234.56
        
        # Encrypt balance
        kyber_ciphertext, encrypted_balance = crypto_service.encrypt_balance(test_balance, kyber_public_key)
        print(f"   ✅ Balance encrypted: {len(kyber_ciphertext)} + {len(encrypted_balance)} bytes")
        
        # Decrypt balance
        decrypted_balance = crypto_service.decrypt_balance(kyber_ciphertext, encrypted_balance, kyber_secret_key)
        print(f"   ✅ Balance decrypted: {decrypted_balance}")
        
        # Verify correctness
        assert abs(test_balance - decrypted_balance) < 0.01, "Balance encryption/decryption failed!"
        print(f"   ✅ Encryption/Decryption verified: {test_balance} == {decrypted_balance}")
        
        # Test 4: Digital Signature
        print("\n4. Testing CRYSTAL-DILITHIUM Digital Signatures...")
        test_message = b"Transaction: Add $100.00 to account 123"
        
        # Sign message
        signature = crypto_service.dilithium_sign(test_message, dilithium_secret_key)
        print(f"   ✅ Message signed: {len(signature)} bytes")
        
        # Verify signature
        is_valid = crypto_service.dilithium_verify(test_message, signature, dilithium_public_key)
        print(f"   ✅ Signature verified: {is_valid}")
        
        assert is_valid, "Digital signature verification failed!"
        
        # Test 5: Invalid Signature Detection
        print("\n5. Testing Invalid Signature Detection...")
        tampered_message = b"Transaction: Add $1000.00 to account 123"  # Changed amount
        is_valid_tampered = crypto_service.dilithium_verify(tampered_message, signature, dilithium_public_key)
        print(f"   ✅ Tampered signature rejected: {not is_valid_tampered}")
        
        assert not is_valid_tampered, "Tampered signature should be invalid!"
        
        print("\n" + "=" * 60)
        print("🎉 All Post-Quantum Cryptography Tests Passed!")
        print("✅ CRYSTAL-Kyber key encapsulation working")
        print("✅ CRYSTAL-DILITHIUM digital signatures working") 
        print("✅ AES encryption working")
        print("✅ Balance encryption/decryption working")
        print("✅ Digital signature verification working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure liboqs-python is installed:")
        print("   pip install liboqs-python")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_post_quantum_crypto()
    sys.exit(0 if success else 1)
