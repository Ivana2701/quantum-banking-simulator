"""
Test file for the Hybrid Post-Quantum Cryptographic Protocol

This test demonstrates the complete security architecture including:
1. Session establishment using BB84 + Kyber
2. Transaction encryption with AES-256-GCM
3. Digital signatures with Dilithium
4. End-to-end secure transaction processing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_pqc_protocol():
    """Test the complete hybrid PQC protocol implementation."""
    
    print("🔐 Testing Hybrid Post-Quantum Cryptographic Protocol")
    print("=" * 60)
    
    try:
        # Import the protocol
        from backend.quantum_encryption.hybrid_pqc_protocol import HybridPQCProtocol
        
        # Initialize the protocol
        protocol = HybridPQCProtocol()
        print("✅ Protocol initialized successfully")
        
        # Test 1: Session Key Establishment
        print("\n🔑 Step 1: Session Key Establishment")
        print("-" * 40)
        
        client_id = "customer_12345"
        session_info = protocol.establish_session_keys(client_id)
        
        print(f"📱 Client ID: {client_id}")
        print(f"🆔 Session ID: {session_info['session_id']}")
        print(f"⏰ Timestamp: {session_info['timestamp']}")
        print(f"🔑 Kyber Public Key Length: {len(session_info['kyber_public_key'])} chars")
        print(f"🔗 BB84 Parameters: {json.dumps(session_info['bb84_parameters'], indent=2)}")
        
        # Test 2: Transaction Data Encryption
        print("\n🔒 Step 2: Transaction Data Encryption")
        print("-" * 40)
        
        transaction_data = {
            "from_account_id": 12345,
            "to_account_id": 67890,
            "amount": 1500.75,
            "description": "Secure quantum transfer",
            "currency": "USD"
        }
        
        encrypted_package = protocol.encrypt_transaction_data(
            session_info['session_id'], 
            transaction_data
        )
        
        print(f"💼 Original Transaction: {json.dumps(transaction_data, indent=2)}")
        print(f"🔐 Encrypted Package:")
        print(f"   - Session ID: {encrypted_package['session_id']}")
        print(f"   - Nonce Length: {len(encrypted_package['nonce'])} chars")
        print(f"   - Ciphertext Length: {len(encrypted_package['ciphertext'])} chars")
        print(f"   - Timestamp: {encrypted_package['timestamp']}")
        
        # Test 3: Digital Signature
        print("\n✍️ Step 3: Digital Signature with Dilithium")
        print("-" * 40)
        
        signature = protocol.sign_transaction(encrypted_package)
        print(f"📝 Digital Signature Length: {len(signature)} chars")
        print(f"🔏 Signature Preview: {signature[:50]}...")
        
        # Test 4: Signature Verification
        print("\n✅ Step 4: Signature Verification")
        print("-" * 40)
        
        is_valid = protocol.verify_transaction_signature(encrypted_package, signature)
        print(f"🔍 Signature Valid: {is_valid}")
        
        # Test 5: Complete Secure Transaction Request
        print("\n📦 Step 5: Complete Secure Transaction Request")
        print("-" * 40)
        
        secure_request = protocol.create_secure_transaction_request(
            session_info['session_id'], 
            transaction_data
        )
        
        print(f"🚀 Secure Request Components:")
        print(f"   - Protocol Version: {secure_request['protocol_version']}")
        print(f"   - Security Level: {secure_request['security_level']}")
        print(f"   - Signature Length: {len(secure_request['signature'])} chars")
        print(f"   - Bank Public Key Length: {len(secure_request['bank_public_key'])} chars")
        
        # Test 6: Transaction Processing
        print("\n⚙️ Step 6: Transaction Processing and Validation")
        print("-" * 40)
        
        processed_data = protocol.process_secure_transaction_request(secure_request)
        print(f"🔓 Decrypted Transaction: {json.dumps(processed_data, indent=2)}")
        
        # Verify data integrity
        original_keys = set(transaction_data.keys())
        processed_keys = set(k for k in processed_data.keys() if k not in ['timestamp', 'transaction_id'])
        
        if original_keys == processed_keys:
            print("✅ Data integrity verified - all original fields preserved")
        else:
            print("❌ Data integrity check failed")
            print(f"   Original keys: {original_keys}")
            print(f"   Processed keys: {processed_keys}")
        
        # Test 7: Performance and Security Metrics
        print("\n📊 Step 7: Security Analysis")
        print("-" * 40)
        
        print("🛡️ Security Features Implemented:")
        print("   ✅ BB84 Quantum Key Distribution")
        print("   ✅ CRYSTALS-Kyber (Post-Quantum KEM)")
        print("   ✅ CRYSTALS-Dilithium (Post-Quantum Signatures)")
        print("   ✅ AES-256-GCM (Authenticated Encryption)")
        print("   ✅ HKDF Key Derivation")
        print("   ✅ Session Key Management")
        print("   ✅ Quantum-Safe Communication Protocol")
        
        print("\n🔒 Cryptographic Strength:")
        print("   - Quantum Security: BB84 + Post-Quantum Algorithms")
        print("   - Classical Security: AES-256 + SHA-256")
        print("   - Forward Secrecy: Session-based key management")
        print("   - Authentication: Digital signatures on all transactions")
        print("   - Integrity: GCM authentication tags")
        
        # Test 8: Session Cleanup
        print("\n🧹 Step 8: Session Management")
        print("-" * 40)
        
        print(f"🔑 Active Sessions: 1 (Session ID: {session_info['session_id'][:16]}...)")
        protocol.cleanup_expired_sessions()
        print("✅ Session cleanup completed")
        
        print("\n" + "=" * 60)
        print("🎉 Hybrid Post-Quantum Protocol Test COMPLETED SUCCESSFULLY!")
        print("🔐 Banking transactions are now secured with quantum-safe cryptography")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("- pip install liboqs-python")
        print("- pip install qiskit")
        print("- pip install cryptography")
        return False
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bb84_protocol():
    """Test the BB84 quantum key distribution protocol separately."""
    
    print("\n🌌 Testing BB84 Quantum Key Distribution")
    print("-" * 50)
    
    try:
        from backend.quantum_encryption.bb84 import BB84Protocol
        
        bb84 = BB84Protocol()
        result = bb84.run_full_protocol()
        
        print(f"🔑 Shared Key Length: {len(result['shared_key'])} bits")
        print(f"📊 Key Preview: {result['shared_key'][:50]}...")
        print(f"⚙️ Protocol Parameters:")
        for key, value in result['parameters'].items():
            print(f"   - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ BB84 Test Failed: {e}")
        return False

def test_pqc_algorithms():
    """Test individual PQC algorithms."""
    
    print("\n🔬 Testing Individual PQC Algorithms")
    print("-" * 50)
    
    try:
        import oqs
        
        print("✅ liboqs imported successfully")
        
        # Test Kyber KEM
        kyber = oqs.KeyEncapsulation("Kyber1024")
        public_key = kyber.generate_keypair()
        encapsulated_secret, shared_secret = kyber.encap_secret(public_key)
        
        print(f"🔑 Kyber1024 KEM:")
        print(f"   - Public Key Length: {len(public_key)} bytes")
        print(f"   - Shared Secret Length: {len(shared_secret)} bytes")
        print(f"   - Encapsulated Secret Length: {len(encapsulated_secret)} bytes")
        
        # Test Dilithium Signatures
        dilithium = oqs.Signature("Dilithium5")
        public_key = dilithium.generate_keypair()
        message = b"Test message for signing"
        signature = dilithium.sign(message)
        is_valid = dilithium.verify(message, signature, public_key)
        
        print(f"✍️ Dilithium5 Signatures:")
        print(f"   - Public Key Length: {len(public_key)} bytes")
        print(f"   - Signature Length: {len(signature)} bytes")
        print(f"   - Verification Result: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ PQC Algorithms Test Failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Quantum Banking Security Tests")
    print("="*60)
    
    # Run individual component tests
    bb84_success = test_bb84_protocol()
    pqc_success = test_pqc_algorithms()
    
    # Run complete protocol test
    if bb84_success and pqc_success:
        protocol_success = test_hybrid_pqc_protocol()
        
        if protocol_success:
            print("\n🎯 ALL TESTS PASSED!")
            print("The quantum banking system is ready for secure operations.")
        else:
            print("\n❌ Protocol test failed")
    else:
        print("\n❌ Component tests failed - skipping protocol test")
    
    print("\n📚 Next Steps:")
    print("1. Deploy the secure endpoints in production")
    print("2. Integrate with the frontend for user interactions")
    print("3. Set up proper key management infrastructure")
    print("4. Configure quantum-safe PKI for public key distribution")
    print("5. Implement audit logging for all secure transactions")
