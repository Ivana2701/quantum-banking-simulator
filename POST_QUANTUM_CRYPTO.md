# Post-Quantum Cryptography Implementation

This document describes the post-quantum cryptography implementation used in the Quantum Banking Simulator for secure customer balance encryption and transaction signing.

## Overview

The system uses three post-quantum cryptographic algorithms:

1. **CRYSTAL-Kyber** for key encapsulation mechanism (KEM)
2. **CRYSTAL-DILITHIUM** for digital signatures
3. **AES-256** for symmetric encryption of sensitive data

## Architecture

### Key Management

Each customer account has associated post-quantum cryptographic keys stored in the `account_encryption` table:

- `kyber_public_key`: Used for encrypting new balance updates
- `kyber_secret_key`: Used for decrypting balance information
- `dilithium_public_key`: Used for verifying transaction signatures  
- `dilithium_secret_key`: Used for signing transactions

### Balance Encryption Process

1. **Account Creation**: 
   - Generate Kyber and Dilithium key pairs
   - Encrypt initial balance of $0.00 using Kyber + AES
   - Store encrypted balance and keys securely

2. **Balance Updates**:
   - Use Kyber to encapsulate a shared secret
   - Use AES-256 with the shared secret to encrypt the balance
   - Sign the transaction with CRYSTAL-DILITHIUM
   - Store the encrypted balance with verification signature

3. **Balance Retrieval**:
   - Extract Kyber ciphertext from stored data
   - Use Kyber secret key to recover the shared secret
   - Use AES-256 to decrypt the actual balance
   - Verify transaction signatures when needed

## Data Format

The encrypted balance is stored as:
```
[4 bytes: Kyber ciphertext length] + [Kyber ciphertext] + [AES encrypted balance]
```

## Security Features

- **Quantum-Resistant**: Algorithms are designed to be secure against quantum computer attacks
- **Forward Secrecy**: New shared secrets are generated for each balance update
- **Integrity Protection**: Digital signatures ensure transaction authenticity
- **Key Isolation**: Each account has unique cryptographic keys

## API Endpoints

### Employee Endpoints

- `GET /employee/customers` - List all customers with encrypted balances
- `GET /employee/customers/{id}/balance` - Get decrypted balance (employee access)
- `POST /employee/customers/{id}/add-money` - Add money with PQC encryption

## Testing

Run the test script to verify the post-quantum cryptography implementation:

```bash
python test_crypto.py
```

This will test:
- Kyber key generation and encapsulation
- Dilithium key generation and signing
- AES encryption/decryption
- End-to-end balance encryption/decryption
- Digital signature verification

## Dependencies

- `liboqs-python`: Python bindings for liboqs (Post-Quantum Cryptography)
- `cryptography`: For AES encryption and key derivation

## Installation

```bash
# Install liboqs system library
# On macOS:
brew install liboqs

# On Ubuntu:
sudo apt-get install liboqs-dev

# Install Python packages
pip install liboqs-python cryptography
```

## Security Considerations

1. **Key Storage**: In production, consider using hardware security modules (HSMs) for key storage
2. **Key Rotation**: Implement periodic key rotation for enhanced security
3. **Audit Logging**: All cryptographic operations are logged for security auditing
4. **Access Control**: Only authorized employees can view/modify customer balances

## Algorithm Details

- **CRYSTAL-Kyber512**: NIST selected post-quantum KEM algorithm
- **CRYSTAL-DILITHIUM2**: NIST selected post-quantum digital signature algorithm  
- **AES-256-CBC**: Symmetric encryption with PBKDF2 key derivation

This implementation provides quantum-resistant security for customer financial data while maintaining performance and usability.
