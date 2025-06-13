# frontend/components/quantum_encryption.py
import base64
import numpy as np
from cryptography.fernet import Fernet
from qiskit import QuantumCircuit, Aer, execute


def generate_bb84_key(length=30):
    alice_bits = np.random.randint(2, size=length)
    alice_bases = np.random.randint(2, size=length)
    qc = QuantumCircuit(length, length)

    for i in range(length):
        if alice_bits[i]:
            qc.x(i)
        if alice_bases[i]:
            qc.h(i)

    qc.measure(range(length), range(length))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend=simulator, shots=1).result()
    bob_results = np.array([int(bit) for bit in list(result.get_counts().keys())[0][::-1]])

    return alice_bits, alice_bases, bob_results

def encrypt_with_bb84(data, bb84_key):
    key_bytes = bb84_key.tobytes()
    if len(key_bytes) < 32:
        key_bytes += b'0' * (32 - len(key_bytes))
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]

    key = base64.urlsafe_b64encode(key_bytes)
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_with_bb84(encrypted_data, bb84_key):
    key_bytes = bb84_key.tobytes()
    if len(key_bytes) < 32:
        key_bytes += b'0' * (32 - len(key_bytes))
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]

    key = base64.urlsafe_b64encode(key_bytes)
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data).decode()
    return decrypted_data
