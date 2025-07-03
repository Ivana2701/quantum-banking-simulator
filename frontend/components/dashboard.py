import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
import numpy as np
from cryptography.fernet import Fernet
import base64

# BB84 Quantum Key Generation
def generate_bb84_key(length=16):
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

# Encryption and Decryption
def encrypt_with_bb84(data, bb84_key):
    key = base64.urlsafe_b64encode(bb84_key[:32].tobytes())
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_with_bb84(encrypted_data, bb84_key):
    key = base64.urlsafe_b64encode(bb84_key[:32].tobytes())
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data).decode()
    return decrypted_data

# Streamlit Dashboard Interface
def show_dashboard():
    st.title("Quantum-secure Transaction Dashboard")

    st.subheader("Simulate Quantum-secure Transaction")
    user_data = st.text_input("Enter Transaction Data")

    if st.button("Encrypt and Send Transaction"):
        alice_bits, alice_bases, shared_key = generate_bb84_key()

        encrypted = encrypt_with_bb84(user_data, shared_key)
        st.write("Encrypted Data Sent to Bank:", encrypted)

        decrypted = decrypt_with_bb84(encrypted, shared_key)
        st.write("Bank Employee Decrypted Data:", decrypted)
