import streamlit as st
import numpy as np
import pandas as pd
import time

# Enhanced Quantum Security Demonstration (BB84 + PQC + AES)
def quantum_security_simulation():
    st.title("🔐 Quantum-Safe Banking Security Demonstration")

    st.markdown("""
    ## 🛡️ Quantum Security Overview

    This simulation demonstrates:
    - **Quantum Key Distribution (QKD)** using the BB84 protocol for secure key establishment.
    - **Post-Quantum Cryptography (PQC)** with Kyber for quantum-resistant key encapsulation.
    - **Symmetric encryption (AES)** for secure transaction data encryption.

    **Why show this?**  
    To transparently illustrate how we protect your financial data against current and future threats, including quantum computing attacks.

    **Key processes visualized:**
    1. **Quantum Key Distribution (BB84)** – Ensures physical interception detection.
    2. **Post-Quantum Cryptography (Kyber)** – Securely transmits encryption keys mathematically.
    3. **AES Encryption** – Encrypts actual transaction data using secure keys.
    """)

    st.markdown("---")

    eve_interception = st.checkbox("🔴 Eve intercepts photons (simulate QKD interception)")

    if st.button("🚀 Run Quantum Security Simulation"):

        st.header("1️⃣ BB84 Quantum Key Distribution")
        alice_bits = np.random.randint(0, 2, size=10)
        alice_bases = np.random.choice(['+', 'x'], size=10)
        bob_bases = np.random.choice(['+', 'x'], size=10)

        columns = ["Photon", "Alice Bit", "Alice Basis"]
        if eve_interception:
            columns += ["Eve Basis", "Eve Measured Bit"]
        columns += ["Bob Basis", "Bob Measured Bit", "Bases Match", "Bits Match"]

        df_results = pd.DataFrame(columns=columns)
        result_container = st.empty()

        for i in range(10):
            row = {
                "Photon": i + 1,
                "Alice Bit": alice_bits[i],
                "Alice Basis": alice_bases[i],
            }

            photon_bit = alice_bits[i]

            if eve_interception:
                eve_basis = np.random.choice(['+', 'x'])
                eve_measured_bit = photon_bit if alice_bases[i] == eve_basis else np.random.randint(0, 2)
                photon_bit = eve_measured_bit
                row.update({"Eve Basis": eve_basis, "Eve Measured Bit": eve_measured_bit})

            bob_measured_bit = photon_bit if alice_bases[i] == bob_bases[i] else np.random.randint(0, 2)
            bases_match = alice_bases[i] == bob_bases[i]
            bits_match = alice_bits[i] == bob_measured_bit if bases_match else np.nan

            row.update({
                "Bob Basis": bob_bases[i],
                "Bob Measured Bit": bob_measured_bit,
                "Bases Match": "✅" if bases_match else "❌",
                "Bits Match": "✅" if bits_match else ("❌" if bases_match else "N/A")
            })

            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
            result_container.dataframe(df_results.style.applymap(lambda x: 'background-color: lightgreen' if x == '✅' else ('background-color: lightcoral' if x == '❌' else '')))
            time.sleep(0.5)

        matched_indices = alice_bases == bob_bases
        alice_key = alice_bits[matched_indices]
        bob_key = df_results.loc[matched_indices, "Bob Measured Bit"].values

        if eve_interception and not np.array_equal(alice_key, bob_key):
            st.error("🚨 Interception detected in QKD! Errors found.")
        else:
            st.success("✅ QKD secure key established without interception.")

        st.markdown("---")
        st.header("2️⃣ Post-Quantum Cryptography (Kyber) Key Exchange")
        pqc_shared_secret = "a4f7c1e2d3b8f901"  # Simulated shared secret
        pqc_ciphertext = "f39c8e12a9bc6d5..."  # Simulated ciphertext
        st.info(f"Kyber ciphertext: {pqc_ciphertext}")
        st.success(f"Kyber-generated shared secret: {pqc_shared_secret}")

        st.markdown("---")
        st.header("3️⃣ AES Encryption of Transaction")
        original_transaction = "Send $1000 to Account XYZ"
        aes_encrypted_transaction = "e4a9f..."  # Simulated encrypted data
        st.write(f"Original transaction: {original_transaction}")
        st.info(f"AES encrypted transaction: {aes_encrypted_transaction}")

        st.markdown("---")
        st.header("🔒 Final Quantum-Secure Transaction")
        st.success("✅ Transaction successfully encrypted with quantum-safe technology!")

# Run the enhanced quantum security simulation
quantum_security_simulation()
