import streamlit as st
import numpy as np
import pandas as pd
import time

# BB84 QKD animation in Streamlit
def bb84_qkd_simulation():
    st.title("Quantum Security Demonstration (BB84)")

    st.markdown("""
    ### 🛡️ Quantum Security Overview

    This simulation demonstrates how our Quantum Key Distribution (QKD) technology, specifically the BB84 protocol, enhances transaction security by reliably detecting interception attempts.

    **Why are you seeing this?**  
    To provide transparency and build confidence in our advanced security methods, this demo illustrates how QKD ensures secure communications.

    **What you'll observe:**  
    - Establishment of secure encryption keys between Alice and Bob.
    - Reliable detection of any interception (by Eve).

    **Recommended viewers:**  
    IT security specialists, compliance officers, auditors, and employees overseeing security infrastructure.
    """)

    eve_interception = st.checkbox("Eve intercepts photons")

    if st.button("Start QKD Simulation"):
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

            alice_photon = alice_bits[i]

            if eve_interception:
                eve_basis = np.random.choice(['+', 'x'])
                eve_measured_bit = alice_photon if alice_bases[i] == eve_basis else np.random.randint(0, 2)
                alice_photon = eve_measured_bit
                row.update({"Eve Basis": eve_basis, "Eve Measured Bit": eve_measured_bit})

            measured_bit = alice_photon if alice_bases[i] == bob_bases[i] else np.random.randint(0, 2)
            bases_match = alice_bases[i] == bob_bases[i]
            bits_match = alice_bits[i] == measured_bit if bases_match else np.nan

            row.update({
                "Bob Basis": bob_bases[i],
                "Bob Measured Bit": measured_bit,
                "Bases Match": "✅" if bases_match else "❌",
                "Bits Match": "✅" if bits_match else ("❌" if bases_match else "N/A")
            })

            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
            result_container.dataframe(df_results.style.applymap(lambda x: 'background-color: lightgreen' if x == '✅' else ('background-color: lightcoral' if x == '❌' else '')))

            time.sleep(0.5)

        st.markdown("---")

        st.markdown("### Reconciliation")
        matched_indices = alice_bases == bob_bases

        st.write(f"Matched Bases: {matched_indices.sum()} out of 10")

        alice_key = alice_bits[matched_indices]
        bob_key = df_results.loc[matched_indices, "Bob Measured Bit"].values

        st.write("Alice's key: ", alice_key)
        st.write("Bob's key: ", bob_key)

        if eve_interception:
            if np.array_equal(alice_key, bob_key):
                st.error("Unexpected: Eve intercepted, but no errors detected (highly unlikely)!")
            else:
                st.error("**Interception detected!** Errors found in keys.")
        else:
            st.success("**No interception detected!** Secure key established.")

# Run the simulation
bb84_qkd_simulation()