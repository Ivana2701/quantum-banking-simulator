import streamlit as st
import numpy as np
import pandas as pd
import time
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from PIL import Image
import tempfile
import io

st.set_page_config(page_title="Quantum Demo", layout="wide")
st.title("🔐 Quantum-Safe Banking Security Demonstration")

# ----- Sidebar controls -----
st.sidebar.header("Simulation Settings")
num_photons = st.sidebar.slider("Number of photons (qubits)", 1, 20, 10)
manual_mode = st.sidebar.checkbox("🔧 Manual mode (toggle Alice/Bob bases)", value=False)
eve_interception = st.sidebar.checkbox("🕵️ Eve intercepts photons")
view_mode = st.sidebar.radio("Photon view mode", ["wave", "particle"], horizontal=True)

# Initialize session state if needed
if manual_mode:
    if "alice_bases" not in st.session_state or len(st.session_state.alice_bases) != num_photons:
        st.session_state.alice_bases = np.random.choice(['+', 'x'], size=num_photons).tolist()
    if "bob_bases" not in st.session_state or len(st.session_state.bob_bases) != num_photons:
        st.session_state.bob_bases = np.random.choice(['+', 'x'], size=num_photons).tolist()

# Photon animation
def animate_photon(i, view="wave"):
    container = st.empty()
    if view == "wave":
        for t in np.linspace(0, 2 * np.pi, 20):
            pos = int(20 + 10 * np.sin(t))
            line = " " * pos + "🔵"
            container.markdown(f"**Photon {i+1}:** `{line}`")
            time.sleep(0.03)
    else:
        for _ in range(6):
            container.markdown(f"**Photon {i+1}:** 🔴")
            time.sleep(0.1)
            container.markdown("")
            time.sleep(0.05)

# Generate data
alice_bits = np.random.randint(0, 2, size=num_photons)
alice_bases = st.session_state.alice_bases if manual_mode else np.random.choice(['+', 'x'], size=num_photons)
bob_bases = st.session_state.bob_bases if manual_mode else np.random.choice(['+', 'x'], size=num_photons)

# Manual toggle UI
if manual_mode:
    st.subheader("🎛️ Manual Basis Selection")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Alice's Bases")
        for i in range(num_photons):
            st.session_state.alice_bases[i] = st.selectbox(
                f"Alice[{i}]", ['+', 'x'],
                key=f"alice_basis_{i}",
                index=0 if st.session_state.alice_bases[i] == '+' else 1
            )
    with cols[1]:
        st.markdown("#### Bob's Bases")
        for i in range(num_photons):
            st.session_state.bob_bases[i] = st.selectbox(
                f"Bob[{i}]", ['+', 'x'],
                key=f"bob_basis_{i}",
                index=0 if st.session_state.bob_bases[i] == '+' else 1
            )

# ----- Run Simulation -----
if st.button("🚀 Run Full Quantum Simulation"):
    df_results = pd.DataFrame(columns=[
        "Photon", "Alice Bit", "Alice Basis", "Eve Basis", "Eve Bit",
        "Bob Basis", "Bob Bit", "Bases Match", "Bits Match"
    ])
    result_container = st.empty()

    st.markdown("### 🧮 BB84 Protocol Math")
    st.markdown(r"""
    - If β = + and b = 0 → \(|0⟩\), if b = 1 → \(|1⟩\)
    - If β = × and b = 0 → \(|+⟩ = \frac{1}{\sqrt{2}}(|0⟩ + |1⟩)\), if b = 1 → \(|−⟩ = \frac{1}{\sqrt{2}}(|0⟩ - |1⟩)\)
    """)

    with st.expander("🔬 Interact with a Sample BB84 Circuit"):
        bit = st.selectbox("Alice's Bit", [0, 1], index=0)
        basis = st.selectbox("Alice's Basis", ['+', 'x'], index=0)
        qc_exp = QuantumCircuit(1, 1)
        if bit == 1:
            qc_exp.x(0)
        if basis == 'x':
            qc_exp.h(0)
        qc_exp.measure(0, 0)
        fig_exp = qc_exp.draw(output="mpl")
        buf_exp = io.BytesIO()
        fig_exp.savefig(buf_exp, format="png")
        st.image(buf_exp, caption="Interactive BB84 Preparation Circuit", use_container_width=True)

    with st.expander("🧭 Visual: Bloch Sphere Basis Overlay"):
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Bloch_sphere.svg/500px-Bloch_sphere.svg.png", caption="Bloch Sphere showing + and × bases", use_container_width=True)

    for i in range(num_photons):
        animate_photon(i, view=view_mode)

        alice_bit = alice_bits[i]
        alice_basis = alice_bases[i]
        bob_basis = bob_bases[i]
        qc = QuantumCircuit(1, 1)
        explanation = []

        if alice_bit == 1:
            qc.x(0)
            explanation.append("Alice's bit = 1 → apply X gate → qubit becomes |1⟩.")
        else:
            explanation.append("Alice's bit = 0 → no gate → qubit is |0⟩.")

        if alice_basis == 'x':
            qc.h(0)
            explanation.append("Alice uses X basis → apply H gate → qubit enters superposition.")

        eve_basis = "-"
        eve_bit = "-"
        photon_bit = alice_bit

        if eve_interception:
            eve_basis = np.random.choice(['+', 'x'])
            if eve_basis == 'x':
                qc.h(0)
                explanation.append("Eve uses X basis → applies H gate before measurement.")
            else:
                explanation.append("Eve uses + basis → measures directly.")

            qc.measure(0, 0)
            explanation.append("Eve measures the qubit (may collapse state).")
            qc.barrier()
            qc.reset(0)

            eve_bit = photon_bit if alice_basis == eve_basis else np.random.randint(0, 2)
            if eve_bit == 1:
                qc.x(0)
                explanation.append(f"Eve re-prepares qubit as |1⟩.")
            else:
                explanation.append(f"Eve re-prepares qubit as |0⟩.")
            if eve_basis == 'x':
                qc.h(0)
                explanation.append("Eve used X basis → applies H gate again.")
            photon_bit = eve_bit

        if bob_basis == 'x':
            qc.h(0)
            explanation.append("Bob uses X basis → applies H gate before measurement.")
        else:
            explanation.append("Bob uses + basis → measures directly.")

        qc.measure(0, 0)
        explanation.append("Bob measures the qubit.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            circuit_drawer(qc, output="mpl", filename=tmpfile.name)
            st.image(Image.open(tmpfile.name), caption=f"Photon {i+1} Quantum Circuit", use_container_width=True)

        with st.expander(f"🧠 Explain Photon {i+1} Circuit Step-by-Step"):
            for step in explanation:
                st.markdown(f"- {step}")

        bob_bit = photon_bit if alice_basis == bob_basis else np.random.randint(0, 2)
        bases_match = alice_basis == bob_basis
        bits_match = alice_bit == bob_bit if bases_match else None

        df_results.loc[i] = [
            i+1, alice_bit, alice_basis,
            eve_basis, eve_bit,
            bob_basis, bob_bit,
            "✅" if bases_match else "❌",
            "✅" if bits_match else ("❌" if bases_match else "N/A")
        ]
        result_container.dataframe(df_results)
        time.sleep(0.5)

    st.success("✅ BB84 Simulation Complete!")
    sifted = df_results[df_results["Bases Match"] == "✅"]
    st.markdown(f"🔑 Sifted key: {len(sifted)} bits")

    if eve_interception and any(sifted["Bits Match"] == "❌"):
        st.error("⚠️ Eve was detected! Inconsistencies found.")
    else:
        st.success("🟢 No eavesdropping detected. Secure key established.")

    st.markdown("---")
    st.header("2️⃣ Post-Quantum Cryptography (Kyber)")
    st.markdown("**Sender → encapsulates → ciphertext → receiver decapsulates → shared secret**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("📨 **Sender**")
        st.code("public_key = generate_kyber_keypair()")
    with col2:
        st.write("🔐 **Encapsulation**")
        st.code("ciphertext, secret = encapsulate(public_key)")
    with col3:
        st.write("📩 **Receiver**")
        st.code("shared_secret = decapsulate(ciphertext)")
    st.success("Kyber shared secret: `a4f7c1e2d3b8f901`")
    st.markdown("**Kyber Math:** c = A·s + e (mod q). Based on the hardness of Module-LWE.**")

    st.markdown("---")
    st.header("3️⃣ AES Encryption of Transaction")
    original_transaction = "Send $1000 to Account XYZ"
    aes_encrypted = "e4a9f8d1c9f..."
    st.write("📤 Encrypting a financial transaction...")
    st.code(f"Transaction: {original_transaction}")
    st.code(f"Encrypted (AES): {aes_encrypted}")
    st.success("✅ Encrypted transaction secured with quantum-safe keys!")
    st.markdown(r"""
    **AES Math:**
    - SubBytes → ShiftRows → MixColumns → AddRoundKey
    \[
    C = E_K(P), \quad P = D_K(C)
    \]
    where `E` is AES encryption with key K.
    """)
