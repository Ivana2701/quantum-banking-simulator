# Required: pip install streamlit pillow
import streamlit as st
from PIL import Image
import os
import base64
import streamlit.components.v1 as components

# Optional imports for interactive BB84 simulation
try:
    import numpy as np
    import pandas as pd
    import time
    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer
    import tempfile
    import io
    QUANTUM_AVAILABLE = True
except ImportError as e:
    QUANTUM_AVAILABLE = False
    st.warning(f"Some quantum simulation features unavailable. Install missing packages: {e}")

def show_qb_learn():
    st.title("🎓 QB-Learn: THE Quantum Banking Learning Platform")

    # --- Tabs for different visualizations ---
    tabs = st.tabs([
        "🔑 BB84 Simulation",
        "🚀 Interactive BB84",
        "📊 Performance Comparison",
        "🧬 Quantum 100 Performance",
        "🔒 Transaction Security",
        "📚 Learn More"
    ])

    # --- Tab 1: BB84 Simulation ---
    with tabs[0]:
        st.header("BB84 Quantum Key Distribution Simulation")
        st.markdown("""
        The BB84 protocol is a quantum key distribution scheme that allows two parties to securely share a cryptographic key.
        Explore how quantum mechanics enables secure communication, and see how eavesdropping can be detected!
        """)
        st.markdown("---")
        st.info("For a full interactive simulation, see the 'Interactive BB84' tab.")
        st.markdown("#### BB84 Key Generation Example (Python)")
        st.code('''
    from backend.quantum_encryption.bb84 import generate_bb84_key
    alice_bits, alice_bases, bob_results = generate_bb84_key(length=10)
    print("Alice's bits:", alice_bits)
    print("Alice's bases:", alice_bases)
    print("Bob's results:", bob_results)
    ''', language="python")
        st.markdown("---")
        st.markdown("**Key Takeaways:** Quantum states cannot be measured without disturbance. Any eavesdropping attempt will be detected by mismatches in the key.")

    # --- Tab 2: Interactive BB84 Simulation ---
    with tabs[1]:
        st.header("🚀 Interactive BB84 Quantum Key Distribution")
        st.markdown("""
        Experience the full BB84 protocol with interactive controls, visual animations, and quantum circuit generation.
        This simulation shows exactly how quantum mechanics protects your banking transactions.
        """)
        
        if not QUANTUM_AVAILABLE:
            st.error("❌ Interactive BB84 simulation requires additional packages.")
            st.markdown("""
            To enable this feature, install the required packages:
            ```bash
            pip install numpy pandas qiskit matplotlib
            ```
            """)
            st.info("📚 For now, check out the other tabs for quantum banking education!")
        else:
            # Sidebar-style controls in the main area
            st.subheader("🎛️ Simulation Settings")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                num_photons = st.slider("Number of photons (qubits)", 1, 20, 10, key="bb84_photons")
            with col2:
                manual_mode = st.checkbox("🔧 Manual mode (toggle Alice/Bob bases)", value=False, key="bb84_manual")
            with col3:
                eve_interception = st.checkbox("🕵️ Eve intercepts photons", key="bb84_eve")
            with col4:
                view_mode = st.radio("Photon view mode", ["wave", "particle"], horizontal=True, key="bb84_view")

            # Initialize session state if needed
            if manual_mode:
                if "alice_bases_bb84" not in st.session_state or len(st.session_state.alice_bases_bb84) != num_photons:
                    st.session_state.alice_bases_bb84 = np.random.choice(['+', 'x'], size=num_photons).tolist()
                if "bob_bases_bb84" not in st.session_state or len(st.session_state.bob_bases_bb84) != num_photons:
                    st.session_state.bob_bases_bb84 = np.random.choice(['+', 'x'], size=num_photons).tolist()

            # Photon animation function
            def animate_photon_bb84(i, view="wave"):
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
            alice_bases = st.session_state.alice_bases_bb84 if manual_mode else np.random.choice(['+', 'x'], size=num_photons)
            bob_bases = st.session_state.bob_bases_bb84 if manual_mode else np.random.choice(['+', 'x'], size=num_photons)

            # Manual toggle UI
            if manual_mode:
                st.subheader("🎛️ Manual Basis Selection")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("#### Alice's Bases")
                    for i in range(num_photons):
                        st.session_state.alice_bases_bb84[i] = st.selectbox(
                            f"Alice[{i}]", ['+', 'x'],
                            key=f"alice_basis_bb84_{i}",
                            index=0 if st.session_state.alice_bases_bb84[i] == '+' else 1
                        )
                with cols[1]:
                    st.markdown("#### Bob's Bases")
                    for i in range(num_photons):
                        st.session_state.bob_bases_bb84[i] = st.selectbox(
                            f"Bob[{i}]", ['+', 'x'],
                            key=f"bob_basis_bb84_{i}",
                            index=0 if st.session_state.bob_bases_bb84[i] == '+' else 1
                        )

            # ----- Run Simulation -----
            if st.button("🚀 Run Full Quantum Simulation", key="bb84_run"):
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
                    bit = st.selectbox("Alice's Bit", [0, 1], index=0, key="bb84_bit")
                    basis = st.selectbox("Alice's Basis", ['+', 'x'], index=0, key="bb84_basis")
                    qc_exp = QuantumCircuit(1, 1)
                    if bit == 1:
                        qc_exp.x(0)
                    if basis == 'x':
                        qc_exp.h(0)
                    qc_exp.measure(0, 0)
                    try:
                        fig_exp = qc_exp.draw(output="mpl")
                        buf_exp = io.BytesIO()
                        fig_exp.savefig(buf_exp, format="png")
                        st.image(buf_exp, caption="Interactive BB84 Preparation Circuit", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Circuit visualization not available: {e}")
                        st.code(str(qc_exp))

                with st.expander("🧭 Visual: Bloch Sphere Basis Overlay"):
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Bloch_sphere.svg/500px-Bloch_sphere.svg.png", caption="Bloch Sphere showing + and × bases", use_container_width=True)

                for i in range(num_photons):
                    try:
                        animate_photon_bb84(i, view=view_mode)
                    except:
                        st.write(f"Processing Photon {i+1}...")

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

                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            circuit_drawer(qc, output="mpl", filename=tmpfile.name)
                            st.image(Image.open(tmpfile.name), caption=f"Photon {i+1} Quantum Circuit", use_container_width=True)
                    except Exception as e:
                        st.write(f"**Photon {i+1} Circuit:**")
                        st.code(str(qc))

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

    # --- Tab 3: Performance Comparison ---
    with tabs[2]:
        st.header("Model Performance Comparison (QSVM, VQC, Classical)")
        st.markdown("""
        This diagram compares the performance of quantum and classical models for fraud detection.
        Observe how quantum models (QSVM, VQC) perform relative to classical approaches.
        """)
        img_path = os.path.join("backend", "fraud_detection", "visualizations", "qsvm_vqc_performance_comparison.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Performance Comparison: Quantum vs Classical Models", use_column_width=True)
        else:
            st.warning("Performance comparison image not found.")

    # --- Tab 4: Quantum 100 Performance Comparison ---
    with tabs[3]:
        st.header("Quantum 100 Qubit Performance Comparison")
        st.markdown("""
        This visualization shows the performance of quantum-inspired and quantum models on a 100-qubit dataset.
        It highlights the scalability and potential of quantum machine learning for large-scale fraud detection.
        """)
        img_path = os.path.join("backend", "fraud_detection", "visualizations", "quantum_100_performance_comparison.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Quantum 100 Performance Comparison", use_column_width=True)
        else:
            st.warning("Quantum 100 performance comparison image not found.")

    # --- Tab 5: Transaction Security ---
    with tabs[4]:
        st.header("Transaction Security: Encryption, Hashing, and Signing")
        st.markdown("""
        In secure banking systems, protecting transaction data is critical. Here's how different cryptographic techniques are used:

        - **Encryption**: Keeps transaction details confidential. Only authorized parties can decrypt and read the data.
        - **Hashing**: Provides data integrity. Any change to the transaction data will result in a different hash.
        - **Digital Signatures**: Prove the authenticity and integrity of a transaction. Only the sender can create a valid signature, and anyone can verify it.
        """)
        st.markdown("---")
        with st.expander("🔐 AES Encryption Example (Python)"):
            st.markdown("""
            Encrypt transaction data using AES-256:
            """)
            st.code('''
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.backends import default_backend
    import os

    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(16)
    plaintext = b"transaction details"
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            ''', language="python")
            st.markdown("**Note:** Store the key securely! Only those with the key can decrypt the data.")
        
        with st.expander("#️⃣ SHA-256 Hashing Example (Python)"):
            st.markdown("""
            Hash transaction data for integrity:
            """)
            st.code('''
    import hashlib

    data = b"transaction details"
    hash_value = hashlib.sha256(data).hexdigest()
            ''', language="python")
            st.markdown("**Note:** Hashing is one-way. You cannot recover the original data from the hash.")
        
        with st.expander("✍️ Ed25519 Digital Signature Example (Python)"):
            st.markdown("""
            Sign and verify transaction data:
            """)
            st.code('''
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    message = b"transaction details"
    signature = private_key.sign(message)
    # To verify:
    public_key.verify(signature, message)
            ''', language="python")
            st.markdown("**Note:** Digital signatures prove authenticity and integrity. Only the holder of the private key can sign.")
        
        st.markdown("---")
        st.markdown("**Summary:**\n- Use **encryption** for confidentiality.\n- Use **hashing** for integrity.\n- Use **digital signatures** for authenticity and non-repudiation.")

        # Interactive widgets
        st.markdown("---")
        st.subheader("🧪 Try it yourself: Interactive Widgets")

        # 1. Hashing widget
        st.markdown("#### Hash a Message (SHA-256)")
        hash_input = st.text_input("Enter a message to hash:", "Hello, quantum world!")
        if hash_input:
            import hashlib
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            st.code(hash_value, language="text")

        # 2. Signing widget (Ed25519)
        st.markdown("#### Sign and Verify a Message (Ed25519)")
        sign_input = st.text_input("Enter a message to sign:", "Quantum signature test")
        if sign_input:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            message = sign_input.encode()
            signature = private_key.sign(message)
            st.code(f"Signature (hex): {signature.hex()}", language="text")
            # Verification
            try:
                public_key.verify(signature, message)
                st.success("Signature verified!")
            except Exception:
                st.error("Signature verification failed.")

        # 3. AES encryption/decryption widget
        st.markdown("#### Encrypt and Decrypt a Message (AES-256)")
        aes_input = st.text_input("Enter a message to encrypt:", "Quantum encryption test")
        if aes_input:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.backends import default_backend
            key = os.urandom(32)
            iv = os.urandom(16)
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(aes_input.encode()) + padder.finalize()
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            st.code(f"Key (base64): {base64.b64encode(key).decode()}\nIV (base64): {base64.b64encode(iv).decode()}\nCiphertext (base64): {base64.b64encode(ciphertext).decode()}", language="text")
            # Decrypt
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
            st.code(f"Decrypted message: {decrypted.decode()}", language="text")

    # --- Tab 6: Learn More ---
    with tabs[5]:
        st.subheader("📚 Understanding Quantum AI Protection")
        
        st.markdown("""
        ### 🤔 How Does Quantum AI Protect Your Money?
        
        Our bank uses two revolutionary types of AI to keep your transactions safe:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🧠 **Classical AI**
            - Uses traditional computer algorithms
            - Analyzes patterns in transaction data
            - Fast and reliable for known fraud types
            - Used by most banks today
            
            **How it works:**
            - Looks at transaction history
            - Compares to known fraud patterns
            - Makes decisions based on rules
            """)
        
        with col2:
            st.markdown("""
            #### ⚛️ **Quantum AI** 
            - Uses quantum computing principles
            - Can detect complex, hidden patterns
            - Better at finding new types of fraud
            - The future of fraud protection
            
            **How it works:**
            - Uses quantum mechanics
            - Processes multiple possibilities at once
            - Finds patterns humans can't see
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔬 **The Science Behind Our Protection**
        
        #### QSVM (Quantum Support Vector Machine)
        """)
        st.latex(r"K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2")
        st.markdown("""
        This equation shows how quantum AI compares transactions in a special "quantum space" 
        where fraud patterns become easier to spot.
        """)
        
        st.markdown("""
        #### VQC (Variational Quantum Classifier)
        """)
        st.latex(r"|\psi(\vec{x}, \vec{\theta})\rangle = U(\vec{\theta}) \cdot \Phi(\vec{x}) |0\rangle")
        st.markdown("""
        This represents how we encode your transaction data into quantum states 
        that our AI can analyze for suspicious patterns.
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🛡️ **What This Means For You**
        
        - **Better Protection**: Quantum AI catches fraud that classical systems miss
        - **Faster Detection**: Real-time analysis of every transaction
        - **Fewer False Alarms**: Smarter AI means less inconvenience for you
        - **Future-Proof**: Ready for tomorrow's sophisticated fraud attempts
        - **Always Learning**: Our AI gets smarter with every transaction
        
        ### 🔒 **Your Privacy & Security**
        
        - Your transaction data is encrypted using quantum-safe methods
        - AI models learn patterns, not personal information
        - All analysis happens in secure, isolated systems
        - You maintain complete control over your account
        """)
        
        st.success("💡 **Bottom Line**: Our quantum-powered AI works 24/7 to protect your money while you focus on what matters most to you!")
