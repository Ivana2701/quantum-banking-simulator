# Required: pip install streamlit pillow
import streamlit as st
from PIL import Image
import os
import base64
import streamlit.components.v1 as components

def show_qb_learn():
    st.title("🎓 QB-Learn: THE Quantum Banking Learning Platform")

    # --- Tabs for different visualizations ---
    tabs = st.tabs([
        "🔑 BB84 Simulation",
        "📊 Performance Comparison",
        "🧬 Quantum 100 Performance",
        "🔒 Transaction Security"
    ])

    # --- Tab 1: BB84 Simulation ---
    with tabs[0]:
        st.header("BB84 Quantum Key Distribution Simulation")
        st.markdown("""
        The BB84 protocol is a quantum key distribution scheme that allows two parties to securely share a cryptographic key.\
        Explore how quantum mechanics enables secure communication, and see how eavesdropping can be detected!
        """)
        st.markdown("---")
        st.info("For a full interactive simulation, see the 'Quantum-Safe Banking Simulation' screen.")
        st.markdown("#### BB84 Key Generation Example (Python)")
        st.code('''\
    from backend.quantum_encryption.bb84 import generate_bb84_key
    alice_bits, alice_bases, bob_results = generate_bb84_key(length=10)
    print("Alice's bits:", alice_bits)
    print("Alice's bases:", alice_bases)
    print("Bob's results:", bob_results)
    ''', language="python")
        st.markdown("---")
        st.markdown("**Key Takeaways:** Quantum states cannot be measured without disturbance. Any eavesdropping attempt will be detected by mismatches in the key.")

    # --- Tab 2: Performance Comparison ---
    with tabs[1]:
        st.header("Model Performance Comparison (QSVM, VQC, Classical)")
        st.markdown("""
        This diagram compares the performance of quantum and classical models for fraud detection.\
        Observe how quantum models (QSVM, VQC) perform relative to classical approaches.
        """)
        img_path = os.path.join("backend", "fraud_detection", "visualizations", "performance_comparison.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Performance Comparison: Quantum vs Classical Models", use_column_width=True)
        else:
            st.warning("Performance comparison image not found.")

    # --- Tab 3: Quantum 100 Performance Comparison ---
    with tabs[2]:
        st.header("Quantum 100 Qubit Performance Comparison")
        st.markdown("""
        This visualization shows the performance of quantum-inspired and quantum models on a 100-qubit dataset.\
        It highlights the scalability and potential of quantum machine learning for large-scale fraud detection.
        """)
        img_path = os.path.join("backend", "fraud_detection", "visualizations", "quantum_100_performance_comparison.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Quantum 100 Performance Comparison", use_column_width=True)
        else:
            st.warning("Quantum 100 performance comparison image not found.")

    # --- Tab 4: Transaction Security ---
    with tabs[3]:
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

        # Mermaid diagram for transaction security flow
        st.markdown("---")
        st.subheader("🔎 Visual: Transaction Security Flow")
        mermaid_code = '''
    flowchart TD
        A["User enters transaction"] --> B["Encrypt with AES or Kyber"]
        B --> C["Send to server"]
        C --> D["Store encrypted in DB"]
        C --> E["Hash for integrity check"]
        C --> F["Sign with private key"]
        F --> G["Signature sent with transaction"]
        E --> H["Hash stored or sent for verification"]
        '''
        components.html(
            f"""
            <div class="mermaid">
            {mermaid_code}
            </div>
            <script type=\"module\">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{"startOnLoad":true}});
            </script>
            """,
            height=400,
        )

        # Post-quantum encryption example (Kyber)
        with st.expander("🔒 Kyber Post-Quantum Encryption Example (Python)"):
            st.markdown("""
            Encrypt transaction data using Kyber (post-quantum safe, requires liboqs-python):
            """)
            st.code('''
    from oqs import KeyEncapsulation

    # Generate keypair
    with KeyEncapsulation("Kyber512") as kem:
        public_key = kem.generate_keypair()
        # Sender encapsulates a shared secret
        ciphertext, shared_secret_enc = kem.encap_secret(public_key)
        # Receiver decapsulates
        shared_secret_dec = kem.decap_secret(ciphertext)
        # Use shared_secret_enc/dec as AES key for symmetric encryption
            ''', language="python")
            st.markdown("**Note:** Kyber is a post-quantum public-key encryption scheme. Use the shared secret as a symmetric key (e.g., for AES). Requires liboqs-python.")

        # Interactive widget placeholder
        st.markdown("---")
        st.subheader("🧪 Try it yourself: Interactive Widgets")
        st.info("Want to see the interactive transaction security flow diagram? Install the Mermaid extension for Streamlit with: `pip install streamlit-extras` and restart your app!")

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

        # 4. Post-quantum (Kyber) code visualization
        st.markdown("---")
        st.subheader("🔒 Post-Quantum Encryption Visualization (Kyber)")
        st.markdown("""
        The following code demonstrates how to use Kyber (a post-quantum public-key encryption scheme) for secure key exchange.\n
        **Note:** This is a code visualization. To run it, you need to install [liboqs-python](https://github.com/open-quantum-safe/liboqs-python).
        """)
        st.code('''
    from oqs import KeyEncapsulation

    # Generate keypair
    with KeyEncapsulation("Kyber512") as kem:
        public_key = kem.generate_keypair()
        # Sender encapsulates a shared secret
        ciphertext, shared_secret_enc = kem.encap_secret(public_key)
        # Receiver decapsulates
        shared_secret_dec = kem.decap_secret(ciphertext)
        # Use shared_secret_enc/dec as AES key for symmetric encryption
        assert shared_secret_enc == shared_secret_dec
        print("Shared secret established!")
        ''', language="python") 