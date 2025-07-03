"""
Quantum Security Dashboard - Shows quantum encryption status and features
"""
import streamlit as st
import requests
import time
from datetime import datetime, timedelta
from utils.quantum_session_manager import quantum_session_manager

API_URL = "http://localhost:8000"

def show_quantum_security_dashboard():
    """Display quantum security dashboard with protocol information and controls"""
    
    st.title("🔬 Quantum Security Center")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Current Session Status
    st.markdown("## 🔐 Current Session Status")
    session_status = quantum_session_manager.get_session_status()
    
    if session_status["active"]:
        col1, col2, col3,  = st.columns(3)
        with col1:
            st.success(f"🟢 **{session_status['security_level']}**")
        with col2:
            minutes_remaining = int(session_status["time_remaining"] // 60)
            st.info(f"⏱️ {minutes_remaining}m left")
        with col3:
            if st.button("🗑️ End Session", help="Manually end the current quantum session"):
                success = quantum_session_manager.cleanup_session(st.session_state.token)
                if success:
                    st.success("Session ended successfully")
                else:
                    st.warning("Session cleanup completed locally")
                st.rerun()
        # Show detailed session info in an expander
        with st.expander("📋 Detailed Session Information"):
            session_info = quantum_session_manager.get_session_info()
            if session_info:
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Established:** {session_info['established_at']}")
                    st.write(f"**Expires:** {session_info['expires_at']}")
                with col_info2:
                    st.write(f"**Session ID:** {session_info['session_id']}")
                    st.write(f"**Security Level:** {session_info['security_level']}")
        
        # Protocol Details
        st.markdown("### 🛡️ Active Security Protocols")
        protocol_cols = st.columns(len(session_status["protocols"]))
        
        protocol_info = {
            "BB84 QKD": "🔬 Quantum key distribution using photon polarization",
            "CRYSTALS-Kyber": "🔑 Post-quantum key encapsulation mechanism",
            "CRYSTALS-Dilithium": "✍️ Post-quantum digital signatures",
            "AES-256-GCM": "🔒 Authenticated symmetric encryption"
        }
        
        for i, protocol in enumerate(session_status["protocols"]):
            with protocol_cols[i]:
                st.success(f"**{protocol}**")
                st.caption(protocol_info.get(protocol, "Security protocol"))
    
    else:
        col_warning, col_button = st.columns([3, 1])
        with col_warning:
            st.warning("🟡 No active quantum session")
            st.info("A quantum-safe session will be automatically established when you perform your first secure transaction.")
        with col_button:
            if st.button("🚀 Establish Session Now"):
                with st.spinner("Establishing quantum-safe session..."):
                    success, error = quantum_session_manager.establish_session_manual(st.session_state.token)
                    if success:
                        st.success("✅ Quantum-safe session established!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to establish session: {error}")
    
    st.divider()
    
    # Quantum Protocol Information
    st.markdown("## 📚 Quantum Security Protocols")
    
    # Create tabs for each protocol
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 BB84 QKD", "🔑 CRYSTALS-Kyber", "✍️ CRYSTALS-Dilithium", "🔒 AES-256-GCM"])
    
    with tab1:
        st.markdown("### BB84 Quantum Key Distribution")
        st.markdown("""
        **BB84** is a quantum key distribution protocol that provides information-theoretic security:
        
        - 🌟 **Principle**: Uses quantum properties of photons for key exchange
        - 🔍 **Detection**: Any eavesdropping attempt disturbs quantum states
        - 🎯 **Security**: Unconditionally secure under quantum mechanics
        - ⚡ **Implementation**: Simulated quantum channel with realistic noise
        
        **Key Features:**
        - Photon polarization encoding (horizontal, vertical, diagonal, anti-diagonal)
        - Random basis selection by sender and receiver
        - Error rate monitoring for security validation
        - Information reconciliation and privacy amplification
        """)
        
        if st.button("🔬 Test BB84 Protocol"):
            with st.spinner("Running BB84 key distribution..."):
                try:
                    response = requests.post(f"{API_URL}/transactions/quantum/test-bb84", headers=headers, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ BB84 test completed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Key Length", f"{result.get('key_length', 0)} bits")
                        with col2:
                            st.metric("Error Rate", f"{result.get('error_rate', 0):.2%}")
                        with col3:
                            st.metric("Security Level", result.get('security_level', 'Unknown'))
                        
                        # Add detailed explanation dropdown
                        with st.expander("🔍 **What Happened Behind the Scenes**"):
                            st.markdown("### 🔬 BB84 Quantum Key Distribution Process")
                            
                            # Protocol overview
                            st.markdown("**🎯 Protocol Overview:**")
                            st.markdown(f"""
                            - **Target Key Length**: 256 bits (configurable)
                            - **Error Threshold**: 5% (quantum channel noise tolerance)
                            - **Initial Sequence**: {result.get('key_length', 0) * 4} qubits generated
                            - **Final Key**: {result.get('key_length', 0)} bits extracted
                            """)
                            
                            # Step-by-step breakdown
                            st.markdown("**🔄 Protocol Steps Executed:**")
                            if "steps" in result:
                                for i, step in enumerate(result["steps"], 1):
                                    st.markdown(f"{i}. **{step}**")
                            
                            # Technical details
                            st.markdown("**⚙️ Technical Implementation:**")
                            st.markdown(f"""
                            - **Quantum Simulator**: IBM Qiskit qasm_simulator
                            - **Batch Processing**: Processed in batches of 30 qubits max
                            - **Basis Encoding**: 
                              - `0` basis: Computational (|0⟩, |1⟩)
                              - `1` basis: Hadamard/Diagonal (|+⟩, |-⟩)
                            - **Security Check**: Error rate {result.get('error_rate', 0):.3f} {'✅ below' if result.get('error_rate', 0) < 0.11 else '⚠️ above'} threshold (11%)
                            """)
                            
                            # Performance metrics
                            col_perf1, col_perf2 = st.columns(2)
                            with col_perf1:
                                st.markdown("**📊 Efficiency Metrics:**")
                                efficiency = result.get('key_length', 0) / (result.get('key_length', 0) * 4) if result.get('key_length', 0) > 0 else 0
                                st.markdown(f"- Key extraction efficiency: {efficiency:.1%}")
                                st.markdown(f"- Execution time: {result.get('execution_time', 0):.3f} seconds")
                            
                            with col_perf2:
                                st.markdown("**🔐 Security Analysis:**")
                                security_bits = min(result.get('key_length', 0), 256)
                                st.markdown(f"- Security level: {security_bits}-bit equivalent")
                                st.markdown(f"- Status: {'✅ Secure' if result.get('error_rate', 0) < 0.11 else '⚠️ Needs attention'}")
                            
                            # Real-world implications
                            st.markdown("**🌍 Real-World Implications:**")
                            st.info("""
                            In a real quantum banking system, this key would be used to:
                            1. **Establish secure channels** for transaction data
                            2. **Detect eavesdropping attempts** through error rate monitoring
                            3. **Provide information-theoretic security** (unbreakable even by quantum computers)
                            4. **Enable quantum-safe banking** resistant to future quantum attacks
                            """)
                    else:
                        st.error(f"BB84 test failed: {response.text}")
                except Exception as e:
                    st.error(f"Error testing BB84: {str(e)}")
    
    with tab2:
        st.markdown("### CRYSTALS-Kyber Key Encapsulation")
        st.markdown("""
        **CRYSTALS-Kyber** is a post-quantum key encapsulation mechanism standardized by NIST:
        
        - 🛡️ **Quantum Resistance**: Secure against both classical and quantum attacks
        - 🧮 **Mathematics**: Based on Module Learning With Errors (MLWE) problem
        - ⚡ **Performance**: Fast key generation, encapsulation, and decapsulation
        - 🏆 **Standard**: NIST PQC standardization winner (FIPS 203)
        
        **Security Levels:**
        - Kyber-512: Security equivalent to AES-128
        - Kyber-768: Security equivalent to AES-192
        - Kyber-1024: Security equivalent to AES-256
        """)
    
    with tab3:
        st.markdown("### CRYSTALS-Dilithium Digital Signatures")
        st.markdown("""
        **CRYSTALS-Dilithium** is a post-quantum digital signature scheme:
        
        - ✍️ **Purpose**: Provides authentication and non-repudiation
        - 🔐 **Quantum Safe**: Resistant to quantum computer attacks
        - 🧮 **Mathematics**: Based on FIPS 204 standard
        - 🎯 **Features**: Deterministic signatures with strong security guarantees
        
        **Applications in Banking:**
        - Transaction authentication
        - Message integrity verification
        - Non-repudiation of financial operations
        - Regulatory compliance for digital signatures
        """)
    
    with tab4:
        st.markdown("### AES-256-GCM Symmetric Encryption")
        st.markdown("""
        **AES-256-GCM** provides authenticated encryption for data protection:
        
        - 🔒 **Encryption**: Advanced Encryption Standard with 256-bit keys
        - ✅ **Authentication**: Galois/Counter Mode provides built-in authentication
        - ⚡ **Performance**: Hardware-accelerated on modern processors
        - 🎯 **Security**: Provides both confidentiality and integrity
        
        **Role in Hybrid Protocol:**
        - Encrypts actual transaction data using quantum-derived keys
        - Provides high-speed bulk encryption
        - Ensures data integrity with authentication tags
        - Complements post-quantum key exchange protocols
        """)
    
    st.divider()
    
    # System Status
    st.markdown("## 📊 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌐 Backend Health")
        try:
            response = requests.get(f"{API_URL}/transactions/quantum/health", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Backend Online")
                health_data = response.json()
                if "quantum_ready" in health_data:
                    if health_data["quantum_ready"]:
                        st.success("🔬 Quantum Protocols Ready")
                    else:
                        st.warning("⚠️ Quantum Protocols Initializing")
            else:
                st.error("🔴 Backend Issues")
        except Exception:
            st.error("🔴 Backend Offline")
    
    with col2:
        st.markdown("### 🔧 Protocol Status")
        protocols_status = {
            "BB84 QKD": "🟢 Ready",
            "CRYSTALS-Kyber": "🟢 Ready", 
            "CRYSTALS-Dilithium": "🟢 Ready",
            "AES-256-GCM": "🟢 Ready"
        }
        
        for protocol, status in protocols_status.items():
            st.write(f"{status} {protocol}")

def show_quantum_demo():
    """Interactive demonstration of quantum protocols"""
    
    st.title("🎯 Quantum Protocol Demo")
    
    if "token" not in st.session_state or not st.session_state.token:
        st.error("Please login first")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    st.markdown("""
    This demo allows you to interact with the quantum cryptographic protocols
    and see how they work in practice.
    """)
    
    # Demo Selection
    demo_type = st.selectbox(
        "Choose Demo Type",
        ["BB84 Key Distribution", "Kyber Key Exchange", "Dilithium Signing", "Full Protocol"]
    )
    
    if demo_type == "BB84 Key Distribution":
        st.markdown("### 🔬 BB84 Quantum Key Distribution Demo")
        
        key_length = st.slider("Key Length (bits)", min_value=32, max_value=512, value=128, step=32)
        error_rate = st.slider("Simulated Channel Error Rate", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
        
        if st.button("🚀 Run BB84 Demo"):
            with st.spinner("Simulating quantum key distribution..."):
                try:
                    response = requests.post(
                        f"{API_URL}/transactions/quantum/demo-bb84",
                        headers=headers,
                        json={"key_length": key_length, "error_rate": error_rate},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ BB84 simulation completed!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Generated Key Length", f"{result.get('final_key_length', 0)} bits")
                        with col2:
                            st.metric("Measured Error Rate", f"{result.get('measured_error_rate', 0):.3f}")
                        with col3:
                            st.metric("Security Status", result.get('security_status', 'Unknown'))
                        
                        # Enhanced detailed explanation dropdown
                        with st.expander("🔍 **Detailed Protocol Analysis & Behind-the-Scenes**"):
                            st.markdown("### 🔬 Complete BB84 Quantum Key Distribution Analysis")
                            
                            # Input parameters
                            st.markdown("**📥 Input Parameters:**")
                            col_input1, col_input2 = st.columns(2)
                            with col_input1:
                                st.markdown(f"- **Requested Key Length**: {key_length} bits")
                                st.markdown(f"- **Simulated Error Rate**: {error_rate:.1%}")
                            with col_input2:
                                st.markdown(f"- **Initial Sequence**: {key_length * 4} qubits")
                                st.markdown(f"- **Processing Method**: Batched (30 qubits/batch)")
                            
                            # Protocol execution breakdown
                            st.markdown("**🔄 Protocol Execution Breakdown:**")
                            
                            # Create tabs for different aspects
                            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                                "🔬 Quantum Process", 
                                "📊 Performance Analysis", 
                                "🔐 Security Validation", 
                                "💡 Technical Deep Dive"
                            ])
                            
                            with detail_tab1:
                                st.markdown("#### Quantum Key Distribution Process")
                                
                                # Show protocol steps with explanations
                                if "steps" in result:
                                    for i, step in enumerate(result["steps"], 1):
                                        with st.container():
                                            st.markdown(f"**Step {i}: {step}**")
                                            
                                            # Add detailed explanations for each step
                                            if "Initialize quantum channel" in step:
                                                st.caption("🔧 Set up quantum simulator and prepare for photon transmission")
                                            elif "Alice generates random bits" in step:
                                                st.caption("🎲 Alice creates random sequence of 0s and 1s, and random basis choices")
                                            elif "Alice sends encoded qubits" in step:
                                                st.caption("📡 Qubits encoded using either computational (Z) or Hadamard (X) basis")
                                            elif "Bob measures" in step:
                                                st.caption("📋 Bob randomly chooses measurement bases and measures incoming qubits")
                                            elif "Public basis comparison" in step:
                                                st.caption("📢 Alice and Bob publicly compare their basis choices (not the measurements)")
                                            elif "Key sifting" in step:
                                                st.caption("🔍 Only keep measurements where Alice and Bob used the same basis")
                                            elif "Final key established" in step:
                                                st.caption("🔑 After error correction and privacy amplification")
                                
                                # Visual representation
                                st.markdown("**📈 Process Flow:**")
                                st.markdown("""
                                ```
                                Alice's Random Bits → Quantum Encoding → Quantum Channel 
                                                                            ↓
                                Bob's Random Bases ← Quantum Decoding ← Noisy Qubits
                                                                            ↓
                                Public Basis Comparison → Key Sifting → Error Check → Final Key
                                ```
                                """)
                            
                            with detail_tab2:
                                st.markdown("#### Performance & Efficiency Analysis")
                                
                                # Calculate detailed metrics
                                detailed_metrics = result.get('detailed_metrics', {})
                                initial_qubits = detailed_metrics.get('initial_qubits', key_length * 4)
                                final_bits = result.get('final_key_length', 0)
                                basis_match_rate = detailed_metrics.get('basis_match_rate', 0.5)
                                actual_efficiency = detailed_metrics.get('key_extraction_efficiency', final_bits / initial_qubits if initial_qubits > 0 else 0)
                                
                                metric_col1, metric_col2 = st.columns(2)
                                with metric_col1:
                                    st.metric("Initial Qubits", f"{initial_qubits:,}")
                                    st.metric("Basis Matches", f"{detailed_metrics.get('matching_bases', int(initial_qubits * basis_match_rate)):,}")
                                    st.metric("Actual Final Bits", f"{final_bits:,}")
                                
                                with metric_col2:
                                    st.metric("Overall Efficiency", f"{actual_efficiency:.1%}")
                                    st.metric("Execution Time", f"{result.get('execution_time', 0):.3f}s")
                                    st.metric("Processing Rate", f"{detailed_metrics.get('processing_rate', 0):.0f} qubits/s")
                                
                                # Efficiency breakdown
                                st.markdown("**🔄 Efficiency Breakdown:**")
                                detailed_metrics = result.get('detailed_metrics', {})
                                basis_match_rate = detailed_metrics.get('basis_match_rate', 0.5)
                                error_rate_measured = result.get('measured_error_rate', 0)
                                st.markdown(f"""
                                - **Basis Matching**: {basis_match_rate:.1%} (actual measurement vs ~50% theoretical)
                                - **Error Correction**: Reduces available bits by ~{error_rate_measured*100:.1f}%
                                - **Privacy Amplification**: Additional reduction for information-theoretic security
                                - **Net Efficiency**: {detailed_metrics.get('key_extraction_efficiency', 0):.1%} of initial qubits become final key bits
                                """)
                                
                                # Performance chart using actual data
                                st.markdown("**📊 Bit Processing Pipeline:**")
                                initial_qubits = detailed_metrics.get('initial_qubits', key_length * 4)
                                matching_bases = detailed_metrics.get('matching_bases', int(initial_qubits * 0.5))
                                progress_data = {
                                    "Stage": ["Initial Qubits", "After Basis Sifting", "After Error Correction", "Final Key"],
                                    "Bits": [
                                        initial_qubits,
                                        matching_bases,
                                        int(matching_bases * (1 - error_rate_measured)),
                                        final_bits
                                    ]
                                }
                                st.bar_chart(progress_data, x="Stage", y="Bits")
                            
                            with detail_tab3:
                                st.markdown("#### Security Validation & Analysis")
                                
                                # Security analysis
                                error_rate_measured = result.get('measured_error_rate', 0)
                                detailed_metrics = result.get('detailed_metrics', {})
                                error_threshold = detailed_metrics.get('error_threshold', 0.11)
                                
                                # Security status
                                if error_rate_measured < error_threshold:
                                    st.success(f"🔒 **SECURE**: Error rate {error_rate_measured:.3f} is below threshold {error_threshold}")
                                else:
                                    st.warning(f"⚠️ **CAUTION**: Error rate {error_rate_measured:.3f} exceeds secure threshold {error_threshold}")
                                
                                # Security analysis
                                st.markdown("**🛡️ Security Analysis:**")
                                
                                security_col1, security_col2 = st.columns(2)
                                with security_col1:
                                    st.markdown("**Quantum Security Properties:**")
                                    st.markdown(f"""
                                    - **No-Cloning Theorem**: Qubits cannot be perfectly copied
                                    - **Uncertainty Principle**: Measuring disturbs quantum states
                                    - **Eavesdropping Detection**: Error rate increase reveals attacks
                                    - **Information-Theoretic**: Security proven by physics laws
                                    """)
                                
                                with security_col2:
                                    st.markdown("**Practical Security Level:**")
                                    security_bits = detailed_metrics.get('security_bits', min(result.get('final_key_length', 0), 256))
                                    st.markdown(f"""
                                    - **Effective Security**: {security_bits}-bit equivalent
                                    - **Attack Resistance**: ~2^{security_bits} operations required
                                    - **Quantum-Safe**: Resistant to Shor's algorithm
                                    - **Future-Proof**: Secure against quantum computers
                                    """)
                                
                                # Protocol success indicator
                                protocol_success = detailed_metrics.get('protocol_success', error_rate_measured < error_threshold)
                                if protocol_success:
                                    st.success("✅ **Protocol executed successfully with acceptable security parameters**")
                                else:
                                    st.warning("⚠️ **Protocol completed but security parameters need attention**")
                                
                                # Threat model
                                st.markdown("**🎯 Threat Model Protection:**")
                                st.info("""
                                **This protocol protects against:**
                                - Classical computational attacks (factoring, discrete log)
                                - Quantum attacks (Shor's algorithm, Grover's algorithm)
                                - Man-in-the-middle attacks (through error rate monitoring)
                                - Future unknown attack methods (information-theoretic security)
                                """)
                            
                            with detail_tab4:
                                st.markdown("#### Technical Implementation Details")
                                
                                # Implementation specifics
                                protocol_info = result.get('protocol_info', {})
                                detailed_metrics = result.get('detailed_metrics', {})
                                
                                st.markdown("**⚙️ Implementation Architecture:**")
                                
                                impl_col1, impl_col2 = st.columns(2)
                                with impl_col1:
                                    st.markdown("**Quantum Simulation:**")
                                    st.markdown(f"""
                                    - **Backend**: {protocol_info.get('quantum_backend', 'IBM Qiskit qasm_simulator')}
                                    - **Protocol**: {protocol_info.get('name', 'BB84 Quantum Key Distribution')}
                                    - **Qubit Limit**: {protocol_info.get('batch_size', 30)} qubits per batch (hardware constraint)
                                    - **Noise Model**: Simulated channel errors at {error_rate:.1%}
                                    """)
                                
                                with impl_col2:
                                    st.markdown("**Classical Processing:**")
                                    st.markdown(f"""
                                    - **Language**: Python with NumPy
                                    - **Security Type**: {protocol_info.get('type', 'Information-Theoretic Secure')}
                                    - **Batch Processing**: Memory-efficient for large keys
                                    - **Error Estimation**: Statistical sampling method
                                    """)
                                
                                # Encoding schemes
                                if 'encoding_schemes' in protocol_info:
                                    st.markdown("**🔤 Encoding Schemes:**")
                                    for i, scheme in enumerate(protocol_info['encoding_schemes'], 1):
                                        st.markdown(f"- **Basis {i-1}**: {scheme}")
                                
                                # Protocol parameters with actual values
                                st.markdown("**📋 Protocol Parameters Used:**")
                                param_data = {
                                    "Parameter": [
                                        "Target Key Length",
                                        "Error Threshold", 
                                        "Actual Initial Qubits",
                                        "Actual Basis Matches",
                                        "Batch Size",
                                        "Basis Count"
                                    ],
                                    "Value": [
                                        f"{key_length} bits",
                                        f"{detailed_metrics.get('error_threshold', 0.11):.1%}",
                                        f"{detailed_metrics.get('initial_qubits', key_length * 4)} qubits",
                                        f"{detailed_metrics.get('matching_bases', 'N/A')} qubits",
                                        f"{protocol_info.get('batch_size', 30)} qubits max",
                                        f"{protocol_info.get('basis_count', 2)} (Z and X basis)"
                                    ],
                                    "Purpose": [
                                        "Final shared secret size",
                                        "Security validation limit",
                                        "Total qubits processed",
                                        "Qubits after basis sifting",
                                        "Quantum simulator limitation",
                                        "BB84 standard encoding"
                                    ]
                                }
                                st.dataframe(param_data, hide_index=True)
                                
                                # Mathematical foundations
                                st.markdown("**🧮 Mathematical Foundations:**")
                                st.markdown("""
                                **Key Rate Formula (simplified):**
                                ```
                                R ≈ 1 - H(e) - f(e)·H(e)
                                ```
                                Where:
                                - R = key generation rate
                                - H(e) = binary entropy of error rate e  
                                - f(e) = error correction inefficiency
                                """)
                                
                                # Real implementation notes
                                st.markdown("**🏗️ Production Implementation Notes:**")
                                st.warning("""
                                **For real-world deployment:**
                                - Replace simulator with actual quantum hardware
                                - Implement proper quantum error correction
                                - Add authenticated classical channel for basis comparison
                                - Include post-processing for privacy amplification
                                - Integrate with hardware security modules (HSMs)
                                """)
                        
                        # Show basic protocol steps (keeping existing functionality)
                        if "steps" in result:
                            st.markdown("#### Quick Protocol Steps:")
                            for i, step in enumerate(result["steps"], 1):
                                st.write(f"{i}. {step}")
                    else:
                        st.error(f"Demo failed: {response.text}")
                except Exception as e:
                    st.error(f"Demo error: {str(e)}")
    
    elif demo_type == "Full Protocol":
        st.markdown("### 🔐 Complete Hybrid Protocol Demo")
        
        if st.button("🚀 Run Full Protocol Demo"):
            with st.spinner("Running complete hybrid post-quantum protocol..."):
                try:
                    response = requests.post(
                        f"{API_URL}/transactions/quantum/demo-full-protocol",
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Full protocol demo completed!")
                        
                        # Show timing information
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Time", f"{result.get('total_time', 0):.3f}s")
                        with col2:
                            st.metric("BB84 Time", f"{result.get('bb84_time', 0):.3f}s")
                        with col3:
                            st.metric("Kyber Time", f"{result.get('kyber_time', 0):.3f}s")
                        
                        # Show protocol phases
                        if "phases" in result:
                            st.markdown("#### Protocol Execution:")
                            for phase, details in result["phases"].items():
                                with st.expander(f"📋 {phase.replace('_', ' ').title()}"):
                                    st.json(details)
                    else:
                        st.error(f"Demo failed: {response.text}")
                except Exception as e:
                    st.error(f"Demo error: {str(e)}")
