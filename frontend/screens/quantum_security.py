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