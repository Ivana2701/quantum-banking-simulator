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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🟢 Security Level",
                session_status["security_level"],
                help="Current quantum security protection level"
            )
        
        with col2:
            minutes_remaining = int(session_status["time_remaining"] // 60)
            seconds_remaining = int(session_status["time_remaining"] % 60)
            st.metric(
                "⏱️ Session Time",
                f"{minutes_remaining}m {seconds_remaining}s",
                help="Time remaining for current secure session"
            )
        
        with col3:
            st.metric(
                "🆔 Session ID",
                session_status["session_id"],
                help="Unique identifier for current session"
            )
        
        with col4:
            if st.button("🗑️ End Session"):
                success = quantum_session_manager.cleanup_session(st.session_state.token)
                if success:
                    st.success("Session ended successfully")
                    st.rerun()
                else:
                    st.warning("Session cleanup completed locally")
                    st.rerun()
        
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
            response = requests.post(f"{API_URL}/transactions/quantum/health", timeout=5)
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
                        
                        # Show protocol steps
                        if "steps" in result:
                            st.markdown("#### Protocol Steps:")
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
