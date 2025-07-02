"""
Frontend configuration for quantum-safe banking system
Handles API communication and quantum-safe protocol settings
"""
import os
import requests

API_URL = "http://localhost:8000"

def get_quantum_safe_enabled():
    """
    Get quantum-safe protocol setting from backend API
    Falls back to environment variable if API is unavailable
    
    Returns:
        bool: True if quantum-safe protocol should be used
    """
    try:
        # First try to get setting from backend
        response = requests.get(f"{API_URL}/config/quantum-safe", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("quantum_safe_enabled", True)
    except:
        pass
    
    # Fallback to environment variable
    return os.getenv("USE_QUANTUM_SAFE_PROTOCOL", "true").lower() in ("true", "1", "yes", "on")

def should_use_quantum_safe():
    """
    Determine if quantum-safe protocol should be used for transactions
    
    Returns:
        bool: True if quantum-safe protocol should be used
    """
    return get_quantum_safe_enabled()

# Cache the setting to avoid repeated API calls
_quantum_safe_enabled = None

def get_cached_quantum_safe_setting():
    """
    Get cached quantum-safe setting, with one-time API call
    
    Returns:
        bool: True if quantum-safe protocol should be used
    """
    global _quantum_safe_enabled
    if _quantum_safe_enabled is None:
        _quantum_safe_enabled = get_quantum_safe_enabled()
    return _quantum_safe_enabled

def refresh_quantum_safe_setting():
    """
    Force refresh of quantum-safe setting from backend
    """
    global _quantum_safe_enabled
    _quantum_safe_enabled = None
