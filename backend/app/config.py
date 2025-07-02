"""
Configuration module for quantum-safe banking system
Handles environment variables and system-wide settings
"""
import os

def get_quantum_safe_enabled():
    """
    Get whether quantum-safe protocol should be used based on environment variable
    
    Returns:
        bool: True if quantum-safe protocol should be used, False otherwise
    """
    return os.getenv("USE_QUANTUM_SAFE_PROTOCOL", "true").lower() in ("true", "1", "yes", "on")

def is_debug_mode():
    """
    Get whether debug mode is enabled
    
    Returns:
        bool: True if debug mode is enabled
    """
    return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")

# Global configuration values
QUANTUM_SAFE_ENABLED = get_quantum_safe_enabled()
DEBUG_MODE = is_debug_mode()
