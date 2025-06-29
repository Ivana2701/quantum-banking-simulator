# backend/app/utils.py
# Utility functions for the Quantum Banking Simulator

def format_currency(amount: float) -> str:
    """Format a number as currency"""
    return f"${amount:,.2f}"

def validate_account_number(account_number: str) -> bool:
    """Validate account number format"""
    return len(account_number) == 10 and account_number.isdigit()