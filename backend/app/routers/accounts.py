from typing import List
import requests
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.services.account_service import AccountService
from app.schemas import AccountCreate, AccountRead
from app.db.database import get_db
from app.core.security import (
    get_current_user, 
    get_password_hash, 
    require_admin, 
    require_employee_or_admin,
    require_same_user_or_admin
)

router = APIRouter(prefix="/accounts", tags=["accounts"])
svc    = AccountService()

@router.get("", response_model=List[AccountRead])
def list_accounts(
    db: Session = Depends(get_db),
    current = Depends(require_employee_or_admin)  # Only employees and admins can list all accounts
):
    """List all accounts - requires employee or admin role"""
    return svc.get_all_users(db)

@router.get("/me", response_model=AccountRead)
def read_current(
    current = Depends(get_current_user)
):
    """Get current user's account information"""
    return current

@router.get("/{account_id}", response_model=AccountRead)
def read_account(
    account_id: int,
    db: Session = Depends(get_db),
    current = Depends(get_current_user)
):
    """Get specific account information - users can only access their own account unless admin"""
    # Check if user can access this account
    if current.account_type.value != "admin" and current.account_id != account_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only access your own account data."
        )
    
    acct = svc.get_by_id(db, account_id)  # Changed to get_by_id instead of get_by_username
    if not acct:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    return acct

@router.post("/createAccount", response_model=AccountRead, status_code=status.HTTP_201_CREATED)
def create_account(
    payload: AccountCreate,
    request: Request,
    db: Session = Depends(get_db),
    current = Depends(require_admin)  # Only admins can create new accounts
):
    """Create a new account - requires admin role"""
    # Function to get public IP
    def get_public_ip():
        try:
            response = requests.get("https://api.ipify.org?format=json", timeout=5)
            if response.status_code == 200:
                return response.json().get("ip")
        except Exception:
            pass
        return None
    
    # Extract IP address from request headers (fallback)
    client_ip = None
    if hasattr(request, 'client') and request.client:
        client_ip = request.client.host
    
    # Check for forwarded IP (if behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        client_ip = real_ip
    
    # Priority order for IP address:
    # 1. IP provided in payload (from frontend "Check My IP" button)
    # 2. Public IP fetched by backend
    # 3. IP from request headers (usually 127.0.0.1 for local development)
    if not payload.ip_address:
        public_ip = get_public_ip()
        if public_ip:
            payload.ip_address = public_ip
        elif client_ip:
            payload.ip_address = client_ip
    
    hashed = get_password_hash(payload.password)
    return svc.create_account(db, payload, hashed)