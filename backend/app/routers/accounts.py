from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.services.account_service import AccountService
from app.schemas import AccountCreate, AccountRead
from app.db.database import get_db
from app.core.security import get_current_user, get_password_hash

router = APIRouter(prefix="/accounts", tags=["accounts"])
svc    = AccountService()

@router.get("", response_model=List[AccountRead])
def list_accounts(
    db: Session = Depends(get_db),
    current = Depends(get_current_user)
):
    return svc.get_all_users(db)

@router.get("/me", response_model=AccountRead)
def read_current(
    current = Depends(get_current_user)
):
    return current

@router.get("/{account_id}", response_model=AccountRead)
def read_account(
    account_id: int,
    db: Session = Depends(get_db)
):
    acct = svc.get_by_username(db, account_id)
    if not acct:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    return acct

@router.post("/createAccount", response_model=AccountRead, status_code=status.HTTP_201_CREATED)
def create_account(
    payload: AccountCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    # Extract IP address from request
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
    
    # Set the IP address in payload if not already provided
    if not payload.ip_address and client_ip:
        payload.ip_address = client_ip
    
    hashed = get_password_hash(payload.password)
    return svc.create_account(db, payload, hashed)