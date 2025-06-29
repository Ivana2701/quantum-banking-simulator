from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
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

@router.post("", response_model=AccountRead, status_code=status.HTTP_201_CREATED)
def create_account(
    payload: AccountCreate,
    db: Session = Depends(get_db)
):
    hashed = get_password_hash(payload.password)
    return svc.create_account(db, payload, hashed)