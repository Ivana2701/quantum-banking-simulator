#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/employee.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import TransactionRead
from app.core.security import get_current_user
from app.db.database import get_db
from datetime import date
from typing   import Optional
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/employee", tags=["employee"])
acct_svc = AccountService()
tx_svc = TransactionService()

@router.get("/transactions/all", response_model=List[TransactionRead])
def all_transactions(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if acct.account_type != "employee":
        raise HTTPException(status_code=403, detail="Not authorized")
    return tx_svc.get_all_transactions(db, from_date, to_date)

@router.get("/transactions/search", response_model=List[TransactionRead])
def search_transactions(
    account_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if acct.account_type != "employee":
        raise HTTPException(status_code=403, detail="Not authorized")
    return tx_svc.get_transactions_by_account(db, account_id)