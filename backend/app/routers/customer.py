#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/customer.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import TransactionBundle, TransactionRead
from app.core.security import get_current_user
from app.db.database import get_db
from datetime import date
from typing   import Optional
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/customer", tags=["customer"])
acct_svc = AccountService()
tx_svc = TransactionService()

@router.get("/balance", response_model=dict)
def get_balance(
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"encrypted_balance": acct.encrypted_balance}

@router.post("/transfer", response_model=TransactionRead)
def send_money(
    to_account_id: int,
    amount: float,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Add logic in service to create transaction record
    tx = tx_svc.create_transaction(db, from_account_id=user.account_id, to_account_id=to_account_id, amount=amount)
    return tx

@router.get("/transactions", response_model=TransactionBundle)
def view_transactions(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Customers see their sent and received
    acct = acct_svc.get_account_by_id(db, user.account_id)
    sent = tx_svc.get_sent_transactions(db, acct.account_id, from_date, to_date)
    received = tx_svc.get_received_transactions(db, acct.account_id, from_date, to_date)
    return TransactionBundle(sent=sent, received=received)