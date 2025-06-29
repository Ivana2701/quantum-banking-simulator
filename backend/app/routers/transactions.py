#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/transactions.py
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import TransactionBundle, TransactionRead
from app.db.database import get_db
from app.core.security import get_current_user

router = APIRouter(prefix="/transactions", tags=["transactions"])
tx_svc   = TransactionService()
acct_svc = AccountService()

@router.get("", response_model=TransactionBundle)
def get_transactions(
    from_date: date,
    to_date: date,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    # reload to get full account_type
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if not acct:
        raise HTTPException(404, "Account not found")

    if acct.account_type == "employee":
        all_tx = tx_svc.get_all_transactions(db, from_date, to_date)
        return TransactionBundle(all=all_tx)

    # customer: split sent vs received
    sent     = tx_svc.get_sent_transactions(db, acct.account_id, from_date, to_date)
    received = tx_svc.get_received_transactions(db, acct.account_id, from_date, to_date)
    return TransactionBundle(sent=sent, received=received)
