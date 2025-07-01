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
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    Get transactions based on user role:
    - Employees/Admins: See all transactions with decrypted amounts
    - Customers: See only their sent and received transactions (amounts encrypted)
    """
    # reload to get full account_type
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if not acct:
        raise HTTPException(404, "Account not found")

    # Role-based access to transactions
    if acct.account_type.value in ["employee", "admin"]:
        # Employees and admins can see all transactions with decrypted amounts
        all_tx = tx_svc.get_all_transactions(db, from_date, to_date)
        # Add decrypted amounts for employee/admin view
        for tx in all_tx:
            tx.amount = tx_svc.decrypt_transaction_amount(db, tx, user.account_id)
        return TransactionBundle(all=all_tx)
    elif acct.account_type.value == "customer":
        # Customers can only see their own transactions (amounts remain encrypted)
        sent     = tx_svc.get_sent_transactions(db, acct.account_id, from_date, to_date)
        received = tx_svc.get_received_transactions(db, acct.account_id, from_date, to_date)
        return TransactionBundle(sent=sent, received=received)
    else:
        raise HTTPException(403, "Access denied")
