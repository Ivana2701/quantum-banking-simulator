#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/customer.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import TransactionBundle, TransactionRead, TransactionCreate
from app.core.security import get_current_user, require_customer
from app.db.database import get_db
from datetime import date
from typing   import Optional
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/customer", tags=["customer"])
acct_svc = AccountService()
tx_svc = TransactionService()

@router.get("/balance", response_model=dict)
def get_balance(
    user = Depends(require_customer),  # Only customers can access this
    db: Session = Depends(get_db)
):
    """Get customer balance - requires customer role"""
    try:
        acct = acct_svc.get_account_by_id(db, user.account_id)
        if not acct:
            raise HTTPException(status_code=404, detail="Account not found")
        
        # Return decrypted balance for customer
        balance = acct_svc.get_balance(db, user.account_id)
        
        # Convert encrypted_balance bytes to base64 string for JSON serialization
        import base64
        encrypted_balance_b64 = base64.b64encode(acct.encrypted_balance).decode('utf-8')
        
        return {
            "balance": float(balance), 
            "encrypted_balance": encrypted_balance_b64,
            "account_id": user.account_id
        }
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting balance for user {user.account_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving balance: {str(e)}")

@router.post("/transfer", response_model=TransactionRead)
def send_money(
    transaction_data: TransactionCreate,
    user = Depends(require_customer),  # Only customers can transfer money
    db: Session = Depends(get_db)
):
    """Transfer money to another account - requires customer role"""
    # Create transaction using the transaction service
    tx = tx_svc.create_transaction(
        db, 
        account_id=user.account_id,  # The account that generated the transaction
        from_account_id=user.account_id, 
        to_account_id=transaction_data.to_account_id, 
        amount=transaction_data.amount
    )
    return tx

@router.get("/transactions", response_model=TransactionBundle)
def view_transactions(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    user = Depends(require_customer),  # Only customers can view their own transactions
    db: Session = Depends(get_db)
):
    """View customer transactions - requires customer role"""
    # Customers see their sent and received transactions (with decrypted amounts for their own transactions)
    acct = acct_svc.get_account_by_id(db, user.account_id)
    sent = tx_svc.get_sent_transactions(db, acct.account_id, from_date, to_date)
    received = tx_svc.get_received_transactions(db, acct.account_id, from_date, to_date)
    
    # Decrypt amounts for customer's own transactions
    for tx in sent:
        tx.amount = tx_svc.decrypt_transaction_amount(db, tx, user.account_id)
    
    for tx in received:
        tx.amount = tx_svc.decrypt_transaction_amount(db, tx, user.account_id)
    
    return TransactionBundle(sent=sent, received=received)