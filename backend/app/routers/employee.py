#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/employee.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import TransactionRead, AccountRead, BalanceUpdateRequest
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Account, AccountTypeEnum
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/employee", tags=["employee"])
acct_svc = AccountService()
tx_svc = TransactionService()

def require_employee(current_user: Account = Depends(get_current_user)):
    """Dependency to ensure the current user is an employee"""
    if current_user.account_type.value not in ["employee", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Employee access required"
        )
    return current_user

@router.get("/transactions/all", response_model=List[TransactionRead])
def all_transactions(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    user = Depends(require_employee),
    db: Session = Depends(get_db)
):
    return tx_svc.get_all_transactions(db, from_date, to_date)

@router.get("/transactions/search", response_model=List[TransactionRead])
def search_transactions(
    account_id: int,
    user = Depends(require_employee),
    db: Session = Depends(get_db)
):
    return tx_svc.get_transactions_by_account(db, account_id)

@router.get("/customers", response_model=List[AccountRead])
def get_all_customers(
    db: Session = Depends(get_db),
    employee_user: Account = Depends(require_employee)
):
    """Get a list of all customers in the system"""
    logger.info(f"Employee {employee_user.username} requested all customers list")
    
    # Only return customers (not employees or admins)
    customers = db.query(Account).filter(
        Account.account_type == AccountTypeEnum.customer
    ).all()
    
    return customers

@router.get("/customers/{customer_id}/balance")
def get_customer_balance(
    customer_id: int,
    db: Session = Depends(get_db),
    employee_user: Account = Depends(require_employee)
):
    """Get a customer's balance (for employee view)"""
    # Verify customer exists
    customer = db.query(Account).filter(
        Account.account_id == customer_id,
        Account.account_type == AccountTypeEnum.customer
    ).first()
    
    if not customer:
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )
    
    try:
        # Get decrypted balance
        balance = acct_svc.get_balance(db, customer_id)
        
        logger.info(f"Employee {employee_user.username} viewed balance for customer {customer.username}")
        
        return {
            "customer_id": customer_id,
            "balance": balance,
            "customer_username": customer.username
        }
        
    except Exception as e:
        logger.error(f"Error getting customer balance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving customer balance"
        )

@router.post("/customers/{customer_id}/add-money")
def add_money_to_customer(
    customer_id: int,
    balance_request: BalanceUpdateRequest,
    db: Session = Depends(get_db),
    employee_user: Account = Depends(require_employee)
):
    """Add money to a customer's balance using post-quantum encryption"""
    if balance_request.amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Amount must be greater than 0"
        )
    
    # Get the customer
    customer = db.query(Account).filter(
        Account.account_id == customer_id,
        Account.account_type == AccountTypeEnum.customer
    ).first()
    
    if not customer:
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )
    
    try:
        # Use account service to add money
        updated_balance = acct_svc.add_money_to_balance(db, customer_id, balance_request.amount)
        
        logger.info(
            f"Employee {employee_user.username} added ${balance_request.amount} "
            f"to customer {customer.username}'s account"
        )
        
        return {
            "message": f"Successfully added ${balance_request.amount} to {customer.username}'s account",
            "customer_id": customer_id,
            "amount_added": balance_request.amount,
            "new_balance": updated_balance
        }
        
    except Exception as e:
        logger.error(f"Error adding money to customer balance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error updating customer balance"
        )