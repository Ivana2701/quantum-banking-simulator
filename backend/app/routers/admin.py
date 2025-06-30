from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Account, Role, AccountTypeEnum
from app.schemas import AccountRead, RoleUpdateRequest, RoleUpdateResponse
from app.core.security import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

def require_admin(current_user: Account = Depends(get_current_user)):
    """Dependency to ensure the current user is an admin"""
    if current_user.account_type.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

@router.get("/users", response_model=List[AccountRead])
def get_all_users(
    db: Session = Depends(get_db),
    admin_user: Account = Depends(require_admin)
):
    """Get a list of all users in the system"""
    logger.info(f"Admin {admin_user.username} requested all users list")
    users = db.query(Account).all()
    return users

@router.patch("/users/{user_id}/role", response_model=RoleUpdateResponse)
def update_user_role(
    user_id: int,
    role_request: RoleUpdateRequest,
    db: Session = Depends(get_db),
    admin_user: Account = Depends(require_admin)
):
    """Update a user's role (admin, employee, customer)"""
    new_role = role_request.new_role
    
    # Validate new role
    valid_roles = {"admin": 3, "employee": 2, "customer": 1}
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {list(valid_roles.keys())}"
        )
    
    # Get the user to update
    user = db.query(Account).filter(Account.account_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from changing their own role
    if user.account_id == admin_user.account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )
    
    # Get the role_id for the new role
    role_id = valid_roles[new_role]
    
    # Update the user's role
    user.role_id = role_id
    user.account_type = AccountTypeEnum(new_role)
    
    try:
        db.commit()
        db.refresh(user)
        logger.info(f"Admin {admin_user.username} changed user {user.username}'s role to {new_role}")
        return RoleUpdateResponse(
            message=f"User {user.username}'s role updated to {new_role}",
            user_id=user.account_id,
            new_role=new_role
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user role: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user role"
        )
