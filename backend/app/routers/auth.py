# backend/app/routers/auth.py
from datetime import timedelta

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import Token, AccountRead
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
from app.services.account_service import AccountService

router = APIRouter(prefix="/auth", tags=["auth"])
svc = AccountService()

@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # Will raise 401 if invalid
    user = svc.authenticate_user(db, form_data.username, form_data.password)

    # Include user information in the token for better access control
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.account_type.value,
            "account_id": user.account_id,
            "full_name": user.full_name
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/validate", response_model=AccountRead)
def validate_token(current_user = Depends(get_current_user)):
    """
    Validate the current token and return user information.
    This endpoint helps maintain persistent sessions.
    """
    return current_user

@router.post("/refresh", response_model=Token)
def refresh_token(current_user = Depends(get_current_user)):
    """
    Refresh the access token for the current user.
    """
    access_token = create_access_token(
        data={
            "sub": current_user.username,
            "role": current_user.account_type.value,
            "account_id": current_user.account_id,
            "full_name": current_user.full_name
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}
