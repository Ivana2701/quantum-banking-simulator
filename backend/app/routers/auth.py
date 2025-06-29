# backend/app/routers/auth.py
from datetime import timedelta

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import Token
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
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

    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}
