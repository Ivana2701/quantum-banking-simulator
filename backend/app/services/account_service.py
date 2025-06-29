from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import Account
from app.schemas import AccountCreate
from app.core.security import get_password_hash, verify_password

class AccountService:
    def get_by_username(self, db: Session, username: str) -> Account:
        return db.query(Account).filter(Account.username == username).first()

    def create_account(
        self,
        db: Session,
        payload: AccountCreate,
        hashed_password: str
    ) -> Account:
        new_acct = Account(
            username=payload.username,
            full_name=payload.full_name,
            email=payload.email,
            password_hash=hashed_password,
            account_type=payload.account_type,
            role_id=payload.role_id
        )
        db.add(new_acct)
        db.commit()
        db.refresh(new_acct)
        return new_acct

    def authenticate_user(
        self,
        db: Session,
        username: str,
        password: str
    ) -> Account:
        user = self.get_by_username(db, username)
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        return user

    def get_all_users(self, db: Session) -> list[Account]:
        return db.query(Account).all()