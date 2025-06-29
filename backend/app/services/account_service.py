from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import Account
from app.schemas import AccountCreate
from app.core.security import get_password_hash, verify_password
import logging

logger = logging.getLogger(__name__)

class AccountService:
    def get_by_username(self, db: Session, username: str) -> Account:
        logger.debug(f"Searching for user with username: {username}")
        user = db.query(Account).filter(Account.username == username).first()
        if user:
            logger.debug(f"Found user: {user.username} with ID: {user.account_id}")  # Fixed: use account_id
        else:
            logger.debug(f"No user found with username: {username}")
        return user

    def create_account(
        self,
        db: Session,
        payload: AccountCreate,
        hashed_password: str
    ) -> Account:
        logger.info(f"Creating new account for username: {payload.username}")
        logger.debug(f"Account details - Email: {payload.email}, Type: {payload.account_type}, Role: {payload.role_id}")
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
        logger.info(f"Successfully created account with ID: {new_acct.account_id}")  # Fixed: use account_id
        return new_acct

    def authenticate_user(
        self,
        db: Session,
        username: str,
        password: str
    ) -> Account:
        logger.info(f"Authentication attempt for username: {username}")
        user = self.get_by_username(db, username)
        
        if not user:
            logger.warning(f"Authentication failed - user not found: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        logger.debug(f"User found, verifying password for: {username}")
        if not verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed - incorrect password for: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        logger.info(f"Authentication successful for username: {username}")
        return user

    def get_all_users(self, db: Session) -> list[Account]:
        return db.query(Account).all()