from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import Account, Address, PhoneNumber, GeoLocation, IPAddress, Device
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
        
        # Set role_id based on account_type if not provided
        role_id = payload.role_id
        if role_id is None:
            role_id = 1 if payload.account_type == "customer" else 2
            
        logger.debug(f"Account details - Email: {payload.email}, Type: {payload.account_type}, Role: {role_id}")
        
        # Create default encrypted balance (empty bytes)
        default_balance = b'\x00\x00\x00\x00'
        
        new_acct = Account(
            username=payload.username,
            full_name=payload.full_name,
            password_hash=hashed_password,
            account_type=payload.account_type,
            role_id=role_id,
            encrypted_balance=default_balance
        )
        db.add(new_acct)
        db.flush()  # Flush to get the account_id
        
        # Add address if provided
        if any([payload.street, payload.city, payload.state, payload.country, payload.postal_code]):
            address = Address(
                street=payload.street,
                city=payload.city,
                state=payload.state,
                country=payload.country,
                postal_code=payload.postal_code
            )
            db.add(address)
            db.flush()
            new_acct.addresses.append(address)
            
        # Add phone if provided
        if payload.phone_number:
            phone = PhoneNumber(
                phone_number=payload.phone_number,
                phone_type=payload.phone_type or "mobile"
            )
            db.add(phone)
            db.flush()
            new_acct.phones.append(phone)
            
        # Add geolocation if provided
        if payload.latitude is not None and payload.longitude is not None:
            geo = GeoLocation(
                latitude=payload.latitude,
                longitude=payload.longitude,
                description=payload.geo_description
            )
            db.add(geo)
            db.flush()
            new_acct.geos.append(geo)
            
        # Add IP address if provided
        if payload.ip_address:
            ip = IPAddress(
                ip_address=payload.ip_address
            )
            db.add(ip)
            db.flush()
            new_acct.ips.append(ip)
            
        # Add device if provided
        if payload.device_name and payload.device_fingerprint:
            device = Device(
                device_name=payload.device_name,
                fingerprint=payload.device_fingerprint
            )
            db.add(device)
            db.flush()
            new_acct.devices.append(device)
        
        db.commit()
        db.refresh(new_acct)
        logger.info(f"Successfully created account with ID: {new_acct.account_id}")
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