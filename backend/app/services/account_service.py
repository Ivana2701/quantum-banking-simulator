from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from app.db.models import Account, Address, PhoneNumber, GeoLocation, IPAddress, Device, AccountTypeEnum
from app.schemas import AccountCreate
from app.core.security import get_password_hash, verify_password
from app.services.crypto_service import crypto_service
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

    def get_by_id(self, db: Session, account_id: int) -> Account:
        """Get account by account ID"""
        logger.debug(f"Searching for user with account_id: {account_id}")
        user = db.query(Account).filter(Account.account_id == account_id).first()
        if user:
            logger.debug(f"Found user: {user.username} with ID: {user.account_id}")
        else:
            logger.debug(f"No user found with account_id: {account_id}")
        return user

    def create_account(
        self,
        db: Session,
        payload: AccountCreate,
        hashed_password: str
    ) -> Account:
        logger.info(f"Creating new account for username: {payload.username}")
        
        # Check if username already exists
        existing_user = self.get_by_username(db, payload.username)
        if existing_user:
            logger.warning(f"Username already exists: {payload.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists. Please choose a different username."
            )
            
        logger.debug(f"Account details - Username: {payload.username}, Type: {payload.account_type}, Role: customer")
        
        try:
            # Create default encrypted balance of 0.0 using post-quantum cryptography
            # Generate Kyber keypair for this account
            kyber_public_key, kyber_secret_key = crypto_service.generate_kyber_keypair()
            
            # Encrypt initial balance of 0.0 using post-quantum crypto
            kyber_ciphertext, encrypted_balance = crypto_service.encrypt_balance(0.0, kyber_public_key)
            
            # Store the Kyber ciphertext + encrypted balance together
            # Format: kyber_ciphertext_length(4 bytes) + kyber_ciphertext + encrypted_balance
            balance_data = len(kyber_ciphertext).to_bytes(4, 'big') + kyber_ciphertext + encrypted_balance
            
            new_acct = Account(
                username=payload.username,
                full_name=payload.full_name,
                password_hash=hashed_password,
                account_type=AccountTypeEnum("customer"),
                role_id=1,
                encrypted_balance=balance_data
            )
            db.add(new_acct)
            db.flush()  # Flush to get the account_id
            
            # Generate and store post-quantum cryptography keys
            dilithium_public_key, dilithium_secret_key = crypto_service.generate_dilithium_keypair()
            
            # Store encryption keys in separate table
            from app.db.models import AccountEncryption
            encryption_data = AccountEncryption(
                account_id=new_acct.account_id,
                kyber_public_key=kyber_public_key,
                kyber_secret_key=kyber_secret_key,
                dilithium_public_key=dilithium_public_key,
                dilithium_secret_key=dilithium_secret_key
            )
            db.add(encryption_data)
            db.flush()
            
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
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Database integrity error while creating account: {str(e)}")
            
            # Check if it's a username constraint violation
            if "accounts_username_key" in str(e) or "username" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists. Please choose a different username."
                )
            else:
                # Generic integrity error
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Account creation failed due to a data conflict. Please check your information and try again."
                )
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error while creating account: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while creating the account. Please try again later."
            )

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

    def get_account_by_id(self, db: Session, account_id: int) -> Account:
        """Get account by ID"""
        logger.debug(f"Searching for account with ID: {account_id}")
        account = db.query(Account).filter(Account.account_id == account_id).first()
        if not account:
            logger.warning(f"No account found with ID: {account_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found"
            )
        return account

    def get_balance(self, db: Session, account_id: int) -> float:
        """Get decrypted balance for an account using post-quantum cryptography"""
        try:
            account = self.get_account_by_id(db, account_id)
            
            # Get the stored encryption keys for this account
            from app.db.models import AccountEncryption
            encryption_data = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == account_id
            ).first()
            
            if not encryption_data:
                logger.warning(f"No encryption data found for account {account_id}")
                return 0.0
            
            try:
                # Parse the encrypted balance data
                balance_data = account.encrypted_balance
                
                # Extract kyber ciphertext length
                kyber_ciphertext_length = int.from_bytes(balance_data[:4], 'big')
                
                # Extract kyber ciphertext and encrypted balance
                kyber_ciphertext = balance_data[4:4 + kyber_ciphertext_length]
                encrypted_balance = balance_data[4 + kyber_ciphertext_length:]
                
                # Decrypt the balance using post-quantum cryptography
                decrypted_balance = crypto_service.decrypt_balance(
                    kyber_ciphertext, 
                    encrypted_balance, 
                    encryption_data.kyber_secret_key
                )
                
                return decrypted_balance
                
            except Exception as e:
                logger.error(f"Error decrypting balance for account {account_id}: {str(e)}")
                # Fallback to 0.0 if decryption fails
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting balance for account {account_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving account balance"
            )

    def add_money_to_balance(self, db: Session, account_id: int, amount: float) -> float:
        """Add money to an account's balance using post-quantum encryption"""
        try:
            account = self.get_account_by_id(db, account_id)
            
            # Get the stored encryption keys for this account
            from app.db.models import AccountEncryption
            encryption_data = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == account_id
            ).first()
            
            if not encryption_data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Encryption data not found for account"
                )
            
            # Get current balance
            current_balance = self.get_balance(db, account_id)
            
            # Calculate new balance
            new_balance = current_balance + amount
            
            # Encrypt the new balance using post-quantum cryptography
            # Get the Kyber public key from the secret key (for re-encryption)
            kyber_ciphertext, encrypted_balance = crypto_service.encrypt_balance(
                new_balance, 
                encryption_data.kyber_public_key
            )
            
            # Create digital signature of the transaction using CRYSTAL-DILITHIUM
            transaction_message = f"add_money:{account_id}:{amount}:{new_balance}".encode('utf-8')
            signature = crypto_service.dilithium_sign(
                transaction_message, 
                encryption_data.dilithium_secret_key
            )
            
            # Store the signature in a transaction log (optional but good practice)
            # For now, we'll just log it
            logger.info(f"Transaction signed with DILITHIUM for account {account_id}")
            
            # Update the encrypted balance
            balance_data = len(kyber_ciphertext).to_bytes(4, 'big') + kyber_ciphertext + encrypted_balance
            account.encrypted_balance = balance_data
            
            db.commit()
            db.refresh(account)
            
            logger.info(f"Added ${amount} to account {account_id}. New balance: ${new_balance}")
            
            return new_balance
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error adding money to account {account_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating account balance"
            )
        
    def create_admin(
        self,
        db: Session,
        payload: AccountCreate,
        hashed_password: str
    ) -> Account:
        logger.info(f"Creating new admin account for username: {payload.username}")
        
        # Check if username already exists
        existing_user = self.get_by_username(db, payload.username)
        if existing_user:
            logger.warning(f"Username already exists: {payload.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists. Please choose a different username."
            )
            
        logger.debug(f"Account details - Username: {payload.username}, Type: admin, Role: 3")
        
        try:
            # Create default encrypted balance of 0.0 using post-quantum cryptography
            # Generate Kyber keypair for this account
            kyber_public_key, kyber_secret_key = crypto_service.generate_kyber_keypair()
            
            # Encrypt initial balance of 0.0 using post-quantum crypto
            kyber_ciphertext, encrypted_balance = crypto_service.encrypt_balance(0.0, kyber_public_key)
            
            # Store the Kyber ciphertext + encrypted balance together
            # Format: kyber_ciphertext_length(4 bytes) + kyber_ciphertext + encrypted_balance
            balance_data = len(kyber_ciphertext).to_bytes(4, 'big') + kyber_ciphertext + encrypted_balance
            
            new_acct = Account(
                username=payload.username,
                full_name=payload.full_name,
                password_hash=hashed_password,
                account_type=AccountTypeEnum("admin"),
                role_id=3,
                encrypted_balance=balance_data
            )
            db.add(new_acct)
            db.flush()  # Flush to get the account_id
            
            # Generate and store post-quantum cryptography keys
            dilithium_public_key, dilithium_secret_key = crypto_service.generate_dilithium_keypair()
            
            # Store encryption keys in separate table
            from app.db.models import AccountEncryption
            encryption_data = AccountEncryption(
                account_id=new_acct.account_id,
                kyber_public_key=kyber_public_key,
                kyber_secret_key=kyber_secret_key,
                dilithium_public_key=dilithium_public_key,
                dilithium_secret_key=dilithium_secret_key
            )
            db.add(encryption_data)
            db.flush()
            
            db.commit()
            db.refresh(new_acct)
            logger.info(f"Successfully created account with ID: {new_acct.account_id}")
            return new_acct

        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error while creating account: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while creating the account. Please try again later."
            )