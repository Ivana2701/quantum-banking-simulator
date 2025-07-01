# backend/app/services/transaction_service.py
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime, date
from fastapi import HTTPException
import logging

from app.db.models import Transaction, Account, AccountEncryption
from app.services.crypto_service import crypto_service

logger = logging.getLogger(__name__)


class TransactionService:
    def get_all_transactions(
        self, db: Session, from_date: date = None, to_date: date = None
    ) -> List[Transaction]:
        query = db.query(Transaction)
        if from_date and to_date:
            start = datetime.combine(from_date, datetime.min.time())
            end   = datetime.combine(to_date,   datetime.max.time())
            query = query.filter(Transaction.created_at.between(start, end))
        return query.all()

    def get_sent_transactions(
        self, db: Session, account_id: int, from_date: date = None, to_date: date = None
    ) -> List[Transaction]:
        query = db.query(Transaction).filter(Transaction.from_account_id == account_id)
        if from_date and to_date:
            start = datetime.combine(from_date, datetime.min.time())
            end   = datetime.combine(to_date,   datetime.max.time())
            query = query.filter(Transaction.created_at.between(start, end))
        return query.all()

    def get_received_transactions(
        self, db: Session, account_id: int, from_date: date = None, to_date: date = None
    ) -> List[Transaction]:
        query = db.query(Transaction).filter(Transaction.to_account_id == account_id)
        if from_date and to_date:
            start = datetime.combine(from_date, datetime.min.time())
            end   = datetime.combine(to_date,   datetime.max.time())
            query = query.filter(Transaction.created_at.between(start, end))
        return query.all()

    def get_transactions_by_account(
        self, db: Session, account_id: int
    ) -> List[Transaction]:
        """Get all transactions (sent and received) for a specific account"""
        return db.query(Transaction).filter(
            (Transaction.from_account_id == account_id) | 
            (Transaction.to_account_id == account_id)
        ).all()

    def create_transaction(
        self, db: Session, account_id: int, from_account_id: int, to_account_id: int, amount: float
    ) -> Transaction:
        """Create a new transaction with encrypted amount using post-quantum cryptography
        
        Args:
            account_id: The account that generated/initiated the transaction
            from_account_id: The source account (sender)
            to_account_id: The destination account (receiver)
            amount: The amount to transfer
        """
        try:
            # Validate accounts exist
            initiating_account = db.query(Account).filter(Account.account_id == account_id).first()
            from_account = db.query(Account).filter(Account.account_id == from_account_id).first()
            to_account = db.query(Account).filter(Account.account_id == to_account_id).first()
            
            if not initiating_account:
                raise HTTPException(status_code=404, detail="Initiating account not found")
            if not from_account:
                raise HTTPException(status_code=404, detail="Source account not found")
            if not to_account:
                raise HTTPException(status_code=404, detail="Destination account not found")
            
            # Validate that sender and receiver are customer accounts only
            if from_account.account_type.value != "customer":
                raise HTTPException(status_code=404, detail="Sender account must be a customer account")
            if to_account.account_type.value != "customer":
                raise HTTPException(status_code=404, detail="Recipient account must be a customer account")
            
            # Get encryption keys for the transaction (use the sender's encryption keys)
            from_encryption = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == from_account_id
            ).first()
            
            if not from_encryption:
                raise HTTPException(status_code=500, detail="Encryption data not found for source account")
            
            # Encrypt the transaction amount using Kyber
            kyber_ciphertext, encrypted_amount = crypto_service.encrypt_balance(
                amount, 
                from_encryption.kyber_public_key
            )
            
            # Store the Kyber ciphertext + encrypted amount together
            # Format: kyber_ciphertext_length(4 bytes) + kyber_ciphertext + encrypted_amount
            amount_data = len(kyber_ciphertext).to_bytes(4, 'big') + kyber_ciphertext + encrypted_amount
            
            # Create digital signature of the transaction using CRYSTAL-DILITHIUM
            transaction_message = f"transfer:{account_id}:{from_account_id}:{to_account_id}:{amount}".encode('utf-8')
            signature = crypto_service.dilithium_sign(
                transaction_message, 
                from_encryption.dilithium_secret_key
            )
            
            # Create transaction record
            transaction = Transaction(
                account_id=account_id,  # The account that generated the transaction
                from_account_id=from_account_id,
                to_account_id=to_account_id,
                encrypted_amount=amount_data,
                is_fraud=False
            )
            
            db.add(transaction)
            db.flush()  # Get transaction ID
            
            # Log the signature for audit purposes
            logger.info(
                f"Transaction {transaction.transaction_id} signed with DILITHIUM "
                f"from account {from_account_id} to {to_account_id}"
            )
            
            # Check sufficient funds and update balances directly to avoid circular import
            # Get current balance of sender
            from_encryption_data = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == from_account_id
            ).first()
            
            if not from_encryption_data:
                db.rollback()
                raise HTTPException(status_code=500, detail="Encryption data not found for sender")
            
            # Get sender's current balance
            sender_balance_data = from_account.encrypted_balance
            sender_kyber_length = int.from_bytes(sender_balance_data[:4], 'big')
            sender_kyber_ciphertext = sender_balance_data[4:4 + sender_kyber_length]
            sender_encrypted_balance = sender_balance_data[4 + sender_kyber_length:]
            
            current_balance = crypto_service.decrypt_balance(
                sender_kyber_ciphertext,
                sender_encrypted_balance,
                from_encryption_data.kyber_secret_key
            )
            
            if current_balance < amount:
                db.rollback()
                raise HTTPException(status_code=400, detail="Insufficient funds")
            
            # Update sender's balance (subtract amount)
            new_sender_balance = current_balance - amount
            sender_kyber_ciphertext, sender_encrypted_balance = crypto_service.encrypt_balance(
                new_sender_balance, from_encryption_data.kyber_public_key
            )
            sender_balance_data = len(sender_kyber_ciphertext).to_bytes(4, 'big') + sender_kyber_ciphertext + sender_encrypted_balance
            from_account.encrypted_balance = sender_balance_data
            
            # Update receiver's balance (add amount)
            to_encryption_data = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == to_account_id
            ).first()
            
            if not to_encryption_data:
                db.rollback()
                raise HTTPException(status_code=500, detail="Encryption data not found for receiver")
            
            receiver_balance_data = to_account.encrypted_balance
            receiver_kyber_length = int.from_bytes(receiver_balance_data[:4], 'big')
            receiver_kyber_ciphertext = receiver_balance_data[4:4 + receiver_kyber_length]
            receiver_encrypted_balance = receiver_balance_data[4 + receiver_kyber_length:]
            
            current_receiver_balance = crypto_service.decrypt_balance(
                receiver_kyber_ciphertext,
                receiver_encrypted_balance,
                to_encryption_data.kyber_secret_key
            )
            
            new_receiver_balance = current_receiver_balance + amount
            receiver_kyber_ciphertext, receiver_encrypted_balance = crypto_service.encrypt_balance(
                new_receiver_balance, to_encryption_data.kyber_public_key
            )
            receiver_balance_data = len(receiver_kyber_ciphertext).to_bytes(4, 'big') + receiver_kyber_ciphertext + receiver_encrypted_balance
            to_account.encrypted_balance = receiver_balance_data
            
            db.commit()
            db.refresh(transaction)
            
            logger.info(f"Transaction {transaction.transaction_id} completed successfully")
            return transaction
            
        except HTTPException:
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating transaction: {str(e)}")
            raise HTTPException(status_code=500, detail="Error creating transaction")

    def decrypt_transaction_amount(
        self, db: Session, transaction: Transaction, account_id: int
    ) -> float:
        """Decrypt transaction amount for display to employees/admins"""
        try:
            # Get encryption keys for the account that created the transaction
            encryption_data = db.query(AccountEncryption).filter(
                AccountEncryption.account_id == transaction.from_account_id
            ).first()
            
            if not encryption_data:
                logger.warning(f"No encryption data found for transaction {transaction.transaction_id}")
                return 0.0
            
            # Parse the encrypted amount data
            amount_data = transaction.encrypted_amount
            
            # Extract kyber ciphertext length
            kyber_ciphertext_length = int.from_bytes(amount_data[:4], 'big')
            
            # Extract kyber ciphertext and encrypted amount
            kyber_ciphertext = amount_data[4:4 + kyber_ciphertext_length]
            encrypted_amount = amount_data[4 + kyber_ciphertext_length:]
            
            # Decrypt the amount using post-quantum cryptography
            decrypted_amount = crypto_service.decrypt_balance(
                kyber_ciphertext, 
                encrypted_amount, 
                encryption_data.kyber_secret_key
            )
            
            return decrypted_amount
            
        except Exception as e:
            logger.error(f"Error decrypting transaction amount for transaction {transaction.transaction_id}: {str(e)}")
            return 0.0
