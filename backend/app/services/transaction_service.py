# backend/app/services/transaction_service.py
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime, date

from app.db.models import Transaction


class TransactionService:
    def get_all_transactions(
        self, db: Session, from_date: date, to_date: date
    ) -> List[Transaction]:
        start = datetime.combine(from_date, datetime.min.time())
        end   = datetime.combine(to_date,   datetime.max.time())
        return (
            db.query(Transaction)
              .filter(Transaction.created_at.between(start, end))
              .all()
        )

    def get_sent_transactions(
        self, db: Session, account_id: int, from_date: date, to_date: date
    ) -> List[Transaction]:
        start = datetime.combine(from_date, datetime.min.time())
        end   = datetime.combine(to_date,   datetime.max.time())
        return (
            db.query(Transaction)
              .filter(
                  Transaction.from_account_id == account_id,
                  Transaction.created_at.between(start, end)
              )
              .all()
        )

    def get_received_transactions(
        self, db: Session, account_id: int, from_date: date, to_date: date
    ) -> List[Transaction]:
        start = datetime.combine(from_date, datetime.min.time())
        end   = datetime.combine(to_date,   datetime.max.time())
        return (
            db.query(Transaction)
              .filter(
                  Transaction.to_account_id == account_id,
                  Transaction.created_at.between(start, end)
              )
              .all()
        )
