from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, random
from datetime import datetime, timedelta

load_dotenv()

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

locations = ['Berlin', 'Dubai', 'London', 'Sofia', 'New York', 'Tokyo']

def generate_transaction():
    amount = round(random.uniform(1, 10000), 2)
    days_ago = random.randint(0, 365)
    transaction_time = datetime.now() - timedelta(days=days_ago)
    location = random.choice(locations)
    is_fraud = random.random() < 0.05
    return (amount, transaction_time, location, is_fraud)

def insert_transactions(n=500):
    transactions = [generate_transaction() for _ in range(n)]
    
    query = text("""
        INSERT INTO transactions_qml (amount, transaction_time, location, is_fraud)
        VALUES (:amount, :transaction_time, :location, :is_fraud)
    """)
    
    with engine.connect() as conn:
        conn.execute(query, transactions)
        conn.commit()

if __name__ == "__main__":
    insert_transactions(500)

## or run this script in db -> 
# 
# INSERT INTO transactions_qml (amount, transaction_time, location, is_fraud)
# SELECT
#     ROUND(RANDOM() * 10000, 2) AS amount, -- random amounts from 0 to 10,000
#     NOW() - INTERVAL '1 day' * ROUND(RANDOM() * 365) AS transaction_time, -- random dates within last year
#     (ARRAY['Berlin', 'Dubai', 'London', 'Sofia', 'New York', 'Tokyo'])[floor(random() * 6 + 1)] AS location,
#     RANDOM() < 0.05 AS is_fraud -- approximately 5% fraud rate
# FROM generate_series(1, 500); -- generates 500 transactions
