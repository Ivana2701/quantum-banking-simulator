import psycopg2
from dotenv import load_dotenv
import os
from random import uniform, choice, randint
from datetime import datetime, timedelta

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)

cur = conn.cursor()

locations = ['Sofia', 'London', 'Berlin', 'Paris', 'New York', 'Dubai', 'Tokyo']

def generate_transaction_data(num_records=1000):
    data = []
    for _ in range(num_records):
        amount = round(uniform(10.0, 5000.0), 2)
        transaction_time = datetime.now() - timedelta(minutes=randint(0, 50000))
        location = choice(locations)
        is_fraud = choice([True] * 2 + [False] * 8)  # 20% fraudulent clearly
        data.append((amount, transaction_time, location, is_fraud))
    return data

transactions = generate_transaction_data()

cur.executemany("""
    INSERT INTO transactions_qml (amount, transaction_time, location, is_fraud)
    VALUES (%s, %s, %s, %s);
""", transactions)

conn.commit()
cur.close()
conn.close()

print(f"✅ {len(transactions)} synthetic transactions inserted clearly.")
