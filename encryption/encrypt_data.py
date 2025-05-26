import os
from dotenv import load_dotenv
import psycopg2
from cryptography.fernet import Fernet

# Load environment variables from .env file
load_dotenv()

# Get database credentials securely from environment variables
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)

cur = conn.cursor()

# Generate AES Key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Save AES key for later use
with open('aes_key.key', 'wb') as file:
    file.write(key)

# Function to encrypt data
def encrypt(data):
    return cipher_suite.encrypt(data.encode())

# Encrypt sensitive data practically
customers_data = [
    ('John Doe', 'john@example.com', encrypt('1234567890')),
    ('Jane Smith', 'jane@example.com', encrypt('9876543210'))
]

# Clear previous data
cur.execute("DELETE FROM transactions;")
cur.execute("DELETE FROM accounts;")
cur.execute("DELETE FROM customers;")

# Insert encrypted customer data practically
cur.executemany(
    "INSERT INTO customers (name, email, encrypted_personal_id) VALUES (%s, %s, %s);",
    customers_data
)

# Fetch customer IDs practically to link accounts
cur.execute("SELECT customer_id FROM customers;")
customer_ids = cur.fetchall()

# Encrypt account balances practically
accounts_data = [
    (customer_ids[0][0], encrypt('500')),
    (customer_ids[1][0], encrypt('1000'))
]

# Insert accounts data practically
cur.executemany(
    "INSERT INTO accounts (customer_id, encrypted_balance) VALUES (%s, %s);",
    accounts_data
)

# Fetch account IDs practically
cur.execute("SELECT account_id FROM accounts;")
account_ids = cur.fetchall()

# Encrypt transactions practically
transactions_data = [
    (account_ids[0][0], encrypt('100')),
    (account_ids[1][0], encrypt('200'))
]

# Insert transactions practically
cur.executemany(
    "INSERT INTO transactions (account_id, encrypted_amount) VALUES (%s, %s);",
    transactions_data
)

conn.commit()
cur.close()
conn.close()

print("✅ Data encrypted and stored successfully.")
