import os
from dotenv import load_dotenv
import psycopg2
from cryptography.fernet import Fernet

load_dotenv()

def encrypt(data):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    with open('aes_key.key', 'wb') as file:
        file.write(key)

    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data

def encrypt_customer_data():
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    cur = conn.cursor()

    customers_data = [
        ('John Doe', 'john@example.com', encrypt('1234567890')),
        ('Jane Smith', 'jane@example.com', encrypt('9876543210'))
    ]

    cur.execute("DELETE FROM transactions;")
    cur.execute("DELETE FROM accounts;")
    cur.execute("DELETE FROM customers;")

    cur.executemany(
        "INSERT INTO customers (name, email, encrypted_personal_id) VALUES (%s, %s, %s);",
        customers_data
    )

    conn.commit()
    cur.close()
    conn.close()
