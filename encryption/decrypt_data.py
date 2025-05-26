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

# Load AES key practically
with open('aes_key.key', 'rb') as file:
    key = file.read()

cipher_suite = Fernet(key)

# Fetch encrypted personal IDs
cur.execute("SELECT name, encrypted_personal_id FROM customers;")
customers = cur.fetchall()

print("Decrypted Personal IDs:")
for name, encrypted_pid in customers:
    decrypted_pid = cipher_suite.decrypt(bytes(encrypted_pid)).decode()
    print(f"{name}: {decrypted_pid}")

cur.close()
conn.close()
