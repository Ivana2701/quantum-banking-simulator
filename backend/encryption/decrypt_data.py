import os
from dotenv import load_dotenv
import psycopg2
from cryptography.fernet import Fernet

load_dotenv()

def decrypt(encrypted_data: bytes):
    with open('aes_key.key', 'rb') as file:
        key = file.read()
    cipher_suite = Fernet(key)
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data

# Optional (for testing purposes only)
if __name__ == "__main__":
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )

    cur = conn.cursor()

    cur.execute("SELECT name, encrypted_personal_id FROM customers;")
    customers = cur.fetchall()

    print("Decrypted Personal IDs:")
    for name, encrypted_pid in customers:
        decrypted_pid = decrypt(bytes(encrypted_pid))
        print(f"{name}: {decrypted_pid}")

    cur.close()
    conn.close()
