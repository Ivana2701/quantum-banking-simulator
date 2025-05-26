import oqs
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)

cur = conn.cursor()

with oqs.KeyEncapsulation('Kyber512') as kem:
    public_key = kem.generate_keypair()
    secret_key = kem.export_secret_key()

    def pqc_decrypt(ciphertext):
        shared_secret = kem.decap_secret(bytes(ciphertext))
        return shared_secret

    cur.execute("SELECT account_id, ciphertext FROM accounts_pqc;")
    encrypted_data = cur.fetchall()

    print("✅ PQC Decrypted Shared Secrets:")
    for account_id, ciphertext in encrypted_data:
        shared_secret = pqc_decrypt(ciphertext)
        print(f"Account ID {account_id}: Shared Secret (hex) = {shared_secret.hex()}")

cur.close()
conn.close()
