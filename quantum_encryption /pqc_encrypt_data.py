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

    cur.execute("SELECT account_id, encrypted_balance FROM accounts;")
    accounts = cur.fetchall()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts_pqc (
        account_id INT PRIMARY KEY,
        ciphertext BYTEA,
        shared_secret BYTEA
    );
    """)

    for account_id, encrypted_balance in accounts:
        ciphertext, shared_secret = kem.encap_secret(public_key)

        cur.execute("""
        INSERT INTO accounts_pqc (account_id, ciphertext, shared_secret) 
        VALUES (%s, %s, %s)
        ON CONFLICT (account_id) DO UPDATE SET
        ciphertext = EXCLUDED.ciphertext,
        shared_secret = EXCLUDED.shared_secret;
        """, (account_id, ciphertext, shared_secret))

conn.commit()
cur.close()
conn.close()

print("✅ PQC Encryption (Kyber512) successful.")
