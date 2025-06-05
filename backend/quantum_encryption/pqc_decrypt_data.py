import oqs
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def pqc_decrypt():
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    cur = conn.cursor()

    with oqs.KeyEncapsulation('Kyber512') as kem:
        cur.execute("SELECT account_id, ciphertext FROM accounts_pqc;")
        encrypted_data = cur.fetchall()

        decrypted_secrets = {}
        for account_id, ciphertext in encrypted_data:
            shared_secret = kem.decap_secret(bytes(ciphertext))
            decrypted_secrets[account_id] = shared_secret.hex()

    cur.close()
    conn.close()
    
    return decrypted_secrets
