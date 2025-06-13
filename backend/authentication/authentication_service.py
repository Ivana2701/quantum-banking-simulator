# backend/authentication/auth_service.py
import psycopg2
import bcrypt
import os
from dotenv import load_dotenv
from backend.database import db_user, db_pass

load_dotenv()

def authenticate_user(username, password, role):
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=db_user,
        password=db_pass,
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT account_id, password_hash
        FROM accounts a
        JOIN roles r ON a.role_id = r.role_id
        WHERE a.username = %s AND LOWER(r.role_name) = LOWER(%s);
    """, (username, role))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        account_id, stored_hash = result
        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
            return account_id
    return None
