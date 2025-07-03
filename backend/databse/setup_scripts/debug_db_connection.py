#!/usr/bin/env python3
"""
Simple database connection test
"""
import os
import sys
sys.path.append('/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend')

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment Variables Debug:")
print(f"USE_SQLITE = {os.getenv('USE_SQLITE')}")
print(f"DATABASE_URL = {os.getenv('DATABASE_URL')}")
print(f"DB_USER = {os.getenv('DB_USER')}")
print(f"DB_PASSWORD = {os.getenv('DB_PASSWORD')}")
print(f"DB_HOST = {os.getenv('DB_HOST')}")
print(f"DB_PORT = {os.getenv('DB_PORT')}")
print(f"DB_NAME = {os.getenv('DB_NAME')}")

print("\n🔧 Database Configuration Logic:")
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"
print(f"USE_SQLITE boolean = {USE_SQLITE}")

if USE_SQLITE:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quantum_banking_dev.db")
    print(f"Using SQLite: {DATABASE_URL}")
else:
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Using PostgreSQL: {DATABASE_URL}")

# Test actual database connection
try:
    from app.db.database import engine
    from sqlalchemy import text
    print(f"\n📊 Actual engine URL: {engine.url}")
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database connection successful")
        
        # Check if tables exist
        try:
            # Use PostgreSQL system tables instead of sqlite_master
            if "postgresql" in str(engine.url):
                tables_result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            else:
                tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            
            tables = [row[0] for row in tables_result]
            print(f"Tables in database: {tables}")
            
            if 'accounts' in tables:
                count_result = conn.execute(text("SELECT COUNT(*) FROM accounts"))
                account_count = count_result.fetchone()[0]
                print(f"Number of accounts: {account_count}")
                
                if account_count > 0:
                    users_result = conn.execute(text("SELECT username FROM accounts LIMIT 5"))
                    usernames = [row[0] for row in users_result]
                    print(f"Sample usernames: {usernames}")
            else:
                print("❌ 'accounts' table does not exist")
        except Exception as table_error:
            print(f"Error checking tables: {table_error}")
        
except Exception as e:
    print(f"Database connection failed: {e}")
