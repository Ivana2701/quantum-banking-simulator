#!/usr/bin/env python3
"""
Add test users to your existing PostgreSQL database
"""
import sys
sys.path.append('/Users/itodorov/code/python/quantum-banking-simulator/backend')

from app.db.database import SessionLocal
from app.db.models import Account, Role, AccountTypeEnum
from app.core.security import get_password_hash
from datetime import datetime

def add_test_users_to_postgresql():
    """Add test users to your PostgreSQL database"""
    db = SessionLocal()
    
    try:
        print("🔧 Adding test users to PostgreSQL database...")
        
        # Check existing data
        existing_accounts = db.query(Account).all()
        print(f"📊 Current accounts in database: {len(existing_accounts)}")
        for acc in existing_accounts:
            print(f"  - {acc.username} (ID: {acc.account_id})")
        
        # Check if roles exist
        existing_roles = db.query(Role).all()
        print(f"📋 Current roles: {[role.role_name for role in existing_roles]}")
        
        # Create missing roles if needed
        required_roles = ["admin", "customer", "employee"]
        for role_name in required_roles:
            existing_role = db.query(Role).filter(Role.role_name == role_name).first()
            if not existing_role:
                role = Role(role_name=role_name)
                db.add(role)
                print(f"✅ Created role: {role_name}")
        
        db.commit()
        
        # Get role for new accounts (use existing one or create customer role)
        customer_role = db.query(Role).filter(Role.role_name == "customer").first()
        admin_role = db.query(Role).filter(Role.role_name == "admin").first()
        
        if not customer_role:
            print("❌ No customer role found. Using first available role.")
            customer_role = db.query(Role).first()
        
        # Add test users that don't conflict with existing ones
        test_users = [
            {
                "username": "testuser",
                "full_name": "Test User For Debugging",
                "password": "password123",
                "account_type": AccountTypeEnum.customer,
                "role_id": customer_role.role_id if customer_role else 1,
            },
            {
                "username": "debuguser",
                "full_name": "Debug Admin User", 
                "password": "debug123",
                "account_type": AccountTypeEnum.admin,
                "role_id": admin_role.role_id if admin_role else customer_role.role_id if customer_role else 1,
            }
        ]
        
        for user_data in test_users:
            # Check if user already exists
            existing_user = db.query(Account).filter(Account.username == user_data["username"]).first()
            if not existing_user:
                hashed_password = get_password_hash(user_data["password"])
                
                # Create user with encrypted balance placeholder
                user = Account(
                    username=user_data["username"],
                    full_name=user_data["full_name"],
                    password_hash=hashed_password,
                    account_type=user_data["account_type"],
                    role_id=user_data["role_id"],
                    encrypted_balance=b"encrypted_test_balance"  # Placeholder for encrypted balance
                )
                
                db.add(user)
                print(f"✅ Created test user: '{user_data['username']}' with password: '{user_data['password']}'")
            else:
                print(f"⚠️  User '{user_data['username']}' already exists")
        
        db.commit()
        
        # Verify final state
        all_accounts = db.query(Account).all()
        print(f"\n📊 Final database state:")
        print(f"  Total accounts: {len(all_accounts)}")
        for acc in all_accounts:
            print(f"  - {acc.username} (ID: {acc.account_id}, Type: {acc.account_type})")
        
        print(f"\n🧪 Available test credentials:")
        print(f"  Username: 'testuser', Password: 'password123'")
        print(f"  Username: 'debuguser', Password: 'debug123'")
        print(f"  (Plus your existing users: alice, bob, carol)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    add_test_users_to_postgresql()
