#!/usr/bin/env python3
"""
Script to create an admin user with username 'admin' and password 'adminpass'
Note: Password must be at least 8 characters due to validation requirements
"""
import sys
import os
sys.path.append('/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend')

from app.db.database import SessionLocal
from app.db.models import Account, Role, AccountTypeEnum
from app.services.account_service import AccountService
from app.schemas import AccountCreate
from app.core.security import get_password_hash

def create_admin_user():
    """Create an admin user with username 'admin' and password 'adminpass'"""
    print("Creating admin user...")
    
    db = SessionLocal()
    account_service = AccountService()
    
    try:
        # Check if admin user already exists
        existing_admin = account_service.get_by_username(db, "admin")
        if existing_admin:
            print("Admin user already exists!")
            print(f"   Username: {existing_admin.username}")
            print(f"   Account Type: {existing_admin.account_type}")
            print(f"   Account ID: {existing_admin.account_id}")
            return existing_admin
        
        # Create AccountCreate payload
        # Note: Using 'adminpass' because minimum password length is 8 characters
        admin_payload = AccountCreate(
            username="admin",
            full_name="System Administrator",
            password="admin123",
            account_type=AccountTypeEnum.admin
        )
        
        # Hash the password
        hashed_password = get_password_hash(admin_payload.password)
        
        # Create the admin account using the create_admin function
        print("Creating admin account...")
        admin_account = account_service.create_admin(
            db=db,
            payload=admin_payload,
            hashed_password=hashed_password
        )
        
        print("Admin user created successfully!")
        print(f"   Username: {admin_account.username}")
        print(f"   Account ID: {admin_account.account_id}")
        
        return admin_account
        
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Admin User Creation Script")
    print("=" * 50)
    
    try:
        admin_account = create_admin_user()
        print("\nScript completed successfully!")
        print("\nLogin credentials:")
        print("   Username: admin")
        print("   Password: admin123")
        print("\nNote: Password is 'admin123' (8 chars) due to validation requirements")
        
    except Exception as e:
        print(f"\n💥 Script failed with error: {str(e)}")
        sys.exit(1)
