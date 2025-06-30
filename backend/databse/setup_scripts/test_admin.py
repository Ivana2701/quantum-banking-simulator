#!/usr/bin/env python3
"""
Test script to verify admin functionality works
"""
import sys
import os
sys.path.append('/Users/itodorov/code/python/quantum-banking-simulator/backend')

from app.db.database import SessionLocal
from app.db.models import Account, Role, AccountTypeEnum
from app.core.security import get_password_hash
import requests
import json

def test_admin_functionality():
    """Test the admin endpoints"""
    print("🧪 Testing Admin Functionality...")
    
    # Check database first
    db = SessionLocal()
    try:
        # Check if admin user exists
        admin_user = db.query(Account).filter(Account.account_type == AccountTypeEnum.admin).first()
        if admin_user:
            print(f"✅ Found admin user: {admin_user.username}")
        else:
            print("❌ No admin user found in database")
            return
        
        # Check roles
        roles = db.query(Role).all()
        print(f"📋 Available roles: {[r.role_name for r in roles]}")
        
        # Test login with admin user
        print("\n🔐 Testing admin login...")
        login_data = {
            "username": "debuguser",  # assuming this is the admin user
            "password": "debug123",
            "grant_type": "password",
            "scope": "",
            "client_id": "",
            "client_secret": "",
        }
        
        try:
            response = requests.post("http://localhost:8000/auth/token", data=login_data, timeout=5)
            if response.status_code == 200:
                token = response.json().get("access_token")
                print("✅ Admin login successful")
                
                # Test getting user list
                headers = {"Authorization": f"Bearer {token}"}
                users_response = requests.get("http://localhost:8000/admin/users", headers=headers, timeout=5)
                
                if users_response.status_code == 200:
                    users = users_response.json()
                    print(f"✅ Successfully retrieved {len(users)} users")
                    for user in users:
                        print(f"  - {user['username']} ({user['account_type']}, Role ID: {user['role_id']})")
                elif users_response.status_code == 403:
                    print("❌ Access denied - user may not have admin privileges")
                else:
                    print(f"❌ Failed to get users: {users_response.status_code} - {users_response.text}")
            else:
                print(f"❌ Admin login failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the backend server is running on http://localhost:8000")
            
    finally:
        db.close()

if __name__ == "__main__":
    test_admin_functionality()
