# test_existing_users.py
"""
Test authentication with existing users in the database
"""
import requests

API_URL = "http://localhost:8000"

def test_existing_users():
    """Test login with existing users using common passwords"""
    
    # List of existing users from the database output
    existing_users = ["alice", "bob", "carol", "ilia"]
    
    # Common passwords to try
    common_passwords = ["password", "123456", "admin", "test", "alice", "bob", "carol", "ilia"]
    
    print("🔍 Testing existing users with common passwords...")
    print("=" * 60)
    
    successful_logins = []
    
    for username in existing_users:
        print(f"\n🧪 Testing user: {username}")
        
        for password in common_passwords:
            form_data = {
                "username": username,
                "password": password,
                "grant_type": "password",
                "scope": "",
                "client_id": "",
                "client_secret": "",
            }
            
            try:
                response = requests.post(f"{API_URL}/auth/token", data=form_data, timeout=5)
                if response.status_code == 200:
                    token_data = response.json()
                    token = token_data["access_token"]
                    
                    # Get user info
                    headers = {"Authorization": f"Bearer {token}"}
                    user_response = requests.get(f"{API_URL}/auth/validate", headers=headers)
                    if user_response.status_code == 200:
                        user_data = user_response.json()
                        role = user_data.get("account_type")
                        
                        print(f"✅ SUCCESS: {username}/{password} - Role: {role}")
                        successful_logins.append({
                            "username": username,
                            "password": password,
                            "role": role,
                            "token": token[:20] + "..."
                        })
                        break  # Stop trying passwords for this user
                    
            except Exception as e:
                continue  # Try next password
        else:
            print(f"❌ No successful login for {username}")
    
    print("\n" + "=" * 60)
    print("📋 SUCCESSFUL LOGIN CREDENTIALS:")
    print("=" * 60)
    
    if successful_logins:
        for login in successful_logins:
            print(f"👤 Username: {login['username']}")
            print(f"🔑 Password: {login['password']}")
            print(f"🎭 Role: {login['role']}")
            print(f"🎫 Token: {login['token']}")
            print("-" * 40)
    else:
        print("❌ No successful logins found!")
        print("💡 You may need to create test users or check the database.")

if __name__ == "__main__":
    test_existing_users()
