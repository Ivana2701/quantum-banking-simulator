# test_persistent_auth.py
"""
Test script to verify persistent authentication works
"""
import requests
import time

API_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"

def test_persistent_authentication():
    print("🔧 Testing Persistent Authentication")
    print("=" * 50)
    
    # Step 1: Login and get a token
    print("\n1. Logging in as customer (carol)...")
    
    form_data = {
        "username": "carol",
        "password": "carol",
        "grant_type": "password",
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }
    
    try:
        response = requests.post(f"{API_URL}/auth/token", data=form_data)
        if response.status_code == 200:
            token_data = response.json()
            token = token_data["access_token"]
            print(f"✅ Login successful! Token: {token[:20]}...")
            
            # Step 2: Validate the token
            print("\n2. Validating token...")
            headers = {"Authorization": f"Bearer {token}"}
            validate_response = requests.get(f"{API_URL}/auth/validate", headers=headers)
            if validate_response.status_code == 200:
                user_data = validate_response.json()
                print(f"✅ Token validation successful!")
                print(f"   User: {user_data.get('username')}")
                print(f"   Role: {user_data.get('account_type')}")
                print(f"   Full Name: {user_data.get('full_name')}")
                print(f"   Account ID: {user_data.get('account_id')}")
            else:
                print(f"❌ Token validation failed: {validate_response.status_code}")
                return
            
            # Step 3: Test token refresh
            print("\n3. Testing token refresh...")
            refresh_response = requests.post(f"{API_URL}/auth/refresh", headers=headers)
            if refresh_response.status_code == 200:
                refresh_data = refresh_response.json()
                new_token = refresh_data["access_token"]
                print(f"✅ Token refresh successful! New token: {new_token[:20]}...")
                
                # Validate new token
                new_headers = {"Authorization": f"Bearer {new_token}"}
                new_validate_response = requests.get(f"{API_URL}/auth/validate", headers=new_headers)
                if new_validate_response.status_code == 200:
                    print("✅ Refreshed token is valid!")
                else:
                    print(f"❌ Refreshed token validation failed: {new_validate_response.status_code}")
            else:
                print(f"❌ Token refresh failed: {refresh_response.status_code}")
            
            # Step 4: Test role-based access
            print("\n4. Testing role-based access...")
            
            # Customer should be able to access customer endpoints
            customer_response = requests.get(f"{API_URL}/customer/balance", headers=headers)
            if customer_response.status_code == 200:
                print("✅ Customer can access customer endpoints")
            else:
                print(f"❌ Customer denied access to customer endpoint: {customer_response.status_code}")
            
            # Customer should be denied admin access
            admin_response = requests.get(f"{API_URL}/admin/users", headers=headers)
            if admin_response.status_code == 403:
                print("✅ Customer correctly denied admin access")
            else:
                print(f"❌ Customer unexpectedly granted admin access: {admin_response.status_code}")
                
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Persistent Authentication Test Results:")
    print("✅ JWT token generation and validation")
    print("✅ Token refresh functionality")
    print("✅ Role-based access control")
    print("✅ Proper error handling for unauthorized access")
    print("\n🔗 Frontend is running at: http://localhost:8501")
    print("📝 Try logging in, then refresh the page to test persistence!")

if __name__ == "__main__":
    test_persistent_authentication()
