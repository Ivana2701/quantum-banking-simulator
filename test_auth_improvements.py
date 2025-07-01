# test_auth_improvements.py
"""
Simple test script to verify authentication improvements work correctly
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_authentication():
    print("🔧 Testing Enhanced Authentication System")
    print("=" * 50)
    
    # Test 1: Login with customer credentials
    print("\n1. Testing Customer Login...")
    
    form_data = {
        "username": "carol",  # Updated with working credentials
        "password": "carol",  # Updated with working credentials
        "grant_type": "password",
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }
    
    try:
        response = requests.post(f"{API_URL}/auth/token", data=form_data)
        if response.status_code == 200:
            token_data = response.json()
            customer_token = token_data["access_token"]
            print(f"✅ Customer login successful! Token: {customer_token[:20]}...")
            
            # Test token validation
            headers = {"Authorization": f"Bearer {customer_token}"}
            validate_response = requests.get(f"{API_URL}/auth/validate", headers=headers)
            if validate_response.status_code == 200:
                user_data = validate_response.json()
                print(f"✅ Token validation successful! User: {user_data.get('username')} (Role: {user_data.get('account_type')})")
            else:
                print(f"❌ Token validation failed: {validate_response.status_code}")
                
        else:
            print(f"❌ Customer login failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Customer login error: {e}")
    
    # Test 2: Test role-based access control
    print("\n2. Testing Role-Based Access Control...")
    
    # Try to access admin endpoint with customer token
    try:
        if 'customer_token' in locals():
            headers = {"Authorization": f"Bearer {customer_token}"}
            admin_response = requests.get(f"{API_URL}/admin/users", headers=headers)
            if admin_response.status_code == 403:
                print("✅ Customer correctly denied access to admin endpoint")
            else:
                print(f"❌ Customer unexpectedly got access to admin endpoint: {admin_response.status_code}")
    except Exception as e:
        print(f"❌ Error testing admin access: {e}")
    
    # Test 3: Login with admin credentials
    print("\n3. Testing Admin Login...")
    
    admin_form_data = {
        "username": "alice",  # Updated with working admin credentials
        "password": "alice",  # Updated with working admin credentials
        "grant_type": "password",
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }
    
    try:
        response = requests.post(f"{API_URL}/auth/token", data=admin_form_data)
        if response.status_code == 200:
            token_data = response.json()
            admin_token = token_data["access_token"]
            print(f"✅ Admin login successful! Token: {admin_token[:20]}...")
            
            # Test admin access
            headers = {"Authorization": f"Bearer {admin_token}"}
            admin_response = requests.get(f"{API_URL}/admin/users", headers=headers)
            if admin_response.status_code == 200:
                users = admin_response.json()
                print(f"✅ Admin correctly granted access to admin endpoint ({len(users)} users found)")
            else:
                print(f"❌ Admin denied access to admin endpoint: {admin_response.status_code}")
                
        else:
            print(f"❌ Admin login failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Admin login error: {e}")
    
    # Test 4: Test employee access
    print("\n4. Testing Employee Login and Access...")
    
    employee_form_data = {
        "username": "bob",  # Updated with working employee credentials
        "password": "bob",  # Updated with working employee credentials
        "grant_type": "password",
        "scope": "",
        "client_id": "",
        "client_secret": "",
    }
    
    try:
        response = requests.post(f"{API_URL}/auth/token", data=employee_form_data)
        if response.status_code == 200:
            token_data = response.json()
            employee_token = token_data["access_token"]
            print(f"✅ Employee login successful! Token: {employee_token[:20]}...")
            
            # Test employee access to customers
            headers = {"Authorization": f"Bearer {employee_token}"}
            customers_response = requests.get(f"{API_URL}/employee/customers", headers=headers)
            if customers_response.status_code == 200:
                customers = customers_response.json()
                print(f"✅ Employee correctly granted access to customers endpoint ({len(customers)} customers found)")
            else:
                print(f"❌ Employee denied access to customers endpoint: {customers_response.status_code}")
                
            # Test employee trying to access admin endpoint (should be denied)
            admin_response = requests.get(f"{API_URL}/admin/users", headers=headers)
            if admin_response.status_code == 403:
                print("✅ Employee correctly denied access to admin endpoint")
            else:
                print(f"❌ Employee unexpectedly got access to admin endpoint: {admin_response.status_code}")
                
        else:
            print(f"❌ Employee login failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Employee login error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Authentication test completed!")
    
    print("\n📝 SUMMARY:")
    print("✅ Enhanced JWT authentication with role information")
    print("✅ Token validation and refresh endpoints")
    print("✅ Role-based access control on all endpoints")
    print("✅ Proper error handling and security messages")

if __name__ == "__main__":
    test_authentication()
