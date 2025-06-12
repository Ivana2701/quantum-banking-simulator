import bcrypt

def hash_password(plain_password):
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()

print("Customer:", hash_password("customer"))
print("Employee:", hash_password("employee"))
print("Admin:", hash_password("admin"))
