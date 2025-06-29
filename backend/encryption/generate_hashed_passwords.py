import bcrypt

def hash_password(plain_password):
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()

print("Customer:", hash_password("customer"))
print("Employee:", hash_password("employee"))
print("Admin:", hash_password("admin"))
print("Jave Smith", hash_password("jsmith"))

print("Jave Smith", hash_password("60000"))
print("Alice", hash_password("alice"))
print("Bob", hash_password("bob"))
print("Carol", hash_password("carol"))



#run -> python generate_hashed_passwords.py
