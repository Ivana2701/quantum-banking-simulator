#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/schemas.py
from datetime import datetime
from typing import List, Optional
import enum, base64
from pydantic import BaseModel, constr, Field, validator

# JWT token schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Account types
class AccountType(str, enum.Enum):
    customer = "customer"
    employee = "employee"
    admin    = "admin"

# Base schema for accounts
class AccountBase(BaseModel):
    username: constr(min_length=3)
    full_name: str
    account_type: AccountType
    status: str = "active"

# Schema for creating an account
class AccountCreate(AccountBase):
    password: constr(min_length=8)
    # Optional contact information
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    phone_number: Optional[str] = None
    phone_type: Optional[str] = "mobile"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_description: Optional[str] = None
    ip_address: Optional[str] = None
    device_name: Optional[str] = None
    device_fingerprint: Optional[str] = None
    role_id: Optional[int] = None

# Schema for reading an account
class AccountRead(AccountBase):
    account_id: int
    encrypted_balance: str = Field(..., description="Base64-encoded ciphertext")
    role_id: int
    created_at: datetime

    @validator("encrypted_balance", pre=True)
    def encode_balance(cls, v):
        if isinstance(v, (bytes, bytearray)):
            return base64.b64encode(v).decode()
        return v

    class Config:
        from_attributes = True

# Address schemas
class AddressBase(BaseModel):
    street: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    postal_code: Optional[str]

class AddressCreate(AddressBase):
    pass

class AddressRead(AddressBase):
    address_id: int

    class Config:
        from_attributes = True

# PhoneNumber schemas
class PhoneNumberBase(BaseModel):
    phone_number: constr(min_length=5)
    phone_type: str

class PhoneNumberCreate(PhoneNumberBase):
    pass

class PhoneNumberRead(PhoneNumberBase):
    phone_id: int

    class Config:
        from_attributes = True

# GeoLocation schemas
class GeoLocationBase(BaseModel):
    latitude: float
    longitude: float
    description: Optional[str] = None

class GeoLocationCreate(GeoLocationBase):
    pass

class GeoLocationRead(GeoLocationBase):
    geo_id: int

    class Config:
        from_attributes = True

# IPAddress schemas
class IPAddressBase(BaseModel):
    ip_address: str

class IPAddressCreate(IPAddressBase):
    pass

class IPAddressRead(IPAddressBase):
    ip_id: int
    first_seen: datetime
    last_seen: datetime

    class Config:
        from_attributes = True

# Device schemas
class DeviceBase(BaseModel):
    device_name: str
    fingerprint: str

class DeviceCreate(DeviceBase):
    pass

class DeviceRead(DeviceBase):
    device_id: int
    last_login: datetime

    class Config:
        from_attributes = True

# Transaction schemas
class TransactionBase(BaseModel):
    from_account_id: int
    to_account_id: int
    is_fraud: bool = False

class TransactionCreate(BaseModel):
    from_account_id: int = None
    to_account_id: int
    amount: float = Field(..., gt=0, description="Amount to transfer")

class TransactionRead(TransactionBase):
    transaction_id: int
    created_at: datetime
    amount: Optional[float] = Field(None, description="Decrypted amount (for employees/admins only)")

    class Config:
        from_attributes = True

class TransactionBundle(BaseModel):
    all: Optional[List[TransactionRead]] = None
    sent: Optional[List[TransactionRead]] = None
    received: Optional[List[TransactionRead]] = None

# Admin schemas
class RoleUpdateRequest(BaseModel):
    new_role: str = Field(..., description="New role for the user (admin, employee, customer)")

class RoleUpdateResponse(BaseModel):
    message: str
    user_id: int
    new_role: str

# Balance schemas
class BalanceUpdateRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Amount to add to the customer's balance")

# Secure Transaction Protocol Schemas
class SessionEstablishmentRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    protocol_version: str = Field(default="1.0", description="Protocol version")

class SessionEstablishmentResponse(BaseModel):
    session_id: str
    client_id: str
    timestamp: str
    kyber_public_key: str
    kyber_encapsulated_secret: str
    bank_public_key: str
    bb84_parameters: dict
    expires_at: Optional[str] = None

class SecureTransactionRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    encrypted_transaction: dict = Field(..., description="Encrypted transaction data")
    signature: str = Field(..., description="Digital signature")
    bank_public_key: str = Field(..., description="Bank's public key for verification")
    protocol_version: str = Field(default="1.0", description="Protocol version")
    security_level: str = Field(default="hybrid-pqc", description="Security level indicator")

class SecureTransactionResponse(BaseModel):
    success: bool
    transaction_id: Optional[str] = None
    message: str
    signature: Optional[str] = None
    timestamp: str

class EncryptedTransactionData(BaseModel):
    session_id: str
    nonce: str
    ciphertext: str
    timestamp: str
