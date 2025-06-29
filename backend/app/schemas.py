#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/schemas.py
from datetime import datetime
from typing import List, Optional
import enum, base64
from pydantic import BaseModel, EmailStr, constr, Field, validator

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
    email: Optional[EmailStr] = None
    account_type: AccountType
    status: str = "active"

# Schema for creating an account
class AccountCreate(AccountBase):
    password: constr(min_length=8)

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
    is_fraud: bool

class TransactionRead(TransactionBase):
    transaction_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TransactionBundle(BaseModel):
    all: Optional[List[TransactionRead]] = None
    sent: Optional[List[TransactionRead]] = None
    received: Optional[List[TransactionRead]] = None
