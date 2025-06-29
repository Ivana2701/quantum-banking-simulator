from sqlalchemy import (
    Column, Integer, String, LargeBinary, DateTime,
    Enum, ForeignKey, Boolean, Numeric, Table
)
from sqlalchemy.orm import relationship
import enum
from .database import Base

# Association tables
account_addresses = Table(
    'account_addresses', Base.metadata,
    Column('account_id', Integer, ForeignKey('accounts.account_id', ondelete='CASCADE'), primary_key=True),
    Column('address_id', Integer, ForeignKey('addresses.address_id', ondelete='CASCADE'), primary_key=True)
)

account_phones = Table(
    'account_phones', Base.metadata,
    Column('account_id', Integer, ForeignKey('accounts.account_id', ondelete='CASCADE'), primary_key=True),
    Column('phone_id', Integer, ForeignKey('phone_numbers.phone_id', ondelete='CASCADE'), primary_key=True)
)

account_geos = Table(
    'account_geos', Base.metadata,
    Column('account_id', Integer, ForeignKey('accounts.account_id', ondelete='CASCADE'), primary_key=True),
    Column('geo_id', Integer, ForeignKey('geolocations.geo_id', ondelete='CASCADE'), primary_key=True)
)

account_ips = Table(
    'account_ips', Base.metadata,
    Column('account_id', Integer, ForeignKey('accounts.account_id', ondelete='CASCADE'), primary_key=True),
    Column('ip_id', Integer, ForeignKey('ip_addresses.ip_id', ondelete='CASCADE'), primary_key=True)
)

account_devices = Table(
    'account_devices', Base.metadata,
    Column('account_id', Integer, ForeignKey('accounts.account_id', ondelete='CASCADE'), primary_key=True),
    Column('device_id', Integer, ForeignKey('devices.device_id', ondelete='CASCADE'), primary_key=True)
)

# Enum for account types
class AccountTypeEnum(enum.Enum):
    customer = "customer"
    employee = "employee"
    admin    = "admin"

# Role model
class Role(Base):
    __tablename__ = "roles"
    role_id   = Column(Integer, primary_key=True, index=True)
    role_name = Column(String, unique=True, nullable=False)

# Account model with rich relationships
class Account(Base):
    __tablename__ = "accounts"
    account_id        = Column(Integer, primary_key=True, index=True)
    username          = Column(String, unique=True, nullable=False, index=True)
    full_name         = Column(String, nullable=False)
    password_hash     = Column(String, nullable=False)
    encrypted_balance = Column(LargeBinary, nullable=False)
    created_at        = Column(DateTime, server_default="now()")
    account_type      = Column(Enum(AccountTypeEnum), nullable=False)
    status            = Column(String, default="active")
    role_id           = Column(Integer, ForeignKey("roles.role_id"), nullable=False)

    addresses = relationship(
        "Address",
        secondary=account_addresses,
        back_populates="accounts",
        lazy="selectin"
    )
    phones = relationship(
        "PhoneNumber",
        secondary=account_phones,
        back_populates="accounts",
        lazy="selectin"
    )
    geos = relationship(
        "GeoLocation",
        secondary=account_geos,
        back_populates="accounts",
        lazy="selectin"
    )
    ips = relationship(
        "IPAddress",
        secondary=account_ips,
        back_populates="accounts",
        lazy="selectin"
    )
    devices = relationship(
        "Device",
        secondary=account_devices,
        back_populates="accounts",
        lazy="selectin"
    )

# Contact detail models
class Address(Base):
    __tablename__ = "addresses"
    address_id  = Column(Integer, primary_key=True, index=True)
    street      = Column(String)
    city        = Column(String)
    state       = Column(String)
    country     = Column(String)
    postal_code = Column(String)
    accounts    = relationship(
        "Account",
        secondary=account_addresses,
        back_populates="addresses"
    )

class PhoneNumber(Base):
    __tablename__ = "phone_numbers"
    phone_id     = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, nullable=False)
    phone_type   = Column(String, nullable=False)
    accounts     = relationship(
        "Account",
        secondary=account_phones,
        back_populates="phones"
    )

class GeoLocation(Base):
    __tablename__ = "geolocations"
    geo_id      = Column(Integer, primary_key=True, index=True)
    latitude    = Column(Numeric, nullable=False)
    longitude   = Column(Numeric, nullable=False)
    description = Column(String)
    accounts    = relationship(
        "Account",
        secondary=account_geos,
        back_populates="geos"
    )

class IPAddress(Base):
    __tablename__ = "ip_addresses"
    ip_id       = Column(Integer, primary_key=True, index=True)
    ip_address  = Column(String, nullable=False)
    first_seen  = Column(DateTime, server_default="now()")
    last_seen   = Column(DateTime, server_default="now()")
    accounts    = relationship(
        "Account",
        secondary=account_ips,
        back_populates="ips"
    )

class Device(Base):
    __tablename__ = "devices"
    device_id    = Column(Integer, primary_key=True, index=True)
    device_name  = Column(String, nullable=False)
    fingerprint  = Column(String, nullable=False)
    last_login   = Column(DateTime, server_default="now()")
    accounts     = relationship(
        "Account",
        secondary=account_devices,
        back_populates="devices"
    )

# Core transaction model
class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id   = Column(Integer, primary_key=True, index=True)
    from_account_id  = Column(Integer, ForeignKey("accounts.account_id"))
    to_account_id    = Column(Integer, ForeignKey("accounts.account_id"))
    encrypted_amount = Column(LargeBinary, nullable=False)
    is_fraud         = Column(Boolean, default=False)
    created_at       = Column(DateTime, server_default="now()")
