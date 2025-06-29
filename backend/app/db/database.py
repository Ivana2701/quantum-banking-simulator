#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Generator
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Check if we should use SQLite for development
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

if USE_SQLITE:
    # SQLite configuration for development/debugging
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quantum_banking_dev.db")
    # SQLite-specific engine configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Needed for SQLite
        pool_pre_ping=True
    )
else:
    # PostgreSQL configuration
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    # Construct the full database URL
    DATABASE_URL = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    
    # PostgreSQL engine configuration
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True
    )
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)
Base = declarative_base()

# Dependency for FastAPI

def get_db() -> Generator:
    """
    FastAPI dependency that yields a SQLAlchemy Session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
