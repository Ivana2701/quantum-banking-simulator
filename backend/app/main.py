from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import Base, engine
from app.routers import auth, accounts, customer, employee, transactions, admin
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config after loading environment
from app.config import QUANTUM_SAFE_ENABLED

Base.metadata.create_all(bind=engine)
app = FastAPI(title="QBank API", debug=True)

logger.info("Starting Quantum Banking API server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health_check():
    """Basic health check endpoint"""
    try:
        # Try to get protocol instance to check quantum readiness
        from quantum_encryption.hybrid_pqc_protocol import get_protocol_instance
        protocol = get_protocol_instance()
        quantum_ready = True
    except Exception:
        quantum_ready = False
    
    return {
        "status": "healthy",
        "quantum_ready": quantum_ready,
        "service": "QBank API"
    }

# Configuration endpoint
@app.get("/config/quantum-safe")
def get_quantum_safe_config():
    """
    Get the current quantum-safe protocol configuration
    """
    return {
        "quantum_safe_enabled": QUANTUM_SAFE_ENABLED,
        "message": "Quantum-safe protocol enabled" if QUANTUM_SAFE_ENABLED else "Standard protocol enabled"
    }

app.include_router(auth)
app.include_router(accounts)
app.include_router(customer)
app.include_router(employee)
app.include_router(transactions)
app.include_router(admin)