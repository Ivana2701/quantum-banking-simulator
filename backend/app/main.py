from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import Base, engine
from app.routers import auth, accounts, customer, employee, transactions, admin
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

app.include_router(auth)
app.include_router(accounts)
app.include_router(customer)
app.include_router(employee)
app.include_router(transactions)
app.include_router(admin)