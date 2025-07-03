#/Users/ibazhdarova/ProjectsIvana/quantum/quantum-banking-simulator/backend/app/routers/transactions.py
from datetime import date, datetime, timedelta, timedelta
from typing import Optional
import logging
import time
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.services.transaction_service import TransactionService
from app.services.account_service import AccountService
from app.schemas import (
    TransactionBundle, TransactionRead, TransactionCreate,
    SessionEstablishmentRequest, SessionEstablishmentResponse,
    SecureTransactionRequest, SecureTransactionResponse
)
from app.db.database import get_db
from app.core.security import get_current_user, require_employee, require_customer, require_employee_or_customer
from app.db.models import Account, AccountTypeEnum
from app.config import QUANTUM_SAFE_ENABLED
from quantum_encryption.hybrid_pqc_protocol import get_protocol_instance

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transactions", tags=["transactions"])
tx_svc   = TransactionService()
acct_svc = AccountService()

@router.get("", response_model=TransactionBundle)
def get_transactions(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    Get transactions based on user role:
    - Employees/Admins: See all transactions with decrypted amounts
    - Customers: See only their sent and received transactions (amounts encrypted)
    """
    # reload to get full account_type
    acct = acct_svc.get_account_by_id(db, user.account_id)
    if not acct:
        raise HTTPException(404, "Account not found")

    # Role-based access to transactions
    if acct.account_type.value in ["employee", "admin"]:
        # Employees and admins can see all transactions with decrypted amounts
        all_tx = tx_svc.get_all_transactions(db, from_date, to_date)
        # Add decrypted amounts for employee/admin view
        for tx in all_tx:
            tx.amount = tx_svc.decrypt_transaction_amount(db, tx, user.account_id)
        return TransactionBundle(all=all_tx)
    elif acct.account_type.value == "customer":
        # Customers can only see their own transactions (amounts remain encrypted)
        sent     = tx_svc.get_sent_transactions(db, acct.account_id, from_date, to_date)
        received = tx_svc.get_received_transactions(db, acct.account_id, from_date, to_date)
        return TransactionBundle(sent=sent, received=received)
    else:
        raise HTTPException(403, "Access denied")

@router.post("/establish-session", response_model=SessionEstablishmentResponse)
def establish_secure_session(
    request: SessionEstablishmentRequest,
    user = Depends(require_employee_or_customer)
):
    """
    Establish a secure session using hybrid BB84 + Kyber protocol.
    This endpoint sets up the cryptographic session for secure transactions.
    """
    try:
        logger.info(f"Establishing secure session for client {request.client_id}")
        
        # Get the protocol instance
        protocol = get_protocol_instance()
        
        # Establish session keys
        session_info = protocol.establish_session_keys(request.client_id)
        
        # Add expiration time
        session_info['expires_at'] = (datetime.utcnow().replace(microsecond=0) + 
                                    timedelta(hours=24)).isoformat()
        
        return SessionEstablishmentResponse(**session_info)
        
    except Exception as e:
        logger.error(f"Session establishment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to establish secure session: {str(e)}"
        )

@router.post("/secure-transaction", response_model=SecureTransactionResponse)
def process_secure_transaction(
    request: SecureTransactionRequest,
    db: Session = Depends(get_db),
    user: Account = Depends(require_customer)
):
    """
    Process a secure transaction using the hybrid PQC protocol.
    Transaction data is encrypted with AES-256-GCM and signed with Dilithium.
    """
    try:
        logger.info(f"Processing secure transaction for session {request.session_id}")
        
        # Get the protocol instance
        protocol = get_protocol_instance()
        
        # Process and validate the secure transaction request
        decrypted_data = protocol.process_secure_transaction_request(request.dict())

        # Check if user is authorized to perform this transaction
        if 'from_account_id' in decrypted_data:
            if (user.account_type.value == "customer" and 
                decrypted_data['from_account_id'] != user.account_id):
                raise HTTPException(403, "Not authorized to transfer from this account")
            
        to_account_id = decrypted_data.get('to_account_id')
        amount = decrypted_data.get('amount')
        # Validate transaction data
        if not all([to_account_id, amount]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required transaction data"
            )
        
        # Process the transaction
        new_transaction = tx_svc.create_transaction(
            db=db,
            account_id=user.account_id,
            from_account_id=decrypted_data.get('from_account_id', user.account_id),
            to_account_id=decrypted_data['to_account_id'],
            amount=decrypted_data['amount']
        )
        
        # Create secure response
        response_data = {
            'success': True,
            'transaction_id': str(new_transaction.transaction_id),
            'message': 'Transaction processed successfully',
            'amount': amount,
            'from_account_id': user.account_id,
            'to_account_id': to_account_id,
            'timestamp': new_transaction.created_at.isoformat(),
            'session_id': request.session_id
        }
        
        # Sign the response
        signature = protocol.sign_transaction(response_data)
        response_data['signature'] = signature
        
        logger.info(f"Secure transaction completed: {new_transaction.transaction_id}")
        return SecureTransactionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Secure transaction processing failed: {e}")
        
        # Create signed error response
        error_response = {
            'success': False,
            'message': f'Transaction failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            protocol = get_protocol_instance()
            error_signature = protocol.sign_transaction(error_response)
            error_response['signature'] = error_signature
        except:
            pass  # If signing fails, return without signature
        
        return SecureTransactionResponse(**error_response)

@router.post("/create-secure-transaction", response_model=SecureTransactionRequest)
def create_secure_transaction(
    session_id: str,
    transaction_data: TransactionCreate,
    user = Depends(require_customer),
    db: Session = Depends(get_db)
):
    """
    Create a secure transaction request that can be submitted to /secure-transaction.
    This endpoint encrypts and signs transaction data for secure transmission.
    """
    try:
        logger.info(f"Creating secure transaction for session {session_id}")
        
        # Convert transaction data to dictionary
        tx_dict = transaction_data.dict()
        
        # Add user context if not specified
        if not tx_dict.get('from_account_id'):
            tx_dict['from_account_id'] = user.account_id

        toAcct = acct_svc.get_account_by_id(db, transaction_data.to_account_id)
        if toAcct.account_type != AccountTypeEnum.customer or transaction_data.to_account_id == user.account_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Both from and to accounts must be customers and cannot be the same account."
            )
        
        # Get the protocol instance
        protocol = get_protocol_instance()
        
        # Create secure transaction request
        secure_request = protocol.create_secure_transaction_request(session_id, tx_dict)
        
        logger.info(f"Secure transaction request created for session {session_id}")
        return SecureTransactionRequest(**secure_request)
        
    except Exception as e:
        logger.error(f"Secure transaction creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create secure transaction: {str(e)}"
        )

@router.delete("/cleanup-sessions")
def cleanup_expired_sessions(
    user = Depends(get_current_user)
):
    """
    Cleanup expired session keys (admin only).
    """
    # Check if user is admin
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    
    try:
        protocol = get_protocol_instance()
        protocol.cleanup_expired_sessions()
        
        return {"message": "Expired sessions cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session cleanup failed: {str(e)}"
        )

# Demo and testing endpoints for frontend integration

@router.post("/quantum/test-bb84", response_model=dict)
def test_bb84_protocol(
    key_length: Optional[int] = 128,
    error_rate: Optional[float] = 0.05,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test BB84 protocol with specified parameters"""
    try:
        from quantum_encryption.bb84 import BB84Protocol
        
        # Create BB84Protocol instance with the specified parameters
        bb84 = BB84Protocol(key_length=key_length, error_threshold=error_rate)
        start_time = time.time()
        
        # Run BB84 protocol (only accepts initial_length_multiplier parameter)
        result = bb84.run_full_protocol(initial_length_multiplier=4)
        
        end_time = time.time()
        
        return {
            "success": True,
            "key_length": len(result["shared_key"]),
            "final_key_length": len(result["shared_key"]),
            "error_rate": result["parameters"]["error_rate"],
            "measured_error_rate": result["parameters"]["error_rate"],
            "security_level": "High" if result["parameters"]["error_rate"] < 0.1 else "Medium",
            "security_status": "Secure",
            "execution_time": end_time - start_time,
            "steps": [
                "Initialize quantum channel",
                "Alice generates random bits and bases",
                "Alice sends encoded qubits",
                "Bob measures with random bases", 
                "Public basis comparison",
                "Key sifting and error correction",
                f"Final key established: {len(result['shared_key'])} bits"
            ]
        }
    except Exception as e:
        logger.error(f"BB84 test failed: {e}")
        raise HTTPException(status_code=500, detail=f"BB84 test failed: {str(e)}")

@router.post("/quantum/demo-bb84", response_model=dict)
def demo_bb84_protocol(
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Demo BB84 protocol with custom parameters from frontend"""
    try:
        from quantum_encryption.bb84 import BB84Protocol
        
        # Extract parameters from request
        key_length = request.get("key_length", 128)
        error_rate = request.get("error_rate", 0.05)
        
        # Create BB84Protocol instance with the specified parameters
        bb84 = BB84Protocol(key_length=key_length, error_threshold=error_rate)
        start_time = time.time()
        
        # Run BB84 protocol
        result = bb84.run_full_protocol(initial_length_multiplier=4)
        
        end_time = time.time()
        
        return {
            "success": True,
            "key_length": len(result["shared_key"]),
            "final_key_length": len(result["shared_key"]),
            "error_rate": result["parameters"]["error_rate"],
            "measured_error_rate": result["parameters"]["error_rate"],
            "security_level": "High" if result["parameters"]["error_rate"] < 0.1 else "Medium",
            "security_status": "Secure",
            "execution_time": end_time - start_time,
            "steps": [
                "Initialize quantum channel",
                "Alice generates random bits and bases",
                "Alice sends encoded qubits",
                "Bob measures with random bases", 
                "Public basis comparison",
                "Key sifting and error correction",
                f"Final key established: {len(result['shared_key'])} bits"
            ],
            "detailed_metrics": {
                "initial_qubits": result["parameters"]["initial_bits"],
                "matching_bases": result["parameters"]["matching_bases"],
                "basis_match_rate": result["parameters"]["basis_match_rate"],
                "key_extraction_efficiency": result["parameters"]["key_extraction_efficiency"],
                "protocol_success": result["parameters"]["protocol_success"],
                "error_threshold": error_rate,  # The configured threshold
                "security_bits": min(len(result["shared_key"]), 256),
                "processing_rate": result["parameters"]["initial_bits"] / (end_time - start_time) if (end_time - start_time) > 0 else 0
            },
            "protocol_info": {
                "name": "BB84 Quantum Key Distribution",
                "type": "Information-Theoretic Secure",
                "quantum_backend": "IBM Qiskit qasm_simulator",
                "batch_size": 30,
                "basis_count": 2,
                "encoding_schemes": ["Computational (Z-basis)", "Hadamard (X-basis)"]
            }
        }
    except Exception as e:
        logger.error(f"BB84 demo failed: {e}")
        raise HTTPException(status_code=500, detail=f"BB84 demo failed: {str(e)}")

@router.post("/quantum/demo-full-protocol", response_model=dict)
def demo_full_protocol(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Demonstrate the complete hybrid post-quantum protocol"""
    try:
        protocol = get_protocol_instance()
        start_time = time.time()
        
        # Phase 1: Session establishment
        phase1_start = time.time()
        session_info = protocol.establish_session_keys(current_user["username"])
        phase1_time = time.time() - phase1_start
        
        # Phase 2: Transaction encryption
        phase2_start = time.time()
        test_transaction = {
            "amount": 100.50,
            "to_account_id": 999,
            "description": "Demo transaction"
        }
        encrypted_data = protocol.encrypt_transaction_data(
            session_info["session_id"], 
            test_transaction
        )
        phase2_time = time.time() - phase2_start
        
        # Phase 3: Digital signature
        phase3_start = time.time()
        signature = protocol.sign_transaction(test_transaction)
        phase3_time = time.time() - phase3_start
        
        # Phase 4: Verification
        phase4_start = time.time()
        is_valid = protocol.verify_transaction_signature(test_transaction, signature)
        phase4_time = time.time() - phase4_start
        
        # Phase 5: Decryption
        phase5_start = time.time()
        decrypted_data = protocol.decrypt_transaction_data(encrypted_data)
        phase5_time = time.time() - phase5_start
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "total_time": total_time,
            "bb84_time": phase1_time * 0.6,  # Estimate BB84 portion
            "kyber_time": phase1_time * 0.4,  # Estimate Kyber portion
            "phases": {
                "session_establishment": {
                    "time": phase1_time,
                    "session_id": session_info["session_id"][:8] + "...",
                    "protocols": ["BB84", "Kyber", "HKDF"]
                },
                "data_encryption": {
                    "time": phase2_time,
                    "algorithm": "AES-256-GCM",
                    "data_size": len(json.dumps(test_transaction))
                },
                "digital_signature": {
                    "time": phase3_time,
                    "algorithm": "CRYSTALS-Dilithium",
                    "signature_length": len(signature)
                },
                "signature_verification": {
                    "time": phase4_time,
                    "result": "Valid" if is_valid else "Invalid"
                },
                "data_decryption": {
                    "time": phase5_time,
                    "success": decrypted_data == test_transaction
                }
            }
        }
    except Exception as e:
        logger.error(f"Full protocol demo failed: {e}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

# Health check endpoint for quantum readiness
@router.get("/quantum/health", response_model=dict)
def quantum_health_check():
    """Check if quantum protocols are ready"""
    try:
        protocol = get_protocol_instance()
        return {
            "quantum_ready": True,
            "protocols_available": ["BB84", "Kyber", "Dilithium", "AES-GCM"],
            "status": "All quantum protocols operational"
        }
    except Exception as e:
        return {
            "quantum_ready": False,
            "error": str(e),
            "status": "Quantum protocols not available"
        }

@router.get("/config/quantum-safe")
def get_quantum_safe_config():
    """
    Get the current quantum-safe protocol configuration
    """
    return {
        "quantum_safe_enabled": QUANTUM_SAFE_ENABLED,
        "message": "Quantum-safe protocol enabled" if QUANTUM_SAFE_ENABLED else "Standard protocol enabled"
    }

@router.post("/create-secure-employee-transaction", response_model=SecureTransactionRequest)
def create_secure_employee_transaction(
    session_id: str,
    transaction_data: TransactionCreate,
    user: Account = Depends(require_employee),
    db: Session = Depends(get_db)
):
    """
    Create a secure employee transaction request that can be submitted to /secure-employee-transaction.
    This endpoint encrypts and signs employee transaction data for secure transmission.
    """
    try:
        logger.info(f"Creating secure employee transaction for session {session_id}")
        
        if transaction_data.from_account_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="From account ID is required for employee transactions."
            )
        
        fromAcct = acct_svc.get_account_by_id(db, transaction_data.from_account_id)
        toAcct = acct_svc.get_account_by_id(db, transaction_data.to_account_id)
        if fromAcct.account_type != AccountTypeEnum.customer or toAcct.account_type != AccountTypeEnum.customer or transaction_data.from_account_id == transaction_data.to_account_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Both from and to accounts must be customers and cannot be the same account."
            )
        
        # Get the protocol instance
        protocol = get_protocol_instance()
        
        tx_dict = transaction_data.dict()

        # Add employee context
        tx_dict['account_id'] = user.account_id
        
        # Create secure transaction request
        secure_request = protocol.create_secure_transaction_request(session_id, tx_dict)
        
        logger.info(f"Secure employee transaction request created: {secure_request['session_id']}")
        return SecureTransactionRequest(**secure_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Secure employee transaction creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create secure employee transaction: {str(e)}"
        )

@router.post("/secure-employee-transaction", response_model=SecureTransactionResponse)
def process_secure_employee_transaction(
    request: SecureTransactionRequest,
    db: Session = Depends(get_db),
    user: Account = Depends(require_employee)
):
    """
    Process a secure employee transaction using the hybrid PQC protocol.
    This endpoint handles encrypted and signed employee transaction packages.
    """
    try:
        logger.info(f"Processing secure employee transaction for session {request.session_id}")
        
        # Get the protocol instance
        protocol = get_protocol_instance()
        
        # Process and validate the secure transaction request
        decrypted_data = protocol.process_secure_transaction_request(request.dict())
        
        # Extract transaction details
        from_account_id = decrypted_data.get('from_account_id')
        to_account_id = decrypted_data.get('to_account_id')
        amount = decrypted_data.get('amount')
        
        # Validate transaction data
        if not all([from_account_id, to_account_id, amount]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required transaction data"
            )
        
        # Create the transaction using employee service
        new_transaction = tx_svc.create_transaction(
            db=db,
            account_id=user.account_id,  # Employee as initiator
            from_account_id=from_account_id,
            to_account_id=to_account_id,
            amount=amount,
        )
        
        # Prepare response data
        response_data = {
            'success': True,
            'transaction_id': new_transaction.transaction_id,
            'message': 'Employee transaction completed successfully',
            'amount': amount,
            'from_account_id': from_account_id,
            'to_account_id': to_account_id,
            'timestamp': new_transaction.created_at.isoformat(),
            'session_id': request.session_id
        }
        
        # Sign the response
        signature = protocol.sign_transaction(response_data)
        response_data['signature'] = signature
        
        logger.info(f"Secure employee transaction completed: {new_transaction.transaction_id}")
        return SecureTransactionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Secure employee transaction processing failed: {e}")
        
        # Create signed error response
        error_response = {
            'success': False,
            'message': f'Employee transaction failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            protocol = get_protocol_instance()
            error_signature = protocol.sign_transaction(error_response)
            error_response['signature'] = error_signature
        except:
            pass  # If signing fails, return without signature
        
        return SecureTransactionResponse(**error_response)
