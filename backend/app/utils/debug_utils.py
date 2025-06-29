"""
Debug utilities for the Quantum Banking Simulator
"""
import json
import traceback
from typing import Any, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def debug_print(message: str, data: Any = None, level: str = "DEBUG"):
    """Enhanced debug printing with timestamp and formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n[{timestamp}] {level}: {message}")
    
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Data: {data}")
    print("-" * 50)

def log_function_entry(func_name: str, **kwargs):
    """Log function entry with parameters"""
    logger.debug(f"Entering {func_name}")
    for key, value in kwargs.items():
        if key == 'password':
            logger.debug(f"  {key}: [REDACTED]")
        else:
            logger.debug(f"  {key}: {value}")

def log_function_exit(func_name: str, result: Any = None):
    """Log function exit with result"""
    logger.debug(f"Exiting {func_name}")
    if result is not None:
        if hasattr(result, '__dict__'):
            logger.debug(f"  Result type: {type(result).__name__}")
        else:
            logger.debug(f"  Result: {result}")

def log_exception(func_name: str, exception: Exception):
    """Log exception with full traceback"""
    logger.error(f"Exception in {func_name}: {str(exception)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

def debug_database_query(query_description: str, query_result: Any):
    """Debug database queries"""
    logger.debug(f"Database Query: {query_description}")
    if hasattr(query_result, '__len__'):
        logger.debug(f"Result count: {len(query_result)}")
    else:
        logger.debug(f"Result: {query_result}")

def debug_api_request(endpoint: str, method: str, data: Dict = None):
    """Debug API requests"""
    logger.info(f"API Request: {method} {endpoint}")
    if data:
        safe_data = {k: "[REDACTED]" if "password" in k.lower() else v 
                    for k, v in data.items()}
        logger.debug(f"Request data: {safe_data}")

def debug_quantum_operation(operation: str, parameters: Dict = None):
    """Debug quantum operations"""
    logger.info(f"Quantum Operation: {operation}")
    if parameters:
        logger.debug(f"Parameters: {parameters}")

class DebugMiddleware:
    """Middleware for debugging FastAPI requests"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            logger.info(f"HTTP Request: {scope['method']} {scope['path']}")
        
        await self.app(scope, receive, send)
