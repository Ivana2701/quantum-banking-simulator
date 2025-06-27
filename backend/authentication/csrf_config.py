# backend/authentication/csrf_config.py

import os
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi_csrf_protect import CsrfProtect, CsrfProtectException

class CsrfSettings(BaseModel):
    # this key must be at least 32 characters in production
    secret_key: str = os.getenv(
        "CSRF_SECRET_KEY",
        "a-really-long-random-string-you-should-change"
    )

@CsrfProtect.load_config
def get_csrf_config() -> CsrfSettings:
    return CsrfSettings()

def register_csrf(app: FastAPI):
    """
    Call this once (in backend/app.py) to wire in the global CSRF exception handler.
    """
    @app.exception_handler(CsrfProtectException)
    def csrf_exception_handler(request, exc: CsrfProtectException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message}
        )
