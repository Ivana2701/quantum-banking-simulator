from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi_csrf_protect import CsrfProtect
from backend.authentication.auth_service import authenticate_user

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str
    role:     str

class LoginResponse(BaseModel):
    account_id: int
    role:       str

@router.get("/csrf-token")
def get_csrf(csrf_protect: CsrfProtect = Depends()):
    """
    Generates a new CSRF token and sets the signed token
    in a secure cookie on the response.
    """
    raw_token = csrf_protect.generate_csrf()
    response = JSONResponse({"csrf_token": raw_token})
    csrf_protect.set_csrf_cookie(response)
    return response

@router.post(
    "/login",
    response_model=LoginResponse,
    dependencies=[Depends(CsrfProtect.validate_csrf_in_cookies)]
)
def login(req: LoginRequest):
    account_id = authenticate_user(req.username, req.password, req.role)
    if account_id is None:
        raise HTTPException(401, "Invalid credentials")
    return {"account_id": account_id, "role": req.role.capitalize()}
