"""Login endpoints for the single configured admin user."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.auth import check_login_allowed, create_token, require_auth, verify_credentials
from app.core.logging import logger

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_at: int


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, body: LoginRequest) -> LoginResponse:
    check_login_allowed(request)
    if not verify_credentials(request, body.username, body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token, expires_at = create_token(body.username)
    logger.info(f"User '{body.username}' logged in")
    return LoginResponse(access_token=token, username=body.username, expires_at=expires_at)


@router.get("/me")
async def me(username: str = Depends(require_auth)) -> dict:
    return {"username": username}


@router.post("/logout")
async def logout(username: str = Depends(require_auth)) -> dict:
    # Tokens are stateless; the client discards its token. Logged for auditing.
    logger.info(f"User '{username}' logged out")
    return {"message": "Logged out"}
