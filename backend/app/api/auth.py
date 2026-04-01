"""Auth endpoints — signup, login, me."""

from fastapi import APIRouter, Depends, HTTPException
from app.core.security import (
    require_auth, create_access_token,
    hash_password, verify_password,
)
from app.models.schemas import (
    SignupRequest, LoginRequest, TokenResponse, UserResponse,
)

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/signup", response_model=TokenResponse)
async def signup(body: SignupRequest):
    """Register a new user. Returns JWT token + user info."""
    from app.db import service as db_svc

    existing = await db_svc.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(body.password)
    user = await db_svc.create_user(
        username=body.username,
        email=body.email,
        hashed_password=hashed,
    )

    if not user:
        raise HTTPException(status_code=500, detail="Failed to create user")

    token = create_access_token(data={"sub": user.username, "user_id": user.id})
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Login with email + password. Returns JWT token + user info."""
    from app.db import service as db_svc

    user = await db_svc.get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    token = create_access_token(data={"sub": user.username, "user_id": user.id})
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(auth: dict = Depends(require_auth)):
    """Get current user info from JWT token."""
    if auth.get("auth") == "disabled":
        return UserResponse(id=0, username="dev_user", email="dev@localhost", is_active=True)

    from app.db import service as db_svc
    user = await db_svc.get_user_by_id(auth["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse.model_validate(user)
