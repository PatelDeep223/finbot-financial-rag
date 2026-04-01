"""
Security module — JWT auth + bcrypt password hashing + rate limiting.

Auth flow:
  1. POST /api/v1/auth/signup  → create user, return JWT
  2. POST /api/v1/auth/login   → validate email+password, return JWT
  3. Authorization: Bearer <token>  → access protected endpoints
  4. Dev mode: if no users in DB → auth skipped entirely
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── PASSWORD HASHING ────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ─── JWT ─────────────────────────────────────────────────────────────────────

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── RATE LIMITER ────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ─── AUTH CHECK ──────────────────────────────────────────────────────────────

async def is_auth_enabled() -> bool:
    """Auth is disabled when no users exist in DB (dev/first-run mode)."""
    try:
        from app.db import service as db_svc
        count = await db_svc.user_count()
        return count > 0
    except Exception:
        return False


# ─── FASTAPI DEPENDENCY ──────────────────────────────────────────────────────

async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> dict:
    """
    Dependency for protected endpoints.
    - Extracts JWT from Authorization: Bearer <token>
    - Returns {"user_id": int, "sub": str}
    - Skips auth in dev mode (no users in DB)
    """
    # Dev mode — no users registered yet
    if not await is_auth_enabled():
        return {"user_id": None, "sub": "dev_mode", "auth": "disabled"}

    # No token provided
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please login to get a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Decode and validate JWT
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("user_id")
    sub = payload.get("sub", "unknown")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id",
        )

    return {"user_id": user_id, "sub": sub, "auth": "jwt"}
