"""FastAPI-зависимости аутентификации. Только required-auth (без public-обхода)."""
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_auth import JWTAuth

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=True)
_jwt_auth: Optional[JWTAuth] = None


def init_jwt_auth(secret_key: str, algorithm: str = "HS256", token_expire_hours: int = 24) -> JWTAuth:
    global _jwt_auth
    _jwt_auth = JWTAuth(secret_key, algorithm, token_expire_hours)
    logger.info("JWT auth initialised")
    return _jwt_auth


def get_jwt_auth() -> JWTAuth:
    if _jwt_auth is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth not configured")
    return _jwt_auth


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    auth = get_jwt_auth()
    payload = auth.verify_token(credentials.credentials)
    if payload is None or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]
