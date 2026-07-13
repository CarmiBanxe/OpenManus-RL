"""JWT-аутентификация для OpenManus (реальная, на PyJWT + bcrypt)."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

logger = logging.getLogger(__name__)


class JWTAuth:
    def __init__(self, secret_key: str, algorithm: str = "HS256", token_expire_hours: int = 24) -> None:
        if not secret_key:
            raise ValueError("secret_key is required (no insecure default)")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_hours = token_expire_hours

    def generate_token(self, user_id: str, extra: Optional[Dict[str, Any]] = None) -> str:
        now = datetime.now(timezone.utc)
        payload: Dict[str, Any] = {"sub": user_id, "iat": now, "exp": now + timedelta(hours=self.token_expire_hours)}
        if extra:
            payload.update(extra)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            logger.warning("JWT expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.warning("Invalid JWT: %s", exc)
            return None

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except (ValueError, TypeError):
            return False
