"""
Rate limiter middleware for the Legion FastAPI server.

Enforces per-client and global request-rate limits to prevent runaway LLM cost.
Reads config from RATE_LIMIT_* environment variables (or budget_config.yaml defaults).

Usage (agent_server.py):
    from openmanus_rl.middleware.rate_limiter import build_rate_limit_middleware
    app.add_middleware(build_rate_limit_middleware())
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — read from env (overridable in tests)
# ---------------------------------------------------------------------------

def _int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _get_config() -> "RateLimitConfig":
    return RateLimitConfig(
        requests_per_minute=_int_env("RATE_LIMIT_RPM", 60),
        requests_per_hour=_int_env("RATE_LIMIT_RPH", 600),
        global_requests_per_minute=_int_env("RATE_LIMIT_GLOBAL_RPM", 300),
        token_budget_per_session=_int_env("RATE_LIMIT_TOKEN_BUDGET", 100_000),
    )


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 600
    global_requests_per_minute: int = 300
    # Token budget is advisory — enforcement requires LLM counting integration
    token_budget_per_session: int = 100_000


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Core limiter (in-process, sliding window — not distributed)
# ---------------------------------------------------------------------------

@dataclass
class _Window:
    timestamps: list[float] = field(default_factory=list)

    def count_since(self, cutoff: float) -> int:
        self.timestamps = [t for t in self.timestamps if t >= cutoff]
        return len(self.timestamps)

    def record(self) -> None:
        self.timestamps.append(time.monotonic())


class RateLimiter:
    """
    Sliding-window request-rate limiter (in-process).

    Not distributed: if multiple server processes are used, each maintains
    its own counter. For distributed rate limiting use Redis (future sprint).
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self._cfg = config or _get_config()
        self._per_client: dict[str, _Window] = defaultdict(_Window)
        self._global = _Window()

    def check_and_record(self, client_id: str) -> None:
        """
        Raise RateLimitExceeded if the client or global limit is exceeded;
        otherwise record the request.
        """
        now = time.monotonic()
        minute_ago = now - 60.0
        hour_ago = now - 3600.0

        # Global gate
        global_rpm = self._global.count_since(minute_ago)
        if global_rpm >= self._cfg.global_requests_per_minute:
            logger.warning("Global rate limit hit: %d rpm", global_rpm)
            raise RateLimitExceeded(
                f"Global rate limit exceeded ({global_rpm}/{self._cfg.global_requests_per_minute} rpm). "
                "Retry after 60 seconds."
            )

        # Per-client gate
        window = self._per_client[client_id]
        client_rpm = window.count_since(minute_ago)
        if client_rpm >= self._cfg.requests_per_minute:
            logger.warning("Client %s rate limit hit: %d rpm", client_id, client_rpm)
            raise RateLimitExceeded(
                f"Rate limit exceeded for client '{client_id}': "
                f"{client_rpm}/{self._cfg.requests_per_minute} rpm. Retry after 60 seconds."
            )

        client_rph = window.count_since(hour_ago)
        if client_rph >= self._cfg.requests_per_hour:
            logger.warning("Client %s hourly limit hit: %d rph", client_id, client_rph)
            raise RateLimitExceeded(
                f"Hourly rate limit exceeded for client '{client_id}': "
                f"{client_rph}/{self._cfg.requests_per_hour} rph. Retry after 3600 seconds."
            )

        self._global.record()
        window.record()

    def get_client_id(self, request: Request) -> str:
        """Extract a client identifier from the request (API key or IP)."""
        api_key = request.headers.get("x-api-key") or request.headers.get("authorization", "")
        if api_key:
            # Use first 16 chars as opaque identifier (never log full key)
            return f"key:{api_key[:16]}"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client = request.client
        if client:
            return f"ip:{client.host}"
        return "ip:unknown"


# ---------------------------------------------------------------------------
# Starlette middleware wrapper
# ---------------------------------------------------------------------------

_LIMITER_EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/metrics", "/readyz", "/livez"})


class _RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        if request.url.path in _LIMITER_EXEMPT_PATHS:
            return await call_next(request)

        client_id = self._limiter.get_client_id(request)
        try:
            self._limiter.check_and_record(client_id)
        except RateLimitExceeded as exc:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "detail": exc.reason},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


def build_rate_limit_middleware(config: Optional[RateLimitConfig] = None) -> type:
    """
    Return a configured Starlette middleware class ready for app.add_middleware().

    Example:
        app.add_middleware(build_rate_limit_middleware())
    """
    limiter = RateLimiter(config)

    class _Configured(_RateLimitMiddleware):
        def __init__(self, app: ASGIApp) -> None:
            super().__init__(app, limiter)

    return _Configured
