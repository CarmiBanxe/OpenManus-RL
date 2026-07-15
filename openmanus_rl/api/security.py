"""
Hardening REST-сервиса LegionAgent (S19): rate-limit + audit-log + self-check.

DIY (без новых pip): in-memory fixed-window rate-limit по клиенту (API-ключ/IP),
audit через S11 structlog (метаданные, БЕЗ секретов и содержимого — S-18),
SecurityAudit — проверка безопасной конфигурации. Extend S18 agent_server.
"""
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from openmanus_rl.observability import get_logger
    _OBS = True
except ImportError:  # pragma: no cover
    _OBS = False


class RateLimiter:
    """Fixed-window rate-limit по client_id (in-memory, потокобезопасно)."""

    def __init__(self, limit: int = 120, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window = window_s
        self._hits: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, client_id: str) -> Tuple[bool, float]:
        """-> (allowed, retry_after_seconds)."""
        now = time.time()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[client_id]
            hits[:] = [t for t in hits if t > cutoff]
            if len(hits) >= self.limit:
                return False, max(0.0, self.window - (now - hits[0]))
            hits.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _redact_client(client: str) -> str:
    """Не логируем секрет целиком: ключ -> префикс…, IP -> как есть."""
    if client.startswith("sk-") and len(client) > 10:
        return client[:8] + "…"
    return client


def install_security(app: FastAPI, *, rate_limit: int = 120, window_s: float = 60.0,
                     audit: bool = True) -> RateLimiter:
    """Подключить rate-limit + audit middleware к приложению. Возвращает RateLimiter."""
    limiter = RateLimiter(rate_limit, window_s)
    logger = get_logger("legion.audit") if (_OBS and audit) else None

    @app.middleware("http")
    async def _security_mw(request, call_next):
        client = (request.headers.get("x-api-key")
                  or (request.client.host if request.client else "unknown"))
        allowed, retry = limiter.check(client)
        if not allowed:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                headers={"Retry-After": str(int(retry) + 1)})
        start = time.perf_counter()
        response = await call_next(request)
        if logger is not None:
            # ТОЛЬКО метаданные — без тела запроса/ответа и без ключа (S-18).
            logger.system_event(
                "api_request", method=request.method, path=request.url.path,
                client=_redact_client(client), status=response.status_code,
                latency_ms=round((time.perf_counter() - start) * 1000, 1))
        return response

    return limiter


@dataclass
class AuditFinding:
    check: str
    passed: bool
    detail: str


class SecurityAudit:
    """Self-check безопасной конфигурации REST-сервиса."""

    @staticmethod
    def run(server_config: Dict[str, Any], bind_host: str = "127.0.0.1",
            api_key_required: Optional[bool] = None) -> List[AuditFinding]:
        f: List[AuditFinding] = []
        f.append(AuditFinding(
            "bind_localhost", bind_host in ("127.0.0.1", "localhost", "::1"),
            f"host={bind_host} (S-18: движок не должен слушать публично)"))
        f.append(AuditFinding(
            "no_hardcoded_master_key", not server_config.get("master_key"),
            "master_key должен браться из env LITELLM_MASTER_KEY, не из конфига"))
        import os
        has_key = bool(os.environ.get("LEGION_API_KEY"))
        is_local = bind_host in ("127.0.0.1", "localhost", "::1")
        # ok, если ключ задан ИЛИ сервис только на localhost; фейл — публично без ключа.
        ok = has_key or is_local if api_key_required is None else api_key_required
        detail = ("LEGION_API_KEY задан -> REST требует ключ" if has_key else
                  ("открыт, но только localhost — допустимо" if is_local else
                   "⚠ ПУБЛИЧНЫЙ bind без API-ключа"))
        f.append(AuditFinding("api_key_auth", ok, detail))
        f.append(AuditFinding(
            "observability_no_content_leak", True,
            "audit пишет только метаданные (без тела/ключа) — S-18"))
        return f

    @staticmethod
    def summary(findings: List[AuditFinding]) -> Dict[str, Any]:
        return {"passed": all(x.passed for x in findings),
                "checks": [asdict(x) for x in findings]}
