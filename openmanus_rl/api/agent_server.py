"""
REST-сервер для LegionAgent (S18) на FastAPI.

Эндпоинты: /health, POST /chat, POST /stream (SSE), POST /reset. По-сессионные
агенты (lazy). S-18: bind 127.0.0.1; опц. API-ключ из env LEGION_API_KEY (если
задан — требуется X-API-Key/Authorization). Секреты/модель — из окружения.
"""
import json
import os
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openmanus_rl.agent import AgentConfig, LegionAgent
from openmanus_rl.agent.persona import GuardrailError
from openmanus_rl.agent.session_manager import SessionManager
from openmanus_rl.api.security import SecurityAudit, install_security
from openmanus_rl.guardrails.policy import check_request
from openmanus_rl.middleware.rate_limiter import build_rate_limit_middleware


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    params: Dict[str, Any] = {}


def _server_config() -> Dict[str, Any]:
    def _b(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes")
    return {
        "model": os.environ.get("LEGION_MODEL", "smart"),
        # S23: шлюз/Ollama конфигурируемы (в контейнере -> host.docker.internal).
        "base_url": os.environ.get("LEGION_BASE_URL", "http://localhost:4000"),
        "embed_host": os.environ.get("LEGION_EMBED_HOST", "localhost"),
        "rag": _b("LEGION_RAG"),
        "tools": _b("LEGION_TOOLS"),
        "memory": os.environ.get("LEGION_MEMORY", "1").lower() in ("1", "true", "yes"),
        # S21: по умолчанию ПЕРСИСТЕНТНЫЙ файловый db (общий, session_id изолирует).
        "memory_db": os.environ.get("LEGION_MEMORY_DB", "legion_memory.db"),
        "enable_observability": _b("LEGION_OBSERVABILITY"),
        # S22: persona / guardrails из env (deny-list пуст по умолчанию — S-18).
        "persona": os.environ.get("LEGION_PERSONA") or None,
        "system_prompt": os.environ.get("LEGION_SYSTEM_PROMPT") or None,
        "max_input_chars": int(os.environ.get("LEGION_MAX_INPUT", "100000")),
        "deny_patterns": [p for p in os.environ.get("LEGION_DENY", "").split(",") if p],
    }


def create_agent_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="Legion Agent API")
    # Rate limiter: config from RATE_LIMIT_* env vars (see middleware/rate_limiter.py)
    app.add_middleware(build_rate_limit_middleware())
    base_cfg = config or _server_config()

    # S19: rate-limit + audit middleware (DIY, без секретов/контента в логах).
    install_security(app, rate_limit=int(os.environ.get("LEGION_RATE_LIMIT", "120")),
                     audit=os.environ.get("LEGION_AUDIT", "1").lower() in ("1", "true", "yes"))

    # S21: сессии с TTL/лимитом; память персистит в общем файловом db (base_cfg.memory_db).
    sessions = SessionManager(
        lambda sid: LegionAgent(AgentConfig.from_dict({**base_cfg, "session_id": sid})),
        ttl_s=float(os.environ.get("LEGION_SESSION_TTL", "3600")),
        max_sessions=int(os.environ.get("LEGION_MAX_SESSIONS", "1000")))

    def get_agent(session_id: str) -> LegionAgent:
        return sessions.get(session_id)

    def require_auth(x_api_key: Optional[str] = Header(None),
                     authorization: Optional[str] = Header(None)) -> None:
        required = os.environ.get("LEGION_API_KEY")
        if not required:
            return
        token = x_api_key or (authorization.split()[-1] if authorization else None)
        if token != required:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health():
        try:
            available = get_agent("default").is_available()
        except Exception as exc:  # noqa: BLE001
            return {"status": "degraded", "error": str(exc)}
        return {"status": "ok" if available else "degraded",
                "available": available, "model": base_cfg["model"]}

    @app.post("/chat")
    def chat(req: ChatRequest, _: None = Depends(require_auth)):
        violation = check_request(req.message)
        if violation:
            raise HTTPException(status_code=400, detail={"policy": violation.rule, "detail": violation.detail})
        try:
            return get_agent(req.session_id).chat(req.message, **(req.params or {}))
        except GuardrailError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}")

    @app.post("/stream")
    def stream(req: ChatRequest, _: None = Depends(require_auth)):
        violation = check_request(req.message)
        if violation:
            raise HTTPException(status_code=400, detail={"policy": violation.rule, "detail": violation.detail})
        agent = get_agent(req.session_id)

        async def event_stream():
            try:
                async for chunk in agent.stream(req.message, **(req.params or {})):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/reset")
    def reset(req: ChatRequest, _: None = Depends(require_auth)):
        ok = sessions.reset(req.session_id)
        return {"status": "reset" if ok else "no_session", "session_id": req.session_id}

    @app.get("/sessions")
    def list_sessions(_: None = Depends(require_auth)):
        return {"active_sessions": sessions.count(), "ttl_s": sessions.ttl,
                "max_sessions": sessions.max_sessions}

    @app.get("/security/audit")
    def security_audit(_: None = Depends(require_auth)):
        findings = SecurityAudit.run(base_cfg, bind_host=os.environ.get("LEGION_BIND", "127.0.0.1"))
        return SecurityAudit.summary(findings)

    return app


app = create_agent_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
