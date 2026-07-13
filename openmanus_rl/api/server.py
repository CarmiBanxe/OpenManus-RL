"""
FastAPI-сервер OpenManus — под РЕАЛЬНЫЙ API агента (select_action), с JWT-auth.

SECURITY (приватный Legion-контур, S-18 §1.2 — красная линия):
  - НЕТ публичного/неаутентифицированного эндпоинта к движку (никакого /query/public);
  - host из конфига по умолчанию 127.0.0.1 (не 0.0.0.0);
  - CORS из конфига (localhost), не wildcard;
  - секрет и админ-креды ТОЛЬКО из env (никаких admin/admin, никакого дефолт-secret);
  - если секрет не задан — генерируется эфемерный на время процесса (токены не переживут рестарт).
"""
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openmanus_rl.monitoring import metrics as _metrics

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent
from openmanus_rl.auth.dependencies import get_current_user, init_jwt_auth
from openmanus_rl.auth.jwt_auth import JWTAuth
from openmanus_rl.config import load_config

logger = logging.getLogger(__name__)

_state: Dict[str, Any] = {"agent": None, "config": None, "admin_user": None, "admin_hash": None}


def _configure_auth() -> None:
    secret = os.environ.get("OPENMANUS_SECRET_KEY")
    if not secret:
        secret = secrets.token_urlsafe(32)
        logger.warning("OPENMANUS_SECRET_KEY not set — using ephemeral secret (tokens won't survive restart)")
    init_jwt_auth(secret)
    _state["admin_user"] = os.environ.get("OPENMANUS_ADMIN_USER")
    pw_hash = os.environ.get("OPENMANUS_ADMIN_PASSWORD_HASH")
    pw_plain = os.environ.get("OPENMANUS_ADMIN_PASSWORD")
    if pw_hash:
        _state["admin_hash"] = pw_hash
    elif pw_plain:
        _state["admin_hash"] = JWTAuth.hash_password(pw_plain)
    else:
        _state["admin_hash"] = None  # login disabled until configured


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(os.environ.get("OPENMANUS_CONFIG_FILE", "production"))
    _state["config"] = cfg
    _state["agent"] = EnhancedDecisionAgent(config=cfg)
    _configure_auth()
    logger.info("OpenManus API started (env=%s)", cfg["environment"])
    try:
        yield
    finally:
        agent = _state.get("agent")
        if agent is not None:
            try:
                await agent.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent cleanup error: %s", exc)


def create_app() -> FastAPI:
    cfg = load_config(os.environ.get("OPENMANUS_CONFIG_FILE", "production"))
    app = FastAPI(title="OpenManus API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg["cors_allow_origins"],  # localhost по умолчанию, не "*"
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    return app


app = create_app()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class QueryRequest(BaseModel):
    text: str
    available_actions: List[str] = ["proceed", "wait"]
    session_id: Optional[str] = None
    priority: float = 0.5
    entities: Optional[List[str]] = None


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "agent": _state.get("agent") is not None,
            "environment": (_state.get("config") or {}).get("environment")}


@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest) -> LoginResponse:
    admin_user, admin_hash = _state.get("admin_user"), _state.get("admin_hash")
    if not admin_user or not admin_hash:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Auth not configured (set OPENMANUS_ADMIN_USER / OPENMANUS_ADMIN_PASSWORD[_HASH])")
    if req.username != admin_user or not JWTAuth.verify_password(req.password, admin_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Bearer"})
    from openmanus_rl.auth.dependencies import get_jwt_auth
    auth = get_jwt_auth()
    return LoginResponse(access_token=auth.generate_token(admin_user),
                         expires_in=auth.token_expire_hours * 3600)


@app.post("/query")
async def query(req: QueryRequest, user: str = Depends(get_current_user)) -> Dict[str, Any]:
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialised")
    state: Dict[str, Any] = {"text": req.text, "user_id": user,
                             "session_id": req.session_id or str(uuid.uuid4())}
    if req.entities:
        state["entities"] = req.entities
    try:
        result = await agent.select_action(state, req.available_actions, priority=req.priority)
    except Exception as exc:  # noqa: BLE001
        logger.error("query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"query error: {exc}")
    return {"status": "success", "result": result}


@app.get("/config")
async def get_config(user: str = Depends(get_current_user)) -> Dict[str, Any]:
    cfg = _state.get("config") or {}
    safe_keys = ("environment", "host", "port", "enable_mean_field_games",
                 "enable_performance_optimization", "cors_allow_origins", "gradio_share")
    return {k: cfg.get(k) for k in safe_keys}


@app.middleware("http")
async def _metrics_middleware(request, call_next):
    import time as _t
    start = _t.perf_counter()
    response = await call_next(request)
    _metrics.record_request(request.method, request.url.path,
                            response.status_code, _t.perf_counter() - start)
    return response


@app.get("/metrics")
async def metrics_endpoint(user: str = Depends(get_current_user)) -> Response:
    # Красная линия (S-18 §1.2): /metrics ТОЛЬКО за аутентификацией, не публичный.
    body, content_type = _metrics.render()
    return Response(content=body, media_type=content_type)


if __name__ == "__main__":
    import uvicorn

    cfg = load_config(os.environ.get("OPENMANUS_CONFIG_FILE", "production"))
    uvicorn.run(app, host=cfg["host"], port=cfg["port"])  # 127.0.0.1 по умолчанию
