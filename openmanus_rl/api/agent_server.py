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


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    params: Dict[str, Any] = {}


def _server_config() -> Dict[str, Any]:
    def _b(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes")
    return {
        "model": os.environ.get("LEGION_MODEL", "smart"),
        "rag": _b("LEGION_RAG"),
        "tools": _b("LEGION_TOOLS"),
        "memory": os.environ.get("LEGION_MEMORY", "1").lower() in ("1", "true", "yes"),
        "memory_db": os.environ.get("LEGION_MEMORY_DB", ":memory:"),
        "enable_observability": _b("LEGION_OBSERVABILITY"),
    }


def create_agent_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="Legion Agent API")
    base_cfg = config or _server_config()
    agents: Dict[str, LegionAgent] = {}

    def get_agent(session_id: str) -> LegionAgent:
        if session_id not in agents:
            agents[session_id] = LegionAgent(
                AgentConfig.from_dict({**base_cfg, "session_id": session_id}))
        return agents[session_id]

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
        try:
            return get_agent(req.session_id).chat(req.message, **(req.params or {}))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}")

    @app.post("/stream")
    def stream(req: ChatRequest, _: None = Depends(require_auth)):
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
        if req.session_id in agents:
            agents[req.session_id].reset()
        return {"status": "reset", "session_id": req.session_id}

    return app


app = create_agent_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
