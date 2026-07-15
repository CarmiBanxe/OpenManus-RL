"""Streaming API для OpenManus: WebSocket (/ws/stream) + SSE (/v1/stream/generate). 127.0.0.1."""
import json
import os
import sys
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter

_HTML = """<!DOCTYPE html>
<html><head><title>OpenManus Streaming Test</title></head>
<body><h1>OpenManus Streaming Test</h1>
<textarea id="prompt" rows="4" cols="50"></textarea><button onclick="send()">Send</button>
<p>Status: <span id="status">Disconnected</span></p><pre id="response"></pre>
<script>
const ws = new WebSocket("ws://localhost:8081/ws/stream");
ws.onopen = () => document.getElementById("status").textContent = "Connected";
ws.onmessage = (e) => { const d = JSON.parse(e.data);
  if (d.error) document.getElementById("response").textContent += "Error: " + d.error;
  else document.getElementById("response").textContent += d.chunk;
  if (d.finished) document.getElementById("status").textContent = "Finished"; };
ws.onclose = () => document.getElementById("status").textContent = "Disconnected";
function send(){ ws.send(JSON.stringify({request_id: Math.random().toString(36).slice(2),
  prompt: document.getElementById("prompt").value, model:"gpt-3.5-turbo", max_tokens:500}));
  document.getElementById("response").textContent=""; }
</script></body></html>"""


class StreamResponse(BaseModel):
    request_id: str
    chunk: str
    finished: bool = False
    error: Optional[str] = None


def create_streaming_app() -> FastAPI:
    app = FastAPI(title="OpenManus Streaming API")
    active_connections: Dict[str, WebSocket] = {}

    @app.get("/")
    async def get_root():
        return HTMLResponse(_HTML)

    @app.websocket("/ws/stream")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        active_connections[connection_id] = websocket
        try:
            while True:
                request_data = json.loads(await websocket.receive_text())
                request_id = request_data.get("request_id", str(uuid.uuid4()))
                prompt = request_data.get("prompt")
                messages = request_data.get("messages")
                adapter = StreamingLiteLLMAdapter({
                    "model": request_data.get("model", "gpt-3.5-turbo")})
                try:
                    kw = {"temperature": request_data.get("temperature", 0.7),
                          "max_tokens": request_data.get("max_tokens", 2048)}
                    if prompt:
                        async for chunk in adapter.stream_generate(prompt, **kw):
                            await websocket.send_text(StreamResponse(request_id=request_id, chunk=chunk).json())
                    elif messages:
                        async for chunk in adapter.stream_chat(messages, **kw):
                            await websocket.send_text(StreamResponse(request_id=request_id, chunk=chunk).json())
                    else:
                        await websocket.send_text(StreamResponse(
                            request_id=request_id, chunk="", error="No prompt or messages provided").json())
                    await websocket.send_text(StreamResponse(
                        request_id=request_id, chunk="", finished=True).json())
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_text(StreamResponse(
                        request_id=request_id, chunk="", error=str(exc)).json())
                finally:
                    await adapter.close()
        except WebSocketDisconnect:
            pass
        finally:
            active_connections.pop(connection_id, None)

    @app.get("/v1/stream/generate")
    async def stream_generate_sse(prompt: str, model: str = "gpt-3.5-turbo",
                                  temperature: float = 0.7, max_tokens: int = 2048):
        async def event_stream():
            adapter = StreamingLiteLLMAdapter({"model": model})
            try:
                async for chunk in adapter.stream_generate(
                        prompt, temperature=temperature, max_tokens=max_tokens):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                await adapter.close()
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/connections")
    async def get_connections():
        return {"active_connections": len(active_connections),
                "connection_ids": list(active_connections.keys())}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_streaming_app(), host="127.0.0.1", port=8081)
