"""Минимальный клиентский SDK для streaming API (WebSocket + SSE)."""
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List

import websockets


class StreamingClient:
    def __init__(self, base_url: str = "http://localhost:8081") -> None:
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws", 1)

    async def _ws_stream(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        async with websockets.connect(f"{self.ws_url}/ws/stream") as ws:
            await ws.send(json.dumps(request))
            async for message in ws:
                data = json.loads(message)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                if data.get("chunk"):
                    yield data["chunk"]
                if data.get("finished"):
                    break

    async def stream_generate(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        async for chunk in self._ws_stream(
                {"request_id": str(uuid.uuid4()), "prompt": prompt, **kwargs}):
            yield chunk

    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        async for chunk in self._ws_stream(
                {"request_id": str(uuid.uuid4()), "messages": messages, **kwargs}):
            yield chunk

    async def stream_generate_sse(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            params = {"prompt": prompt, **{k: str(v) for k, v in kwargs.items()}}
            async with session.get(f"{self.base_url}/v1/stream/generate", params=params) as response:
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    if data.get("content"):
                        yield data["content"]
