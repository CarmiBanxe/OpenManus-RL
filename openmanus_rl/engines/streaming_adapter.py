"""
Streaming-адаптер для LiteLLM (async, SSE) с опциональной observability.

Non-streaming методы делегируются одному переиспользуемому LiteLLMAdapter (S10),
чтобы не плодить health-GET и не терять метрики. Retry — только ДО первого токена
(иначе потребитель получил бы начало ответа дважды). Observability — opt-in (S11 API).
"""
import asyncio
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
import requests

from openmanus_rl.engines.enhanced_factory import LiteLLMAdapter

try:
    from openmanus_rl.observability import get_logger, get_metrics_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBSERVABILITY_AVAILABLE = False

_JSON = None
try:
    import orjson as _JSON  # type: ignore  # быстрый, если есть
except Exception:  # noqa: BLE001
    import json as _JSON  # type: ignore


class StreamingLiteLLMAdapter:
    """LiteLLM-адаптер с потоковой (SSE) генерацией."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.base_url = self.config.get("base_url", "http://localhost:4000")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.timeout = int(self.config.get("timeout", 60))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.fallback_models = list(self.config.get("fallback_models", []))
        self.master_key = self.config.get("master_key") or os.environ.get("LITELLM_MASTER_KEY", "")
        # Legion-шлюз содержит reasoning-модели (delta.reasoning_content до delta.content).
        # По умолчанию отдаём только ответ (content); include_reasoning=True — и «мышление».
        self.include_reasoning = bool(self.config.get("include_reasoning", False))

        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._sync: Optional[LiteLLMAdapter] = None  # ленивый non-streaming делегат

        self._available: Optional[bool] = None
        self._check_litellm_availability()

        self.enable_observability = bool(self.config.get("enable_observability", False))
        self._obs = self.enable_observability and OBSERVABILITY_AVAILABLE
        if self._obs:
            self.metrics = get_metrics_collector()
            self.logger = get_logger()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self.session is None or self.session.closed:
                headers = {"Authorization": f"Bearer {self.master_key}"} if self.master_key else {}
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout), headers=headers)
            return self.session

    async def close(self) -> None:
        async with self._session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    def _check_litellm_availability(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code in (200, 401):
                self._available = True
                return True
        except Exception:  # noqa: BLE001
            pass
        self._available = False
        return False

    def is_available(self) -> bool:
        return self._available is True

    def _health_check(self) -> Dict[str, Any]:
        if self.is_available():
            return {"status": "healthy", "message": "LiteLLM is available"}
        return {"status": "unhealthy", "message": "LiteLLM is not available"}

    def _extract_content(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            if delta.get("content"):
                return delta["content"]
            if self.include_reasoning and delta.get("reasoning_content"):
                return delta["reasoning_content"]
            return ""
        return choice.get("text", "") or ""

    async def _make_stream_request(self, endpoint: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Один потоковый запрос к LiteLLM (SSE). Метрики — opt-in."""
        session = await self._get_session()
        request_id = str(uuid.uuid4())
        start = time.time()
        tokens = 0
        first_token_at: Optional[float] = None
        status = "success"

        if self._obs:
            self.metrics.record_request_start("litellm", self.model)
            self.logger.request(engine_type="litellm", model=self.model,
                                operation="stream", request_id=request_id, endpoint=endpoint)
        try:
            async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                if response.status != 200:
                    status = "error"
                    text = await response.text()
                    raise RuntimeError(f"Ошибка запроса к LiteLLM: {response.status}: {text[:200]}")
                # aiohttp StreamReader итерируется построчно (readline) — SSE-safe.
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = _JSON.loads(data_str)
                    except Exception:  # noqa: BLE001  (битый чанк)
                        continue
                    content = self._extract_content(data)
                    if content:
                        if first_token_at is None:
                            first_token_at = time.time() - start
                        tokens += 1
                        if self._obs:
                            self.metrics.record_tokens("litellm", self.model, 1)
                        yield content
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            status = "error"
            raise  # пробрасываем для retry-обёртки (не оборачиваем в RuntimeError)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            if self._obs:
                self.logger.error(engine_type="litellm", operation="stream",
                                request_id=request_id, error=str(exc))
            raise
        finally:
            if self._obs:
                duration = time.time() - start
                if status == "error":
                    self.metrics.record_error("litellm", "stream_error")
                self.metrics.record_request_end("litellm", self.model, status, "stream", duration, tokens)
                if first_token_at is not None:
                    self.metrics.record_custom_metric(
                        "openmanus_streaming_ttft_seconds", first_token_at,
                        {"engine_type": "litellm", "model": self.model})
                    if duration > 0 and tokens > 0:
                        self.metrics.record_custom_metric(
                            "openmanus_streaming_tokens_per_second", tokens / duration,
                            {"engine_type": "litellm", "model": self.model})

    async def _stream_with_retry(self, endpoint: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Retry ТОЛЬКО до первого выданного токена (без дублей ответа)."""
        yielded = False
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in self._make_stream_request(endpoint, payload):
                    yielded = True
                    yield chunk
                return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if yielded or attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    def _payload(self, extra: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": kwargs.get("model", self.model), "stream": True,
            "temperature": kwargs.get("temperature", 0.7), "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048), **extra,
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        return payload

    async def stream_generate(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        extra: Dict[str, Any] = {"prompt": prompt}
        if "system" in kwargs:
            extra["system"] = kwargs["system"]
        async for chunk in self._stream_with_retry("/v1/completions", self._payload(extra, kwargs)):
            yield chunk

    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        async for chunk in self._stream_with_retry("/v1/chat/completions",
                                                    self._payload({"messages": messages}, kwargs)):
            yield chunk

    # --- non-streaming делегирование к одному переиспользуемому LiteLLMAdapter ---
    def _get_sync(self) -> LiteLLMAdapter:
        if self._sync is None:
            self._sync = LiteLLMAdapter(self.config)
        return self._sync

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self._get_sync().generate(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return self._get_sync().chat(messages, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        return self._get_sync().get_metrics()

    def list_models(self) -> List[Dict[str, Any]]:
        return self._get_sync().list_models()

    def model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        return self._get_sync().model_info(model_name)


def create_streaming_adapter(config: Optional[Dict[str, Any]] = None) -> StreamingLiteLLMAdapter:
    return StreamingLiteLLMAdapter(config)
