"""
Streaming + tools в одном потоке (S20) — снимает ограничение S15/S17.

Паттерн (по находке S15: streaming tool_calls на шлюзе ненадёжны — приходят как
JSON в content): инструменты разрешаем NON-stream (структурированные tool_calls),
затем стримим ФИНАЛ над сообщениями+результатами. В stream-запросе tools не
передаются → гарантированный текстовый ответ (без tool_calls в потоке).
"""
from typing import Any, AsyncGenerator, Dict, List, Optional

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter
from openmanus_rl.engines.tool_calling_adapter import ToolCallingAdapter
from openmanus_rl.tool_calling.registry import ToolRegistry


class StreamingToolCallingAdapter:
    """resolve инструментов (non-stream) → стриминг финального ответа."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 registry: Optional[ToolRegistry] = None) -> None:
        self.config = config or {}
        self.tools = ToolCallingAdapter(self.config, registry=registry)
        self.stream = StreamingLiteLLMAdapter(self.config)
        self.last_tools_used: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return self.stream.is_available()

    async def close(self) -> None:
        await self.stream.close()

    async def stream_chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncGenerator[str, None]:
        resolved, tools_used = self.tools.resolve(messages, **kwargs)
        self.last_tools_used = tools_used
        async for chunk in self.stream.stream_chat(resolved, **kwargs):
            yield chunk

    async def stream_generate(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        async for chunk in self.stream_chat([{"role": "user", "content": prompt}], **kwargs):
            yield chunk
