"""
Memory-aware обёртка над StreamingLiteLLMAdapter (S12) + ConversationMemory (S13).

Перед стримом подмешивает накопленный диалоговый контекст в messages, после —
сохраняет user/assistant turn'ы и сжимает при переполнении. Не подменяет S12:
композиция (has-a StreamingLiteLLMAdapter + ConversationMemory).
"""
from typing import Any, AsyncGenerator, Dict, List, Optional

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.sqlite_memory import SQLiteMemory

try:
    from openmanus_rl.observability import get_metrics_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBSERVABILITY_AVAILABLE = False


class MemoryAwareStreamingAdapter:
    """Стриминг с диалоговой памятью (context inject + persist)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 memory: Optional[ConversationMemory] = None) -> None:
        self.config = config or {}
        self.stream = StreamingLiteLLMAdapter(self.config)
        if memory is not None:
            self.memory = memory
        else:
            backend = SQLiteMemory(self.config.get("memory_db", "conversations.db"))
            self.memory = ConversationMemory(
                backend=backend,
                session_id=self.config.get("session_id", "default"),
                max_turns=int(self.config.get("max_turns", 20)),
                summarize=bool(self.config.get("memory_summarize", False)))
        self.enable_observability = bool(self.config.get("enable_observability", False))
        self._obs = self.enable_observability and OBSERVABILITY_AVAILABLE
        if self._obs:
            self.metrics = get_metrics_collector()

    def is_available(self) -> bool:
        return self.stream.is_available()

    async def close(self) -> None:
        await self.stream.close()

    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        context = self.memory.get_context()  # прежние turn'ы (до текущего обмена)
        for m in messages:
            if m.get("role") == "user":
                self.memory.store_turn("user", m["content"])
        full = context + list(messages)
        if self._obs:
            self.metrics.record_custom_metric(
                "openmanus_memory_context_turns", float(len(context)),
                {"session": self.memory.session_id})
        parts: List[str] = []
        async for chunk in self.stream.stream_chat(full, **kwargs):
            parts.append(chunk)
            yield chunk
        self.memory.store_turn("assistant", "".join(parts))
        self.memory.trim_if_needed()

    async def stream_generate(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        # completions-стиль тоже делаем диалого-осведомлённым через chat.
        async for chunk in self.stream_chat([{"role": "user", "content": prompt}], **kwargs):
            yield chunk

    # non-streaming делегируется базовому адаптеру (без изменения памяти).
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self.stream.generate(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return self.stream.chat(messages, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        return self.stream.get_metrics()


def create_memory_aware_adapter(config: Optional[Dict[str, Any]] = None) -> MemoryAwareStreamingAdapter:
    return MemoryAwareStreamingAdapter(config)
