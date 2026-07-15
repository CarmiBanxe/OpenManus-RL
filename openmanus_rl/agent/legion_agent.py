"""
LegionAgent (S17) — единый фасад над всеми слоями движка.

Композиция (не reinvent): StreamingLiteLLMAdapter (S12), ToolCallingAdapter (S15),
ConversationMemory/SemanticMemory (S13/S14). Память централизована в агенте
(инъекция контекста + persist) — под-адаптеры получают её как None, чтобы не
дублировать записи. chat() = полный non-stream пайплайн (RAG+tools+memory);
stream() = RAG+memory+SSE (без tools — streaming tool_calls ненадёжны, см. S15).
"""
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter
from openmanus_rl.engines.tool_calling_adapter import ToolCallingAdapter
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.sqlite_memory import SQLiteMemory

from .config import AgentConfig
from .persona import Guardrails, resolve_system_prompt


class LegionAgent:
    """Единый интерфейс: chat/stream с автоматической памятью, RAG и tool-calling."""

    def __init__(self, config: Optional[AgentConfig] = None,
                 memory: Optional[ConversationMemory] = None, registry: Any = None) -> None:
        self.config = config or AgentConfig()
        cfg = self.config.engine_config()

        if memory is not None:
            self.memory: Optional[ConversationMemory] = memory
        elif self.config.memory:
            backend = self._build_backend()
            self.memory = ConversationMemory(
                backend=backend, session_id=self.config.session_id,
                max_turns=self.config.max_turns, summarize=self.config.memory_summarize)
        else:
            self.memory = None

        self.stream_adapter = StreamingLiteLLMAdapter(cfg)
        # под-адаптеры без своей памяти — агент централизует persist.
        self.tool_adapter = ToolCallingAdapter(cfg, registry=registry) if self.config.tools else None
        self.last_tools_used: List[Dict[str, Any]] = []
        # S22: persona (system-prompt) + операционные guardrails.
        self.system_prompt = resolve_system_prompt(self.config.persona, self.config.system_prompt)
        self.guardrails = Guardrails(max_input_chars=self.config.max_input_chars,
                                     deny_patterns=self.config.deny_patterns)

    def _build_backend(self):
        if self.config.rag:
            from openmanus_rl.memory.embeddings import OllamaEmbeddingProvider
            from openmanus_rl.memory.semantic_memory import SemanticMemory
            return SemanticMemory(
                OllamaEmbeddingProvider(self.config.embed_model, host=self.config.embed_host),
                self.config.memory_db)
        return SQLiteMemory(self.config.memory_db)

    def is_available(self) -> bool:
        return self.stream_adapter.is_available()

    def _context(self, message: str) -> List[Dict[str, str]]:
        if self.memory is None:
            return []
        if self.config.rag:
            return self.memory.get_relevant(message, self.config.rag_k)
        return self.memory.get_context()

    def _finish(self, content: str) -> None:
        if self.memory is not None:
            self.memory.store_turn("assistant", content)
            self.memory.trim_if_needed()

    def _prepare(self, message: str) -> List[Dict[str, str]]:
        self.guardrails.check(message)  # S22: raise GuardrailError при нарушении
        context = self._context(message)
        if self.memory is not None:
            self.memory.store_turn("user", message)
        system = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return system + context + [{"role": "user", "content": message}]

    def chat(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Полный non-stream пайплайн: RAG-контекст -> (tool-loop) -> ответ -> persist."""
        full = self._prepare(message)
        if self.tool_adapter is not None:
            result = self.tool_adapter.run(full, **kwargs)
            content, tools_used = result["content"], result["tools_used"]
        else:
            resp = self.stream_adapter.chat(full, **kwargs)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            tools_used = []
        self._finish(content)
        return {"content": content, "tools_used": tools_used}

    async def stream(self, message: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Стриминг ответа с RAG+памятью и (S20) tools: resolve non-stream -> стрим финала."""
        full = self._prepare(message)
        if self.tool_adapter is not None:
            stream_msgs, self.last_tools_used = self.tool_adapter.resolve(full, **kwargs)
        else:
            stream_msgs, self.last_tools_used = full, []
        parts: List[str] = []
        async for chunk in self.stream_adapter.stream_chat(stream_msgs, **kwargs):
            parts.append(chunk)
            yield chunk
        self._finish("".join(parts))

    def reset(self) -> None:
        if self.memory is not None:
            self.memory.clear()

    async def close(self) -> None:
        await self.stream_adapter.close()


def create_agent(config: Union[AgentConfig, Dict[str, Any], None] = None, **kwargs: Any) -> LegionAgent:
    if isinstance(config, dict):
        config = AgentConfig.from_dict(config)
    elif config is None:
        config = AgentConfig(**kwargs) if kwargs else AgentConfig()
    return LegionAgent(config)
