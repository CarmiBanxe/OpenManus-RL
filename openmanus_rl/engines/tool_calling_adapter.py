"""
Tool-calling адаптер (S15): агентик-луп поверх LiteLLMAdapter (S10, non-streaming).

Разведка показала: шлюз надёжно отдаёт СТРУКТУРИРОВАННЫЕ tool_calls в non-stream
(в streaming они текут как JSON в content — ненадёжно для парсинга). Поэтому
tool-решения делаем non-stream, инструменты исполняем безопасно, результат
возвращаем модели, пока не будет финального ответа (или лимит итераций).
Опционально сохраняем обмен в ConversationMemory (S13).
"""
import json
from typing import Any, Dict, List, Optional

from openmanus_rl.engines.enhanced_factory import LiteLLMAdapter
from openmanus_rl.tool_calling.builtins import register_builtins
from openmanus_rl.tool_calling.executor import ToolExecutor
from openmanus_rl.tool_calling.registry import ToolRegistry


class ToolCallingAdapter:
    """Агентик tool-calling поверх LiteLLM (non-streaming)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 registry: Optional[ToolRegistry] = None, memory: Any = None) -> None:
        self.config = config or {}
        self.llm = LiteLLMAdapter(self.config)
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.registry = registry if registry is not None else register_builtins(ToolRegistry())
        self.executor = ToolExecutor(self.registry, timeout=float(self.config.get("tool_timeout", 10.0)))
        self.max_iters = int(self.config.get("max_tool_iters", 5))
        self.memory = memory

    def is_available(self) -> bool:
        return self.llm.is_available()

    def run(self, messages: List[Dict[str, Any]], model: Optional[str] = None,
            tool_choice: str = "auto", **kwargs: Any) -> Dict[str, Any]:
        """Прогнать tool-calling луп; вернуть {content, tools_used, iterations, messages}."""
        msgs: List[Dict[str, Any]] = [dict(m) for m in messages]
        if self.memory is not None:
            for m in messages:
                if m.get("role") == "user":
                    self.memory.store_turn("user", m["content"])
        tools_used: List[Dict[str, Any]] = []
        seen: set = set()  # сигнатуры (name,args) — защита от зацикливания на повторах

        for i in range(self.max_iters):
            payload = {
                "model": model or self.model, "messages": msgs, "stream": False,
                "tools": self.registry.schemas(), "tool_choice": tool_choice,
                "temperature": kwargs.get("temperature", 0),
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            result = self.llm._make_request("/v1/chat/completions", payload)
            msg = result["choices"][0]["message"]
            tool_calls = msg.get("tool_calls")

            if not tool_calls:
                content = msg.get("content", "") or ""
                if self.memory is not None:
                    self.memory.store_turn("assistant", content)
                    self.memory.trim_if_needed()
                return {"content": content, "tools_used": tools_used,
                        "iterations": i + 1, "messages": msgs}

            msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
            fresh = False
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                sig = (name, json.dumps(args, sort_keys=True))
                if sig not in seen:
                    fresh = True
                    seen.add(sig)
                output = self.executor.execute(name, args)
                tools_used.append({"name": name, "arguments": args, "output": output})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "name": name, "content": output})
            if not fresh:  # только повторные вызовы -> к форсированному финалу
                break

        # лимит итераций (или только повторы) — форсируем финальный ответ без инструментов
        final = self.llm._make_request("/v1/chat/completions", {
            "model": model or self.model, "messages": msgs, "stream": False,
            "temperature": kwargs.get("temperature", 0), "max_tokens": kwargs.get("max_tokens", 1024)})
        content = final["choices"][0]["message"].get("content", "") or ""
        if self.memory is not None:
            self.memory.store_turn("assistant", content)
            self.memory.trim_if_needed()
        return {"content": content, "tools_used": tools_used,
                "iterations": i + 1, "messages": msgs, "truncated": True}


def create_tool_calling_adapter(config: Optional[Dict[str, Any]] = None,
                                registry: Optional[ToolRegistry] = None) -> ToolCallingAdapter:
    return ToolCallingAdapter(config, registry)
