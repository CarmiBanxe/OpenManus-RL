"""Безопасное исполнение инструментов (таймаут + перехват ошибок)."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict

from .registry import ToolRegistry


class ToolExecutor:
    """Исполняет зарегистрированный инструмент с таймаутом; ошибки -> строка, не падение."""

    def __init__(self, registry: ToolRegistry, timeout: float = 10.0) -> None:
        self.registry = registry
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=4)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self.registry.has(name):
            return f"Error: unknown tool '{name}'"
        tool = self.registry.get(name)
        try:
            future = self._pool.submit(tool.func, **(arguments or {}))
            return str(future.result(timeout=self.timeout))
        except FuturesTimeout:
            return f"Error: tool '{name}' timed out after {self.timeout}s"
        except Exception as exc:  # noqa: BLE001  (инструмент не должен ронять агента)
            return f"Error: tool '{name}' failed: {exc}"

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
