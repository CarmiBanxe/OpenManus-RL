"""
Диалоговый слой памяти (chat-shaped) поверх существующего memory-слоя.

ConversationMemory — тонкая обёртка: хранит turn'ы (role/content) в бэкенде
(SQLiteMemory по умолчанию; можно FileMemory-совместимый), отдаёт контекст для
промптов, при переполнении сжимает старые turn'ы через reused simple_summarize
(или простым усечением, если summarize=False). НЕ дублирует BaseMemory/FileMemory.
"""
from typing import Any, Callable, Dict, List, Optional

from .sqlite_memory import SQLiteMemory
from .summarized_memory import simple_summarize


class ConversationMemory:
    """Chat-обёртка над SQLiteMemory (turns role/content, контекст, сжатие)."""

    def __init__(self, backend: Optional[SQLiteMemory] = None, session_id: str = "default",
                 max_turns: int = 20, keep_recent: int = 6, summarize: bool = False,
                 summarizer: Optional[Callable[..., str]] = None,
                 summarize_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.backend = backend if backend is not None else SQLiteMemory()
        self.session_id = session_id
        self.max_turns = max_turns
        self.keep_recent = min(keep_recent, max_turns)
        self.summarize = summarize
        self._summarizer = summarizer or simple_summarize
        self.summarize_kwargs = summarize_kwargs or {}

    def store_turn(self, role: str, content: str) -> None:
        self.backend.add_turn(self.session_id, role, content)

    def get_context(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Последние turn'ы как список message-ей {role, content}."""
        turns = self.backend.get_turns(self.session_id, limit=max_turns or self.max_turns)
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    def get_messages_with_context(self, new_messages: List[Dict[str, str]],
                                  max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Префиксует накопленный контекст к новым сообщениям."""
        return self.get_context(max_turns) + list(new_messages)

    def query(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        return [{"role": t["role"], "content": t["content"]}
                for t in self.backend.search(self.session_id, query, limit)]

    def trim_if_needed(self) -> bool:
        """Если turn'ов больше max_turns — сжать (summarize) или усечь старые."""
        if self.backend.count(self.session_id) <= self.max_turns:
            return False
        if self.summarize:
            turns = self.backend.get_turns(self.session_id)
            old, recent = turns[:-self.keep_recent], turns[-self.keep_recent:]
            summary = self._summarizer(
                [f"{t['role']}: {t['content']}" for t in old], **self.summarize_kwargs)
            self.backend.clear(self.session_id)
            self.backend.add_turn(self.session_id, "system", f"[Summary of earlier conversation] {summary}")
            for t in recent:
                self.backend.add_turn(self.session_id, t["role"], t["content"])
        else:
            self.backend.trim_to(self.session_id, self.max_turns)
        return True

    def clear(self) -> None:
        self.backend.clear(self.session_id)

    def count(self) -> int:
        return self.backend.count(self.session_id)
