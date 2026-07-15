"""
SessionManager (S21): по-сессионные LegionAgent'ы с TTL и лимитом.

Агенты живут в RAM с TTL (idle -> evict), а память персистит в общем файловом
SQLite (session_id изолирует сессии — S13). Evict освобождает RAM; при повторном
обращении session_id агент пересоздаётся и перечитывает историю из БД.
"""
import threading
import time
from typing import Callable, Dict, List, Optional

from .legion_agent import LegionAgent


class SessionManager:
    def __init__(self, factory: Callable[[str], LegionAgent],
                 ttl_s: float = 3600.0, max_sessions: int = 1000) -> None:
        self._factory = factory
        self.ttl = ttl_s
        self.max_sessions = max_sessions
        self._store: Dict[str, list] = {}  # sid -> [agent, last_access]
        self._lock = threading.Lock()

    def _evict_expired_unlocked(self) -> None:
        cutoff = time.time() - self.ttl
        for sid in [s for s, (_, t) in self._store.items() if t < cutoff]:
            del self._store[sid]

    def get(self, session_id: str) -> LegionAgent:
        with self._lock:
            self._evict_expired_unlocked()
            if session_id not in self._store:
                if len(self._store) >= self.max_sessions:
                    oldest = min(self._store, key=lambda s: self._store[s][1])
                    del self._store[oldest]
                self._store[session_id] = [self._factory(session_id), time.time()]
            self._store[session_id][1] = time.time()
            return self._store[session_id][0]

    def reset(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                self._store[session_id][0].reset()
                return True
            return False

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._evict_expired_unlocked()
            return len(self._store)

    def sessions(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())
