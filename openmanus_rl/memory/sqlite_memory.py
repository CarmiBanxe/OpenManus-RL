"""
SQLite-бэкенд для диалоговой памяти (stdlib sqlite3).

SQLiteMemory расширяет SimpleMemory (совместим с BaseMemory), добавляя
персистентное хранение диалоговых turn'ов (role/content по session_id).
Batch-API (reset/store/fetch) наследуется без изменений — форма не ломается.
"""
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .memory import SimpleMemory


class SQLiteMemory(SimpleMemory):
    """Персистентное хранилище диалоговых turn'ов поверх SimpleMemory."""

    def __init__(self, db_path: str = "conversations.db") -> None:
        super().__init__()
        self.db_path = db_path
        self._lock = threading.Lock()
        # check_same_thread=False: используем один коннект под своим локом.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )""")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON turns(session_id, id)")
            self._conn.commit()

    def add_turn(self, session_id: str, role: str, content: str,
                 ts: Optional[float] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns (session_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (session_id, ts if ts is not None else time.time(), role, content))
            self._conn.commit()
            return int(cur.lastrowid)

    def get_turns(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if limit is None:
                rows = self._conn.execute(
                    "SELECT role, content, ts FROM turns WHERE session_id=? ORDER BY id ASC",
                    (session_id,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT role, content, ts FROM turns WHERE session_id=? "
                    "ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
                rows = list(reversed(rows))
            return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]

    def search(self, session_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM turns WHERE session_id=? AND content LIKE ? "
                "ORDER BY id DESC LIMIT ?", (session_id, f"%{query}%", limit)).fetchall()
            return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]

    def count(self, session_id: str) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id=?", (session_id,)).fetchone()[0])

    def trim_to(self, session_id: str, keep: int) -> int:
        """Удалить самые старые turn'ы, оставив последние `keep`. Возвращает число удалённых."""
        with self._lock:
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM turns WHERE session_id=? ORDER BY id DESC LIMIT -1 OFFSET ?",
                (session_id, keep)).fetchall()]
            if ids:
                self._conn.executemany("DELETE FROM turns WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
            return len(ids)

    def clear(self, session_id: Optional[str] = None) -> None:
        with self._lock:
            if session_id is None:
                self._conn.execute("DELETE FROM turns")
            else:
                self._conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
