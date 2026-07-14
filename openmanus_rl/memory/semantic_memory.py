"""
SemanticMemory — SQLiteMemory (S13) + векторный поиск (S14).

Хранит эмбеддинг каждого turn'а (BLOB float32) и ищет по косинусной близости
(numpy brute-force — достаточно для масштаба диалога). Без провайдера ведёт себя
как SQLiteMemory (semantic_search -> substring fallback). Не дублирует S13.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import EmbeddingProvider
from .sqlite_memory import SQLiteMemory


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q


class SemanticMemory(SQLiteMemory):
    """Персистентная диалоговая память с эмбеддинг-поиском."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None,
                 db_path: str = "conversations.db") -> None:
        self.provider = provider  # до super().__init__ (нужен в add_turn, не в _init_schema)
        super().__init__(db_path)

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB
                )""")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON turns(session_id, id)")
            self._conn.commit()

    def add_turn(self, session_id: str, role: str, content: str,
                 ts: Optional[float] = None) -> int:
        emb_blob = None
        if self.provider is not None:
            try:
                vec = self.provider.embed(content)
                if vec:
                    emb_blob = np.asarray(vec, dtype=np.float32).tobytes()
            except Exception:  # noqa: BLE001  (эмбеддинг не критичен для сохранения turn'а)
                emb_blob = None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns (session_id, ts, role, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (session_id, ts if ts is not None else time.time(), role, content, emb_blob))
            self._conn.commit()
            return int(cur.lastrowid)

    def semantic_search(self, session_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Top-k turn'ов по косинусной близости. Fallback на substring без провайдера/вектора."""
        if self.provider is None:
            return self.search(session_id, query, limit)
        try:
            qvec = self.provider.embed(query)
        except Exception:  # noqa: BLE001
            qvec = []
        if not qvec:
            return self.search(session_id, query, limit)

        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, embedding FROM turns "
                "WHERE session_id=? AND embedding IS NOT NULL", (session_id,)).fetchall()
        if not rows:
            return self.search(session_id, query, limit)

        q = np.asarray(qvec, dtype=np.float32)
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = _cosine_scores(q, matrix)
        order = np.argsort(-scores)[:limit]
        return [{"role": rows[i]["role"], "content": rows[i]["content"],
                 "score": float(scores[i])} for i in order]
