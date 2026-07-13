"""
Hermes Memory Integration — SQLite-backed persistent memory for EnhancedDecisionAgent
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Default location — can be overridden in config
DEFAULT_DB_PATH = Path.home() / ".openmanus" / "hermes_memory.db"


class HermesMemoryIntegration:
    """
    SQLite-backed memory system with 4 stores:
    - working   : short-term session state (cleared on session end)
    - semantic  : fact/concept storage (long-term, searchable)
    - episodic  : time-tagged event log (append-only)
    - skills    : learned decision patterns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        db_path_str: str = self.config.get("db_path", str(DEFAULT_DB_PATH))
        self.db_path = Path(db_path_str)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        logger.info(f"HermesMemoryIntegration initialized: {self.db_path}")

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _initialize_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS working_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_id TEXT
                );

                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept TEXT NOT NULL,
                    facts TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    action_taken TEXT,
                    outcome TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skills_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL UNIQUE,
                    skill_data TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Working memory (session-scoped)
    # ------------------------------------------------------------------

    def store_working(
        self,
        key: str,
        value: Any,
        session_id: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        serialized = json.dumps(value, default=str)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO working_memory (key, value, created_at, updated_at, session_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    session_id = excluded.session_id
                """,
                (key, serialized, now, now, session_id),
            )

    def retrieve_working(self, key: str) -> Optional[Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM working_memory WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def clear_working_session(self, session_id: str) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM working_memory WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Semantic memory (facts / concepts)
    # ------------------------------------------------------------------

    def store_semantic(
        self,
        concept: str,
        facts: Dict[str, Any],
        confidence: float = 0.8,
        source: Optional[str] = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        serialized = json.dumps(facts, default=str)
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_memory (concept, facts, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (concept, serialized, confidence, source, now, now),
            )
            return cursor.lastrowid or 0

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, concept, facts, confidence, source, created_at
                FROM semantic_memory
                WHERE (concept LIKE ? OR facts LIKE ?)
                  AND confidence >= ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", min_confidence, limit),
            ).fetchall()

        results = []
        for row in rows:
            try:
                facts = json.loads(row["facts"])
            except (json.JSONDecodeError, TypeError):
                facts = {"raw": row["facts"]}
            results.append(
                {
                    "id": row["id"],
                    "concept": row["concept"],
                    "facts": facts,
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Episodic memory (event log — append-only, I-24 style)
    # ------------------------------------------------------------------

    def store_episode(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        action_taken: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        serialized = json.dumps(event_data, default=str)
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodic_memory (event_type, event_data, action_taken, outcome, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, serialized, action_taken, outcome, now),
            )
            return cursor.lastrowid or 0

    def retrieve_episodes(
        self,
        event_type: Optional[str] = None,
        limit: int = 50,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params: List[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, event_type, event_data, action_taken, outcome, timestamp
                FROM episodic_memory
                {where}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        results = []
        for row in rows:
            try:
                data = json.loads(row["event_data"])
            except (json.JSONDecodeError, TypeError):
                data = {"raw": row["event_data"]}
            results.append(
                {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "event_data": data,
                    "action_taken": row["action_taken"],
                    "outcome": row["outcome"],
                    "timestamp": row["timestamp"],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Skills memory (decision patterns)
    # ------------------------------------------------------------------

    def store_skill(
        self,
        skill_name: str,
        skill_data: Dict[str, Any],
        success_rate: float = 0.5,
    ) -> None:
        now = datetime.utcnow().isoformat()
        serialized = json.dumps(skill_data, default=str)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO skills_memory (skill_name, skill_data, success_rate, usage_count, created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    skill_data = excluded.skill_data,
                    success_rate = excluded.success_rate,
                    updated_at = excluded.updated_at
                """,
                (skill_name, serialized, success_rate, now, now),
            )

    def retrieve_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM skills_memory WHERE skill_name = ?", (skill_name,)
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["skill_data"])
        except (json.JSONDecodeError, TypeError):
            data = {"raw": row["skill_data"]}
        return {
            "skill_name": row["skill_name"],
            "skill_data": data,
            "success_rate": row["success_rate"],
            "usage_count": row["usage_count"],
        }

    def increment_skill_usage(self, skill_name: str, success: bool = True) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT usage_count, success_rate FROM skills_memory WHERE skill_name = ?",
                (skill_name,),
            ).fetchone()
            if row is None:
                return
            count = row["usage_count"] + 1
            rate = (row["success_rate"] * row["usage_count"] + (1.0 if success else 0.0)) / count
            conn.execute(
                "UPDATE skills_memory SET usage_count = ?, success_rate = ?, updated_at = ? WHERE skill_name = ?",
                (count, rate, now, skill_name),
            )

    # ------------------------------------------------------------------
    # Retrieval for EnhancedDecisionAgent
    # ------------------------------------------------------------------

    def get_relevant_memories(
        self,
        context: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Retrieves relevant memories across all stores for a given context."""
        query_terms = self._extract_query_terms(context)

        semantic = []
        for term in query_terms[:3]:
            semantic.extend(self.search_semantic(term, limit=limit // len(query_terms) + 1))

        episodic = self.retrieve_episodes(limit=limit)

        working_keys = [f"context_{k}" for k in list(context.keys())[:5]]
        working = {k: self.retrieve_working(k) for k in working_keys if self.retrieve_working(k) is not None}

        return {
            "semantic": semantic[:limit],
            "episodic": episodic[:limit],
            "working": working,
        }

    @staticmethod
    def _extract_query_terms(context: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for k, v in context.items():
            if isinstance(v, str) and len(v) > 3:
                terms.append(v[:50])
            elif isinstance(k, str):
                terms.append(k)
        return terms[:5] if terms else ["decision"]
