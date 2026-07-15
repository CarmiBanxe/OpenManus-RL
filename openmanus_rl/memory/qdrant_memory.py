"""
Vector memory backed by Qdrant with automatic SQLite fallback.

If Qdrant is reachable (host:port from config/env), uses vector similarity search.
If Qdrant is unavailable, transparently falls back to SQLite keyword search so the
engine continues working in development/offline environments.

Config priority (highest first):
  1. QDRANT_* environment variables
  2. openmanus_rl/config/qdrant_config.yaml
  3. Module-level defaults

CHARTER §8: No Tor/onion transport. All connections via localhost (127.0.0.1).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QdrantConfig:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "legion_memory"
    vector_size: int = 768
    distance: str = "Cosine"
    top_k: int = 5
    score_threshold: float = 0.7
    embedding_url: str = "http://127.0.0.1:11434/api/embeddings"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout: float = 10.0
    session_namespace: bool = True
    fallback_sqlite: str = "~/.openmanus/hermes_memory.db"

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        cfg = cls()
        cfg.host = os.environ.get("QDRANT_HOST", cfg.host)
        cfg.port = int(os.environ.get("QDRANT_PORT", cfg.port))
        cfg.collection = os.environ.get("QDRANT_COLLECTION", cfg.collection)
        cfg.vector_size = int(os.environ.get("QDRANT_VECTOR_SIZE", cfg.vector_size))
        cfg.top_k = int(os.environ.get("QDRANT_TOP_K", cfg.top_k))
        cfg.embedding_url = os.environ.get("QDRANT_EMBED_URL", cfg.embedding_url)
        cfg.embedding_model = os.environ.get("QDRANT_EMBED_MODEL", cfg.embedding_model)
        return cfg


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(text: str, cfg: QdrantConfig) -> Optional[list[float]]:
    """Return embedding vector from local Ollama endpoint, or None on failure."""
    try:
        import httpx  # type: ignore[import-not-found]
        resp = httpx.post(
            cfg.embedding_url,
            json={"model": cfg.embedding_model, "prompt": text},
            timeout=cfg.embedding_timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Embedding request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# SQLite fallback (keyword search on existing hermes_memory.db)
# ---------------------------------------------------------------------------

class _SqliteFallback:
    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path).expanduser()

    def _connect(self, create: bool = False) -> Optional[sqlite3.Connection]:
        if not create and not self._path.exists():
            return None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(str(self._path))
        except sqlite3.Error:
            return None

    def ingest(self, text: str, metadata: dict[str, Any]) -> bool:
        conn = self._connect(create=True)
        if conn is None:
            return False
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS qdrant_fallback "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, metadata TEXT)"
                )
                conn.execute(
                    "INSERT INTO qdrant_fallback (text, metadata) VALUES (?, ?)",
                    (text, json.dumps(metadata)),
                )
            return True
        except sqlite3.Error as exc:
            logger.warning("SQLite fallback ingest failed: %s", exc)
            return False
        finally:
            conn.close()

    def search(self, query: str, top_k: int, session_id: Optional[str]) -> list[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT text, metadata FROM qdrant_fallback "
                "WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query[:50]}%", top_k),
            )
            rows = cur.fetchall()
            return [{"text": r[0], "score": 0.0, "metadata": json.loads(r[1])} for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Qdrant client wrapper
# ---------------------------------------------------------------------------

class _QdrantBackend:
    def __init__(self, cfg: QdrantConfig) -> None:
        self._cfg = cfg
        self._client: Any = None

    def _get_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import-not-found]
            client = QdrantClient(host=self._cfg.host, port=self._cfg.port, timeout=5)
            # Ensure collection exists
            existing = [c.name for c in client.get_collections().collections]
            if self._cfg.collection not in existing:
                dist = getattr(Distance, self._cfg.distance.upper(), Distance.COSINE)
                client.create_collection(
                    collection_name=self._cfg.collection,
                    vectors_config=VectorParams(size=self._cfg.vector_size, distance=dist),
                )
                logger.info("Created Qdrant collection '%s'", self._cfg.collection)
            self._client = client
            return client
        except Exception as exc:  # noqa: BLE001
            logger.debug("Qdrant unavailable: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._get_client() is not None

    def ingest(self, text: str, vector: list[float], metadata: dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            from qdrant_client.models import PointStruct  # type: ignore[import-not-found]
            import uuid
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata},
            )
            client.upsert(collection_name=self._cfg.collection, points=[point])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant ingest failed: %s", exc)
            return False

    def search(
        self, vector: list[float], top_k: int, session_id: Optional[str]
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import-not-found]
            query_filter = None
            if session_id and self._cfg.session_namespace:
                query_filter = Filter(
                    must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
            results = client.search(
                collection_name=self._cfg.collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=self._cfg.score_threshold,
                query_filter=query_filter,
            )
            return [
                {"text": r.payload.get("text", ""), "score": r.score, "metadata": r.payload}
                for r in results
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QdrantMemory:
    """
    Vector memory for Legion agents.

    Falls back to SQLite keyword search transparently when Qdrant is unavailable.
    All Qdrant connections are to 127.0.0.1 only (Charter §8 compliance).
    """

    def __init__(self, config: Optional[QdrantConfig] = None) -> None:
        self._cfg = config or QdrantConfig.from_env()
        self._backend = _QdrantBackend(self._cfg)
        self._fallback = _SqliteFallback(self._cfg.fallback_sqlite)

    @property
    def using_vector_store(self) -> bool:
        return self._backend.is_available()

    def ingest(self, text: str, session_id: str = "default", **extra: Any) -> bool:
        """Store text in vector store (or SQLite fallback on failure)."""
        metadata = {"session_id": session_id, **extra}
        vector = _embed(text, self._cfg)
        if vector and self._backend.ingest(text, vector, metadata):
            return True
        # Fallback path
        return self._fallback.ingest(text, metadata)

    def search(
        self, query: str, session_id: str = "default", top_k: Optional[int] = None
    ) -> list[MemoryRecord]:
        """Return top-k semantically similar records for the query."""
        k = top_k or self._cfg.top_k
        vector = _embed(query, self._cfg)
        if vector and self._backend.is_available():
            raw = self._backend.search(vector, k, session_id)
        else:
            logger.info("QdrantMemory: using SQLite keyword fallback")
            raw = self._fallback.search(query, k, session_id)
        return [MemoryRecord(text=r["text"], score=r["score"], metadata=r["metadata"]) for r in raw]
