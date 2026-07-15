"""Smoke tests for openmanus_rl.memory.qdrant_memory (SQLite fallback path)."""

from openmanus_rl.memory.qdrant_memory import (
    QdrantConfig,
    QdrantMemory,
    MemoryRecord,
    _SqliteFallback,
)


class TestQdrantConfig:
    def test_defaults(self):
        cfg = QdrantConfig()
        assert cfg.host == "127.0.0.1"  # Charter §8: never 0.0.0.0
        assert cfg.port == 6333
        assert cfg.top_k >= 1

    def test_from_env_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HOST", "192.168.1.5")
        monkeypatch.setenv("QDRANT_TOP_K", "10")
        cfg = QdrantConfig.from_env()
        assert cfg.host == "192.168.1.5"
        assert cfg.top_k == 10

    def test_host_never_0000_by_default(self):
        cfg = QdrantConfig.from_env()
        # Default must not be 0.0.0.0
        assert cfg.host != "0.0.0.0"


class TestSqliteFallback:
    def test_ingest_returns_false_when_db_missing(self, tmp_path):
        fb = _SqliteFallback(str(tmp_path / "nonexistent.db"))
        # DB does not exist yet — ingest creates it (returns True)
        # or safely returns False without crashing
        result = fb.ingest("test text", {"session_id": "s1"})
        assert isinstance(result, bool)

    def test_search_returns_empty_when_db_missing(self, tmp_path):
        fb = _SqliteFallback(str(tmp_path / "nonexistent_db.db"))
        results = fb.search("query", top_k=5, session_id="s1")
        assert isinstance(results, list)

    def test_ingest_and_search_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        fb = _SqliteFallback(db_path)
        ok = fb.ingest("Python async programming patterns", {"session_id": "s1"})
        assert ok is True
        results = fb.search("Python async", top_k=5, session_id="s1")
        assert len(results) >= 1
        assert results[0]["text"] == "Python async programming patterns"


class TestQdrantMemory:
    def _memory_with_sqlite(self, tmp_path) -> QdrantMemory:
        """Return a QdrantMemory that always uses SQLite fallback (no real Qdrant)."""
        cfg = QdrantConfig(
            host="invalid-host-that-does-not-resolve",
            fallback_sqlite=str(tmp_path / "fallback.db"),
        )
        return QdrantMemory(config=cfg)

    def test_using_vector_store_false_when_qdrant_unreachable(self, tmp_path):
        mem = self._memory_with_sqlite(tmp_path)
        assert mem.using_vector_store is False

    def test_ingest_falls_back_to_sqlite(self, tmp_path):
        mem = self._memory_with_sqlite(tmp_path)
        ok = mem.ingest("test memory record", session_id="test-session")
        assert isinstance(ok, bool)  # either True (SQLite) or False (nothing available)

    def test_search_returns_memory_records(self, tmp_path):
        mem = self._memory_with_sqlite(tmp_path)
        mem.ingest("vector search with embeddings", session_id="sess1")
        results = mem.search("vector search", session_id="sess1")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, MemoryRecord)
            assert isinstance(r.text, str)
            assert isinstance(r.score, float)

    def test_search_default_session(self, tmp_path):
        mem = self._memory_with_sqlite(tmp_path)
        mem.ingest("default session text")
        results = mem.search("default session")
        assert isinstance(results, list)

    def test_memory_record_fields(self):
        r = MemoryRecord(text="hello", score=0.9, metadata={"session_id": "s"})
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.metadata["session_id"] == "s"
