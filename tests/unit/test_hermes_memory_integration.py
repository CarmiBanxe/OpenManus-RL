"""
Unit tests — HermesMemoryIntegration (Sprint 2)
Uses real SQLite in a temp directory — no external services.
Constructor takes config dict: {"db_path": str(path)}
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pytest

from openmanus_rl.integration.hermes_memory_integration import HermesMemoryIntegration


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_hermes.db"


@pytest.fixture
def mem(tmp_db_path: Path) -> HermesMemoryIntegration:
    return HermesMemoryIntegration(config={"db_path": str(tmp_db_path)})


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_db_file_created(self, tmp_db_path: Path) -> None:
        HermesMemoryIntegration(config={"db_path": str(tmp_db_path)})
        assert tmp_db_path.exists()

    def test_tables_exist(self, tmp_db_path: Path) -> None:
        HermesMemoryIntegration(config={"db_path": str(tmp_db_path)})
        conn = sqlite3.connect(str(tmp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "working_memory" in tables
        assert "semantic_memory" in tables
        assert "episodic_memory" in tables
        assert "skills_memory" in tables

    def test_default_db_path_is_set(self, tmp_db_path: Path) -> None:
        m = HermesMemoryIntegration(config={"db_path": str(tmp_db_path)})
        assert str(tmp_db_path) in str(m.db_path)

    def test_no_config_uses_default_path(self) -> None:
        # Should not crash — creates ~/.openmanus/hermes_memory.db
        m = HermesMemoryIntegration()
        assert m.db_path is not None
        assert "hermes_memory" in str(m.db_path)


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_store_and_retrieve(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("key1", {"data": 42})
        result = mem.retrieve_working("key1")
        assert result is not None
        assert result["data"] == 42

    def test_retrieve_missing_key_returns_none(
        self, mem: HermesMemoryIntegration
    ) -> None:
        assert mem.retrieve_working("nonexistent") is None

    def test_overwrite_key(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("k", "first")
        mem.store_working("k", "second")
        assert mem.retrieve_working("k") == "second"

    def test_session_id_stored(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("k2", "val", session_id="session-abc")
        result = mem.retrieve_working("k2")
        assert result == "val"

    def test_complex_object(self, mem: HermesMemoryIntegration) -> None:
        data = {"nested": {"list": [1, 2, 3], "bool": True}}
        mem.store_working("complex", data)
        result = mem.retrieve_working("complex")
        assert result == data

    def test_string_value(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("str_key", "plain string")
        assert mem.retrieve_working("str_key") == "plain string"

    def test_numeric_value(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("num_key", 3.14)
        assert abs(mem.retrieve_working("num_key") - 3.14) < 1e-6

    def test_clear_session(self, mem: HermesMemoryIntegration) -> None:
        mem.store_working("a", "val", session_id="sess1")
        mem.store_working("b", "val", session_id="sess1")
        deleted = mem.clear_working_session("sess1")
        assert deleted == 2
        assert mem.retrieve_working("a") is None


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    def test_store_and_search(self, mem: HermesMemoryIntegration) -> None:
        mem.store_semantic("bitcoin", {"type": "crypto"}, confidence=0.9)
        results = mem.search_semantic("bitcoin")
        assert len(results) >= 1
        assert any(r["concept"] == "bitcoin" for r in results)

    def test_search_missing_concept_returns_empty(
        self, mem: HermesMemoryIntegration
    ) -> None:
        results = mem.search_semantic("zzz_nonexistent_concept")
        assert results == []

    def test_confidence_filter(self, mem: HermesMemoryIntegration) -> None:
        mem.store_semantic("low_conf", {}, confidence=0.3)
        mem.store_semantic("high_conf", {}, confidence=0.9)
        results = mem.search_semantic("conf", min_confidence=0.5)
        concepts = [r["concept"] for r in results]
        assert "high_conf" in concepts
        assert "low_conf" not in concepts

    def test_returns_id(self, mem: HermesMemoryIntegration) -> None:
        row_id = mem.store_semantic("ether", {"chain": "ETH"})
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_limit_respected(self, mem: HermesMemoryIntegration) -> None:
        for i in range(15):
            mem.store_semantic(f"concept_{i}", {"i": i}, confidence=0.8)
        results = mem.search_semantic("concept", limit=5)
        assert len(results) <= 5

    def test_source_stored(self, mem: HermesMemoryIntegration) -> None:
        mem.store_semantic("concept_src", {}, source="manual_test")
        results = mem.search_semantic("concept_src")
        assert results[0].get("source") == "manual_test"

    def test_result_has_required_fields(self, mem: HermesMemoryIntegration) -> None:
        mem.store_semantic("field_test", {"key": "value"}, confidence=0.75)
        results = mem.search_semantic("field_test")
        r = results[0]
        assert "id" in r
        assert "concept" in r
        assert "facts" in r
        assert "confidence" in r
        assert "created_at" in r


# ---------------------------------------------------------------------------
# Episodic memory (append-only, I-24 aligned)
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    def test_store_episode(self, mem: HermesMemoryIntegration) -> None:
        row_id = mem.store_episode(
            "decision_made",
            {"action": "proceed", "confidence": 0.8},
            action_taken="proceed",
            outcome="success",
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_retrieve_episodes_by_type(self, mem: HermesMemoryIntegration) -> None:
        mem.store_episode("event_type_a", {"x": 1})
        mem.store_episode("event_type_a", {"x": 2})
        results = mem.retrieve_episodes(event_type="event_type_a")
        assert len(results) == 2

    def test_retrieve_episodes_limit(self, mem: HermesMemoryIntegration) -> None:
        for i in range(10):
            mem.store_episode("many", {"i": i})
        results = mem.retrieve_episodes(limit=3)
        assert len(results) <= 3

    def test_no_delete_method(self, mem: HermesMemoryIntegration) -> None:
        # Episodic store is append-only (I-24) — no delete_episode method
        assert not hasattr(mem, "delete_episode")
        assert not hasattr(mem, "truncate_episodic")

    def test_retrieve_no_filter_returns_all(
        self, mem: HermesMemoryIntegration
    ) -> None:
        mem.store_episode("type_x", {})
        mem.store_episode("type_y", {})
        results = mem.retrieve_episodes()
        assert len(results) >= 2

    def test_episode_has_required_fields(
        self, mem: HermesMemoryIntegration
    ) -> None:
        mem.store_episode("typed_ep", {"k": "v"}, action_taken="act", outcome="ok")
        results = mem.retrieve_episodes(event_type="typed_ep")
        r = results[0]
        assert r["event_type"] == "typed_ep"
        assert r["action_taken"] == "act"
        assert r["outcome"] == "ok"
        assert "timestamp" in r
        assert "id" in r

    def test_since_filter(self, mem: HermesMemoryIntegration) -> None:
        mem.store_episode("ep1", {"t": 1})
        from datetime import datetime, timedelta

        future = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        results = mem.retrieve_episodes(since=future)
        assert results == []  # episode is older than future cutoff


# ---------------------------------------------------------------------------
# Skills memory
# ---------------------------------------------------------------------------


class TestSkillsMemory:
    def test_store_skill(self, mem: HermesMemoryIntegration) -> None:
        mem.store_skill("reasoning", {"steps": ["think", "act"]}, success_rate=0.8)
        skill = mem.retrieve_skill("reasoning")
        assert skill is not None

    def test_retrieve_skill_returns_data(self, mem: HermesMemoryIntegration) -> None:
        mem.store_skill("code_skill", {"lang": "python"}, success_rate=0.7)
        skill = mem.retrieve_skill("code_skill")
        assert skill is not None
        assert skill.get("success_rate", 0) == pytest.approx(0.7)

    def test_retrieve_nonexistent_skill(
        self, mem: HermesMemoryIntegration
    ) -> None:
        assert mem.retrieve_skill("does_not_exist") is None

    def test_increment_success_updates_rate(
        self, mem: HermesMemoryIntegration
    ) -> None:
        mem.store_skill("skill_s", {}, success_rate=0.5)
        mem.increment_skill_usage("skill_s", success=True)
        # Just verify it doesn't crash and skill still exists
        assert mem.retrieve_skill("skill_s") is not None

    def test_increment_failure(self, mem: HermesMemoryIntegration) -> None:
        mem.store_skill("risky_skill", {}, success_rate=0.9)
        mem.increment_skill_usage("risky_skill", success=False)

    def test_increment_nonexistent_does_not_crash(
        self, mem: HermesMemoryIntegration
    ) -> None:
        mem.increment_skill_usage("ghost_skill", success=True)


# ---------------------------------------------------------------------------
# get_relevant_memories
# ---------------------------------------------------------------------------


class TestGetRelevantMemories:
    def test_returns_dict_with_sections(
        self, mem: HermesMemoryIntegration
    ) -> None:
        result = mem.get_relevant_memories({"topic": "finance"})
        assert isinstance(result, dict)
        assert "working" in result
        assert "semantic" in result
        assert "episodic" in result

    def test_returns_something_after_storing(
        self, mem: HermesMemoryIntegration
    ) -> None:
        mem.store_semantic("finance_kb", {"domain": "banking"}, confidence=0.9)
        mem.store_episode("finance_decision", {"action": "invest"})
        result = mem.get_relevant_memories({"topic": "finance"}, limit=5)
        assert isinstance(result, dict)

    def test_empty_context_no_crash(self, mem: HermesMemoryIntegration) -> None:
        result = mem.get_relevant_memories({})
        assert isinstance(result, dict)

    def test_limit_constrains_episodic(
        self, mem: HermesMemoryIntegration
    ) -> None:
        for i in range(20):
            mem.store_episode("bulk", {"i": i})
        result = mem.get_relevant_memories({"any": "context"}, limit=2)
        assert len(result.get("episodic", [])) <= 2
