"""
Unit tests — EnhancedDecisionAgent (Sprint 2)
Mocks: agent.memory, agent._remizov_engine, agent.osint_agent.select_action_async.

Actual attributes:
  agent.memory        — HermesMemoryIntegration
  agent._remizov_engine — DecisionEngineWithRemizov (optional)
  agent.osint_agent   — LegionOSINTEnhancedAgent
  agent.base_agent    — SmartDecisionAgent (optional, sync)

Result keys from select_action:
  action, confidence, explanation, source (from _make_decision)
  episode_id, timestamp, osint_enhanced, remizov_used (added by select_action)
"""
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _base_agent(action: str = "base_action") -> MagicMock:
    agent = MagicMock()
    agent.select_action = MagicMock(return_value=action)
    return agent


def _memory_mock() -> MagicMock:
    mem = MagicMock()
    mem.get_relevant_memories = MagicMock(
        return_value={"working": [], "semantic": [], "episodic": []}
    )
    mem.store_episode = MagicMock(return_value=42)
    mem.store_skill = MagicMock()
    mem.store_working = MagicMock()
    mem.retrieve_episodes = MagicMock(return_value=[])
    mem.increment_skill_usage = MagicMock()
    return mem


def _osint_mock(
    action: str = "osint_action", osint_enhanced: bool = True
) -> MagicMock:
    osint = MagicMock()
    osint.select_action_async = AsyncMock(
        return_value={
            "action": action,
            "osint_enhanced": osint_enhanced,
            "risk_factors": [],
            "confidence_score": 0.5,
            "context": {},
        }
    )
    osint.set_base_agent = MagicMock()
    return osint


def _remizov_mock(
    action: str = "remizov_action", confidence: float = 0.85
) -> MagicMock:
    engine = MagicMock()
    engine.make_enhanced_decision = AsyncMock(
        return_value={
            "action": action,
            "confidence": confidence,
            "explanation": "Remizov ODE recommendation",
            "transition_probabilities": {action: 1.0},
            "risk_assessment": {"risk_level": 0.2, "is_acceptable": True},
            "analytical": True,
        }
    )
    return engine


def _make_agent(
    base: Optional[MagicMock] = None,
    tmp_path: Optional[Path] = None,
) -> EnhancedDecisionAgent:
    """Build EnhancedDecisionAgent with mocked internals to avoid real I/O."""
    cfg: Dict[str, Any] = {}
    if tmp_path:
        cfg["memory"] = {"db_path": str(tmp_path / "test.db")}
    agent = EnhancedDecisionAgent(base_agent=base, config=cfg)
    agent.memory = _memory_mock()
    agent.osint_agent = _osint_mock()
    agent._remizov_engine = _remizov_mock()
    return agent


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEnhancedDecisionAgentInit:
    def test_default_construction(self, tmp_path: Path) -> None:
        agent = EnhancedDecisionAgent(
            config={"memory": {"db_path": str(tmp_path / "mem.db")}}
        )
        assert agent.config is not None
        assert agent.base_agent is None

    def test_with_base_agent(self, tmp_path: Path) -> None:
        base = _base_agent()
        agent = EnhancedDecisionAgent(
            base_agent=base,
            config={"memory": {"db_path": str(tmp_path / "mem.db")}},
        )
        assert agent.base_agent is base

    def test_memory_attribute_exists(self, tmp_path: Path) -> None:
        agent = EnhancedDecisionAgent(
            config={"memory": {"db_path": str(tmp_path / "mem.db")}}
        )
        assert agent.memory is not None

    def test_osint_agent_attribute_exists(self, tmp_path: Path) -> None:
        agent = EnhancedDecisionAgent(
            config={"memory": {"db_path": str(tmp_path / "mem.db")}}
        )
        assert agent.osint_agent is not None


# ---------------------------------------------------------------------------
# select_action — full pipeline result shape
# ---------------------------------------------------------------------------


class TestSelectAction:
    @pytest.mark.asyncio
    async def test_returns_action_key(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action(
            {"confidence": 0.5}, ["do_something", "wait", "analyze"]
        )
        assert "action" in result

    @pytest.mark.asyncio
    async def test_returns_confidence(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action({}, ["a", "b"])
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_returns_explanation(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action({}, ["a", "b"])
        assert isinstance(result.get("explanation"), str)

    @pytest.mark.asyncio
    async def test_has_episode_id(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        agent.memory.store_episode = MagicMock(return_value=77)
        result = await agent.select_action({}, ["x"])
        assert result.get("episode_id") == 77

    @pytest.mark.asyncio
    async def test_has_timestamp(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action({}, ["x"])
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)

    @pytest.mark.asyncio
    async def test_has_osint_enhanced_flag(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action({}, ["a"])
        assert "osint_enhanced" in result

    @pytest.mark.asyncio
    async def test_has_remizov_used_flag(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        result = await agent.select_action({}, ["a"])
        assert "remizov_used" in result

    @pytest.mark.asyncio
    async def test_episode_stored_in_memory(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        await agent.select_action({"x": 1}, ["go", "stop"])
        agent.memory.store_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_memories_retrieved(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        await agent.select_action({}, ["a"])
        agent.memory.get_relevant_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_osint_called(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        await agent.select_action({}, ["a"])
        agent.osint_agent.select_action_async.assert_called_once()


# ---------------------------------------------------------------------------
# _make_decision — path coverage
# ---------------------------------------------------------------------------


class TestMakeDecision:
    @pytest.mark.asyncio
    async def test_remizov_path_when_high_confidence(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(base=_base_agent("base"), tmp_path=tmp_path)
        remizov_result = {
            "action": "optimal_action",
            "confidence": 0.95,
            "explanation": "ODE solved",
            "analytical": True,
        }
        result = await agent._make_decision(
            {"ctx": True}, ["optimal_action", "other"], remizov_result
        )
        assert result["source"] == "remizov"
        assert result["action"] == "optimal_action"

    @pytest.mark.asyncio
    async def test_base_agent_path_when_remizov_low_confidence(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(base=_base_agent("base_choice"), tmp_path=tmp_path)
        remizov_result = {
            "action": "remizov_choice",
            "confidence": 0.3,
            "analytical": False,
        }
        result = await agent._make_decision(
            {}, ["base_choice", "other"], remizov_result
        )
        assert result["source"] in ("base_agent", "fallback")

    @pytest.mark.asyncio
    async def test_base_agent_sync_str_returned(
        self, tmp_path: Path
    ) -> None:
        base = MagicMock()
        base.select_action = MagicMock(return_value="sync_string_action")
        agent = _make_agent(base=base, tmp_path=tmp_path)
        result = await agent._make_decision(
            {}, ["sync_string_action"], {"confidence": 0.0, "analytical": False}
        )
        assert result["action"] == "sync_string_action"
        assert result["source"] == "base_agent"

    @pytest.mark.asyncio
    async def test_base_agent_dict_returned(self, tmp_path: Path) -> None:
        base = MagicMock()
        base.select_action = MagicMock(
            return_value={"action": "dict_action", "confidence": 0.88}
        )
        agent = _make_agent(base=base, tmp_path=tmp_path)
        result = await agent._make_decision(
            {}, ["dict_action"], {"confidence": 0.0, "analytical": False}
        )
        assert result["action"] == "dict_action"
        assert result["source"] == "base_agent"

    @pytest.mark.asyncio
    async def test_fallback_when_no_base_agent(self, tmp_path: Path) -> None:
        agent = _make_agent(base=None, tmp_path=tmp_path)
        agent.base_agent = None  # ensure None
        result = await agent._make_decision(
            {}, ["fallback_action"], {"confidence": 0.0, "analytical": False}
        )
        assert result["action"] == "fallback_action"
        assert result["source"] == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_default_when_empty_actions(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(base=None, tmp_path=tmp_path)
        agent.base_agent = None
        result = await agent._make_decision(
            {}, [], {"confidence": 0.0, "analytical": False}
        )
        assert result["action"] == "default"
        assert result["source"] == "fallback"

    @pytest.mark.asyncio
    async def test_base_agent_exception_falls_through(
        self, tmp_path: Path
    ) -> None:
        base = MagicMock()
        base.select_action = MagicMock(side_effect=RuntimeError("crash"))
        agent = _make_agent(base=base, tmp_path=tmp_path)
        result = await agent._make_decision(
            {}, ["safe_action"], {"confidence": 0.0, "analytical": False}
        )
        # Should fall through to fallback, not raise
        assert result["source"] == "fallback"


# ---------------------------------------------------------------------------
# _context_to_features
# ---------------------------------------------------------------------------


class TestContextToFeatures:
    def test_numeric_values_extracted(self, tmp_path: Path) -> None:
        import numpy as np

        agent = _make_agent(tmp_path=tmp_path)
        state = {"a": 1.0, "b": 2.0, "c": 0.5}
        vec = agent._context_to_features(state)
        assert len(vec) >= 3
        assert 1.0 in vec or 2.0 in vec

    def test_empty_state_default_vector(self, tmp_path: Path) -> None:
        import numpy as np

        agent = _make_agent(tmp_path=tmp_path)
        vec = agent._context_to_features({})
        assert len(vec) == 1
        assert vec[0] == pytest.approx(0.5)

    def test_booleans_converted(self, tmp_path: Path) -> None:
        import numpy as np

        agent = _make_agent(tmp_path=tmp_path)
        vec = agent._context_to_features({"flag": True, "off": False})
        assert 1.0 in vec or 0.0 in vec

    def test_strings_ignored(self, tmp_path: Path) -> None:
        import numpy as np

        agent = _make_agent(tmp_path=tmp_path)
        vec = agent._context_to_features({"text": "ignored", "num": 0.3})
        assert len(vec) >= 1  # "num" extracted, "text" ignored


# ---------------------------------------------------------------------------
# update_from_feedback
# ---------------------------------------------------------------------------


class TestUpdateFromFeedback:
    @pytest.mark.asyncio
    async def test_success_feedback_calls_store_skill(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        agent.memory.retrieve_episodes = MagicMock(
            return_value=[{"id": 42, "action_taken": "proceed"}]
        )
        await agent.update_from_feedback(
            episode_id=42, outcome="success", success=True
        )
        agent.memory.store_skill.assert_called()

    @pytest.mark.asyncio
    async def test_failure_feedback_calls_increment(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        agent.memory.retrieve_episodes = MagicMock(
            return_value=[{"id": 99, "action_taken": "risky"}]
        )
        await agent.update_from_feedback(
            episode_id=99, outcome="failure", success=False
        )
        agent.memory.increment_skill_usage.assert_called()

    @pytest.mark.asyncio
    async def test_nonexistent_episode_no_crash(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        agent.memory.retrieve_episodes = MagicMock(return_value=[])
        await agent.update_from_feedback(
            episode_id=99999, outcome="unknown", success=False
        )
        # retrieve_episodes called, store_skill NOT called (no matching episode)
        agent.memory.store_skill.assert_not_called()


# ---------------------------------------------------------------------------
# remizov_used flag in result
# ---------------------------------------------------------------------------


class TestRemizovUsedFlag:
    @pytest.mark.asyncio
    async def test_remizov_used_true_when_analytical(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path=tmp_path)
        agent._remizov_engine = _remizov_mock("best", confidence=0.9)
        result = await agent.select_action({}, ["best", "other"])
        assert result["remizov_used"] is True

    @pytest.mark.asyncio
    async def test_remizov_used_false_when_engine_raises(
        self, tmp_path: Path
    ) -> None:
        # If make_enhanced_decision throws, _remizov_decide returns analytical=False
        agent = _make_agent(tmp_path=tmp_path)
        engine = MagicMock()
        engine.make_enhanced_decision = AsyncMock(side_effect=RuntimeError("engine unavailable"))
        agent._remizov_engine = engine
        result = await agent.select_action({}, ["a", "b"])
        assert result["remizov_used"] is False
