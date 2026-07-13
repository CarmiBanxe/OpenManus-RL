"""
Enhanced Decision Agent — Sprint 2 + Sprint 3 + Sprint 4.
Объединяет: MultimodalContextManager, VoicePipelineIntegration,
LegionOSINTEnhancedAgent, HermesMemoryIntegration, DecisionEngineWithRemizov,
Qwen3OmniIntegration, DeepHedgingFramework, SignatureMethods,
MeanFieldGames, PerformanceOptimizer.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from openmanus_rl.integration.hermes_memory_integration import HermesMemoryIntegration
from openmanus_rl.integration.legion_osint_integration import LegionOSINTEnhancedAgent
from openmanus_rl.integration.multimodal_context_manager import (
    InputModality,
    MultimodalContextManager,
)
from openmanus_rl.integration.voice_pipeline_integration import VoicePipelineIntegration

# Sprint 3 — optional deps (torch / iisignature may not be installed in all envs)
try:
    from openmanus_rl.decision.deep_hedging import DeepHedgingFramework
    from openmanus_rl.decision.signature_methods import SignatureMethods
    from openmanus_rl.integration.qwen3_omni_integration import Qwen3OmniIntegration

    _SPRINT3_AVAILABLE = True
except Exception as _sprint3_import_err:  # noqa: BLE001
    _SPRINT3_AVAILABLE = False
    _sprint3_import_err_msg = str(_sprint3_import_err)

# Sprint 4 — MFG + PerformanceOptimizer (numpy-only, always available)
try:
    from openmanus_rl.decision.mean_field_games import MeanFieldGames
    from openmanus_rl.optimization.performance_optimizer import PerformanceOptimizer

    _SPRINT4_AVAILABLE = True
except Exception as _sprint4_import_err:  # noqa: BLE001
    _SPRINT4_AVAILABLE = False
    _sprint4_import_err_msg = str(_sprint4_import_err)

logger = logging.getLogger(__name__)

if not _SPRINT3_AVAILABLE:
    logger.warning("Sprint 3 components unavailable: %s", _sprint3_import_err_msg)

if not _SPRINT4_AVAILABLE:
    logger.warning("Sprint 4 components unavailable: %s", _sprint4_import_err_msg)


class EnhancedDecisionAgent:
    """
    Sprint 2 + Sprint 3 + Sprint 4 — Улучшенный агент принятия решений.
    Интегрирует мультимодальный контекст, OSINT, память Hermes, Remizov solver,
    Qwen3-Omni, Deep Hedging, Signature Methods, Mean Field Games и PerformanceOptimizer.
    """

    def __init__(
        self,
        base_agent: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}
        self.base_agent = base_agent

        # Sprint 2 components
        self.context_manager = MultimodalContextManager(
            self.config.get("context_manager", {})
        )
        self.voice_pipeline = VoicePipelineIntegration(
            self.config.get("voice_pipeline", {})
        )
        self.memory = HermesMemoryIntegration(self.config.get("memory", {}))
        self.osint_agent = LegionOSINTEnhancedAgent(self.config.get("osint", {}))

        if base_agent is not None:
            self.osint_agent.set_base_agent(base_agent)

        # Remizov engine — optional, import lazily to handle missing scipy
        self._remizov_engine: Optional[Any] = None

        # Sprint 3 feature flags
        self.enable_qwen3_omni: bool = self.config.get("enable_qwen3_omni", True)
        self.enable_deep_hedging: bool = self.config.get("enable_deep_hedging", True)
        self.enable_signature_methods: bool = self.config.get(
            "enable_signature_methods", True
        )

        # Sprint 3 components (None when _SPRINT3_AVAILABLE is False)
        self.qwen3_omni: Optional[Any] = None
        self.deep_hedging: Optional[Any] = None
        self.signature_methods: Optional[Any] = None

        if _SPRINT3_AVAILABLE:
            try:
                self.qwen3_omni = Qwen3OmniIntegration(
                    self.config.get("qwen3_omni", {})
                )
                self.deep_hedging = DeepHedgingFramework(
                    self.config.get("deep_hedging", {})
                )
                self.signature_methods = SignatureMethods(
                    self.config.get("signature_methods", {})
                )
            except Exception as exc:
                logger.warning("Sprint 3 component init failed: %s", exc)

        # Sprint 4 feature flags
        self.enable_mean_field_games: bool = self.config.get("enable_mean_field_games", True)
        self.enable_performance_optimization: bool = self.config.get(
            "enable_performance_optimization", True
        )

        # Sprint 4 components
        self.mean_field_games: Optional[Any] = None
        self.performance_optimizer: Optional[Any] = None

        if _SPRINT4_AVAILABLE:
            try:
                if self.enable_mean_field_games:
                    self.mean_field_games = MeanFieldGames(
                        self.config.get("mean_field_games", {})
                    )
                if self.enable_performance_optimization:
                    self.performance_optimizer = PerformanceOptimizer(
                        self.config.get("performance_optimizer", {})
                    )
                    self.performance_optimizer.start_monitoring(interval=5.0)
            except Exception as exc:
                logger.warning("Sprint 4 component init failed: %s", exc)

        logger.info(
            "EnhancedDecisionAgent initialized (sprint3=%s, sprint4=%s)",
            _SPRINT3_AVAILABLE,
            _SPRINT4_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def select_action(
        self,
        state: Dict[str, Any],
        available_actions: List[str],
        voice_input: Optional[bytes] = None,
        priority: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Полный цикл: голос → контекст → OSINT → память → Remizov → решение.
        """
        start_ts = __import__("datetime").datetime.utcnow().isoformat()

        # Step 1: build multimodal context
        unified_context = await self._build_context(state, voice_input)

        # Step 2: retrieve relevant memories
        memories = self.memory.get_relevant_memories(unified_context, limit=5)
        unified_context["memories"] = memories

        # Step 3: OSINT enrichment
        available_actions_dicts = [{"action": a} for a in available_actions]
        osint_result = await self.osint_agent.select_action_async(
            unified_context, available_actions_dicts, priority=priority
        )
        for k, v in osint_result.get("context", {}).items():
            unified_context.setdefault(k, v)
        unified_context["osint_risk_factors"] = osint_result.get("risk_factors", [])
        unified_context["osint_confidence"] = osint_result.get("confidence_score", 0.0)

        # Step 4: Remizov analytical decision
        remizov_result = await self._remizov_decide(unified_context, available_actions)

        # Step 5: base agent fallback / confirmation
        final_decision = await self._make_decision(
            unified_context, available_actions, remizov_result
        )

        # Step 6: persist episode
        episode_id = self.memory.store_episode(
            event_type="decision",
            event_data={
                "state_keys": list(state.keys()),
                "available_actions": available_actions,
                "priority": priority,
            },
            action_taken=final_decision.get("action"),
            outcome=None,
        )

        return {
            **final_decision,
            "episode_id": episode_id,
            "timestamp": start_ts,
            "osint_enhanced": osint_result.get("osint_enhanced", False),
            "remizov_used": remizov_result.get("analytical", False),
        }

    # ------------------------------------------------------------------
    # Sprint 3 — advanced voice processing
    # ------------------------------------------------------------------

    async def process_voice_input_advanced(
        self,
        audio_data: Union[bytes, np.ndarray],
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Продвинутая обработка голосового ввода с Qwen3-Omni."""
        if not self.enable_qwen3_omni or self.qwen3_omni is None:
            return await self.voice_pipeline.process_voice_command(
                audio_data, context=base_context or {}
            )

        try:
            result: Dict[str, Any] = await self.qwen3_omni.process_voice_input(
                audio_data, base_context
            )

            if "error" not in result:
                if self.enable_deep_hedging and self.deep_hedging is not None:
                    result["risk_analysis"] = self._analyze_risk_for_voice_input(
                        result, base_context
                    )
                if (
                    self.enable_signature_methods
                    and self.signature_methods is not None
                    and "text_response" in result
                ):
                    result["signature_analysis"] = (
                        self._analyze_signature_for_voice_input(result, base_context)
                    )

            return result

        except Exception as exc:
            logger.error("Advanced voice input processing error: %s", exc)
            return await self.voice_pipeline.process_voice_command(
                audio_data, context=base_context or {}
            )

    def _analyze_risk_for_voice_input(
        self,
        voice_result: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Анализ риска для голосового ввода по ключевым словам."""
        try:
            text = voice_result.get("text_response", "").lower()
            risk_keywords = ["risk", "loss", "danger", "warning", "caution"]
            found = [kw for kw in risk_keywords if kw in text]
            score = min(1.0, len(found) / len(risk_keywords))
            return {
                "risk_score": score,
                "risk_level": (
                    "high" if score > 0.6 else ("medium" if score > 0.3 else "low")
                ),
                "risk_factors": found,
            }
        except Exception as exc:
            logger.error("Risk analysis for voice input error: %s", exc)
            return {"error": str(exc)}

    def _analyze_signature_for_voice_input(
        self,
        voice_result: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Анализ сигнатур для голосового ввода по истории решений."""
        try:
            decision_history: List[Dict[str, Any]] = (
                context.get("decision_history", []) if context else []
            )
            if not decision_history:
                return {"error": "No decision history available"}

            sig_methods = self.signature_methods
            if sig_methods is None:
                return {"error": "SignatureMethods not initialized"}

            signature = sig_methods.compute_decision_history_signature(decision_history)
            features = sig_methods._extract_signature_features(signature)  # noqa: SLF001

            return {
                "signature": signature,
                "signature_features": features,
                "pattern_detected": (
                    len(decision_history) > 5
                    and isinstance(features.get("std"), float)
                    and features["std"] > 0.1
                ),
            }
        except Exception as exc:
            logger.error("Signature analysis for voice input error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    async def _build_context(
        self,
        state: Dict[str, Any],
        voice_input: Optional[bytes],
    ) -> Dict[str, Any]:
        inputs = []

        if voice_input:
            voice_result = await self.voice_pipeline.process_voice_command(
                voice_input, context=state
            )
            if voice_result.get("success"):
                inputs.append((InputModality.VOICE, voice_input))
                state = {**state, **voice_result.get("context_enrichment", {})}

        if state.get("text_input"):
            inputs.append((InputModality.TEXT, state["text_input"]))

        if state.get("osint_data"):
            inputs.append((InputModality.OSINT, state["osint_data"]))

        if inputs:
            return self.context_manager.create_unified_context(inputs, base_context=state)
        return dict(state)

    # ------------------------------------------------------------------
    # Remizov analytical pass
    # ------------------------------------------------------------------

    async def _remizov_decide(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Dict[str, Any]:
        engine = self._get_remizov_engine()
        if engine is None:
            return {"analytical": False, "reason": "RemizovEngine not available"}

        try:
            state_features = self._context_to_features(context)
            result = await engine.make_enhanced_decision(
                context,
                available_actions,
                context_features=state_features,
            )
            result["analytical"] = True
            return result
        except Exception as exc:
            logger.warning("Remizov decision error: %s", exc)
            return {"analytical": False, "error": str(exc)}

    def _get_remizov_engine(self) -> Optional[Any]:
        if self._remizov_engine is not None:
            return self._remizov_engine
        try:
            from openmanus_rl.decision.remizov_solver import DecisionEngineWithRemizov

            self._remizov_engine = DecisionEngineWithRemizov(
                config=self.config.get("remizov", {})
            )
            return self._remizov_engine
        except Exception as exc:
            logger.warning("Could not load DecisionEngineWithRemizov: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Final decision synthesis
    # ------------------------------------------------------------------

    async def _make_decision(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
        remizov_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Финальный синтез: Remizov + base_agent → лучшее действие."""
        if remizov_result.get("analytical") and remizov_result.get("confidence", 0) >= 0.6:
            return {
                "action": remizov_result.get(
                    "action", available_actions[0] if available_actions else "default"
                ),
                "confidence": remizov_result.get("confidence", 0.6),
                "explanation": remizov_result.get("explanation", "Remizov solver decision"),
                "source": "remizov",
            }

        if self.base_agent is not None:
            try:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: self.base_agent.select_action(context, available_actions),
                )
                if isinstance(raw, str):
                    return {
                        "action": raw,
                        "confidence": 0.7,
                        "explanation": f"base_agent selected: {raw}",
                        "source": "base_agent",
                    }
                if isinstance(raw, dict):
                    raw.setdefault("source", "base_agent")
                    return raw
            except Exception as exc:
                logger.warning("base_agent.select_action error: %s", exc)

        remizov_action = remizov_result.get("action")
        fallback = remizov_action or (available_actions[0] if available_actions else "default")
        return {
            "action": fallback,
            "confidence": 0.4,
            "explanation": "Fallback: base_agent unavailable, using Remizov / first-available",
            "source": "fallback",
        }

    # ------------------------------------------------------------------
    # Feedback / learning
    # ------------------------------------------------------------------

    async def update_from_feedback(
        self,
        episode_id: int,
        outcome: str,
        success: bool,
    ) -> None:
        """Обновление памяти и навыков после получения обратной связи."""
        episodes = self.memory.retrieve_episodes(limit=1)
        for ep in episodes:
            if ep.get("id") == episode_id:
                skill_name = f"decision_{ep.get('action_taken', 'unknown')}"
                self.memory.store_skill(
                    skill_name=skill_name,
                    skill_data={"episode_id": episode_id, "outcome": outcome},
                    success_rate=1.0 if success else 0.0,
                )
                self.memory.increment_skill_usage(skill_name, success=success)
                logger.info("Feedback recorded for episode %s: %s", episode_id, outcome)
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _context_to_features(self, context: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []
        for v in context.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(float(v))
            elif isinstance(v, bool):
                values.append(1.0 if v else 0.0)
        return np.array(values[:10]) if values else np.array([0.5])

    # ------------------------------------------------------------------
    # Sprint 4 — Mean Field Games
    # ------------------------------------------------------------------

    async def analyze_multi_agent_scenario(
        self,
        scenario_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Анализ сценария с множеством агентов через MFG."""
        if not self.enable_mean_field_games or self.mean_field_games is None:
            return {"error": "Mean Field Games not enabled or unavailable"}

        try:
            mfg_result = self.mean_field_games.solve_mfg()
            if not mfg_result.get("success", False):
                return {"error": "MFG solving failed", "details": mfg_result}

            nash_result = self.mean_field_games.compute_nash_equilibrium()
            initial_states = self.mean_field_games._generate_initial_distribution()  # noqa: SLF001
            simulation_result = self.mean_field_games.simulate_market_dynamics(initial_states)

            result: Dict[str, Any] = {
                "mfg_solution": mfg_result,
                "nash_equilibrium": nash_result,
                "market_simulation": simulation_result,
                "scenario_config": scenario_config,
            }

            if self.memory is not None:
                self.memory.store_episode(
                    event_type="multi_agent_analysis",
                    event_data={
                        "type": "multi_agent_analysis",
                        "timestamp": datetime.utcnow().isoformat(),
                        "scenario_config": scenario_config,
                    },
                    action_taken="mfg_solve",
                    outcome=None,
                )

            return result

        except Exception as exc:
            logger.error("Multi-agent scenario analysis error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Sprint 4 — Performance optimisation
    # ------------------------------------------------------------------

    async def optimize_performance(self) -> Dict[str, Any]:
        """Оптимизация производительности и возврат текущей статистики."""
        if not self.enable_performance_optimization or self.performance_optimizer is None:
            return {"error": "Performance optimization not enabled or unavailable"}

        try:
            self.performance_optimizer.optimize_memory_usage()
            stats = self.performance_optimizer.get_performance_stats()
            recommendations = self._generate_performance_recommendations(stats)
            return {"current_stats": stats, "recommendations": recommendations}

        except Exception as exc:
            logger.error("Performance optimization error: %s", exc)
            return {"error": str(exc)}

    def _generate_performance_recommendations(
        self, stats: Dict[str, Any]
    ) -> List[str]:
        """Генерация рекомендаций по производительности на основе текущих метрик."""
        recommendations: List[str] = []

        if stats.get("current_memory_usage", 0) > 0.8:
            recommendations.append(
                "High memory usage. Consider reducing batch sizes or enabling aggressive cleanup."
            )
        if stats.get("current_cpu_usage", 0) > 0.8:
            recommendations.append(
                "High CPU usage. Consider reducing parallelism or enabling more efficient algorithms."
            )
        if stats.get("current_gpu_memory_usage", 0) > 0.8:
            recommendations.append(
                "High GPU memory usage. Consider model quantization or reducing model size."
            )

        for op, times in stats.get("processing_times_by_operation", {}).items():
            if times.get("mean", 0) > 1.0:
                recommendations.append(
                    f"Slow operation '{op}' (mean={times['mean']:.2f}s). Consider caching or optimization."
                )

        return recommendations

    async def cleanup(self) -> None:
        if self.performance_optimizer is not None:
            self.performance_optimizer.cleanup()
        await self.osint_agent.cleanup()
