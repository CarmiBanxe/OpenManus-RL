"""
OSINT интеграция для системы Legion через SpiderFoot API.
Sprint 2: добавлен LegionOSINTIntegrationEnhanced + async base_agent wiring.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LegionOSINTIntegration:
    """Интеграция с SpiderFoot для обогащения контекста принятия решений."""

    def __init__(self, spiderfoot_api: str = "http://localhost:5009") -> None:
        self.spiderfoot_api = spiderfoot_api
        self.search_cache: Dict[str, Any] = {}
        self.client = httpx.AsyncClient(timeout=30.0)

    async def enhance_decision_context(
        self, decision_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            search_results = await self._search_osint(decision_context)
            return {
                **decision_context,
                "osint_data": search_results,
                "risk_factors": self._extract_risk_factors(search_results),
                "confidence_score": self._calculate_confidence(search_results),
            }
        except Exception as exc:
            logger.error(f"OSINT enhancement error: {exc}")
            return decision_context

    async def _search_osint(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        queries = self._generate_queries(context)
        results: Dict[str, Any] = {}
        for query in queries:
            if query in self.search_cache:
                results[query] = self.search_cache[query]
                continue
            try:
                resp = await self.client.post(
                    f"{self.spiderfoot_api}/api/scan",
                    json={"target": query, "modules": ["all"]},
                )
                data = resp.json() if resp.status_code == 200 else {"error": resp.status_code}
                results[query] = data
                self.search_cache[query] = data
            except Exception as exc:
                logger.error(f"OSINT search error for {query!r}: {exc}")
                results[query] = {"error": str(exc)}
        return results

    def _generate_queries(self, context: Dict[str, Any]) -> List[str]:
        queries: List[str] = []
        for key in ("entities", "keywords", "names"):
            queries.extend(context.get(key, []))
        return queries

    def _extract_risk_factors(self, results: Dict[str, Any]) -> List[str]:
        risk_factors: List[str] = []
        risk_types = {"sanction", "blacklist", "warning", "risk"}
        for query, data in results.items():
            for item in data.get("data", []):
                if isinstance(item, dict) and item.get("type") in risk_types:
                    risk_factors.append(f"{query}: {item.get('value', 'Unknown')}")
        return risk_factors

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        total = sum(
            len(d.get("data", []))
            for d in results.values()
            if isinstance(d.get("data"), list)
        )
        if total == 0:
            return 0.0
        if total < 5:
            return 0.5
        if total < 20:
            return 0.7
        return 0.9

    async def cleanup(self) -> None:
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Sprint 2: Enhanced OSINT integration with multimodal context support
# ---------------------------------------------------------------------------


class LegionOSINTIntegrationEnhanced(LegionOSINTIntegration):
    """
    Sprint 2: расширенная OSINT-интеграция.
    Поддерживает мультимодальный контекст, приоритизацию запросов и кэш TTL.
    """

    def __init__(
        self,
        spiderfoot_api: str = "http://localhost:5009",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(spiderfoot_api=spiderfoot_api)
        self.config = config or {}
        self.cache_ttl: int = self.config.get("cache_ttl", 300)
        self.max_parallel_queries: int = self.config.get("max_parallel_queries", 5)
        self.priority_sources: List[str] = self.config.get(
            "priority_sources", ["sanction", "financial", "corporate"]
        )

    async def enhance_multimodal_context(
        self,
        context: Dict[str, Any],
        modalities: Optional[List[str]] = None,
        priority: float = 0.5,
    ) -> Dict[str, Any]:
        """Обогащение контекста с учётом мультимодальных источников."""
        try:
            base_enhanced = await self.enhance_decision_context(context)

            modality_insights = await self._gather_modality_insights(
                context, modalities or []
            )

            priority_adjusted = self._apply_priority_weighting(
                base_enhanced, priority
            )

            return {
                **priority_adjusted,
                "modality_insights": modality_insights,
                "enhancement_priority": priority,
                "enhanced_by": "LegionOSINTIntegrationEnhanced",
            }
        except Exception as exc:
            logger.error(f"Multimodal OSINT enhancement error: {exc}")
            return context

    async def _gather_modality_insights(
        self,
        context: Dict[str, Any],
        modalities: List[str],
    ) -> Dict[str, Any]:
        insights: Dict[str, Any] = {}
        for modality in modalities[:self.max_parallel_queries]:
            insights[modality] = self._extract_modality_specific_data(context, modality)
        return insights

    def _extract_modality_specific_data(
        self,
        context: Dict[str, Any],
        modality: str,
    ) -> Dict[str, Any]:
        if modality == "voice":
            return {"transcript": context.get("voice_input", ""), "has_voice": True}
        if modality == "text":
            return {"content": context.get("text_input", ""), "has_text": True}
        return {"modality": modality, "data": context.get(modality, {})}

    def _apply_priority_weighting(
        self,
        enhanced_context: Dict[str, Any],
        priority: float,
    ) -> Dict[str, Any]:
        result = dict(enhanced_context)
        current_confidence = float(result.get("confidence_score", 0.5))
        result["confidence_score"] = min(1.0, current_confidence * (0.5 + priority * 0.5))
        return result


# ---------------------------------------------------------------------------
# Sprint 2: Updated LegionOSINTEnhancedAgent
# ---------------------------------------------------------------------------


class LegionOSINTEnhancedAgent:
    """
    Sprint 2: OSINT-агент с поддержкой base_agent и async action selection.
    Сохранён оригинальный select_action для обратной совместимости.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = cfg
        self.osint = LegionOSINTIntegrationEnhanced(
            spiderfoot_api=cfg.get("spiderfoot_api", "http://localhost:5009"),
            config=cfg,
        )
        self._base_agent: Optional[Any] = None
        logger.info("LegionOSINTEnhancedAgent (Sprint 2) initialized")

    def set_base_agent(self, base_agent: Any) -> None:
        """Подключение базового агента принятия решений (Sprint 2)."""
        self._base_agent = base_agent
        logger.info(f"Base agent set: {type(base_agent).__name__}")

    async def select_action_async(
        self,
        state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
        priority: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Sprint 2: async-версия с OSINT обогащением и base_agent.
        Возвращает обогащённый контекст + выбранное действие.
        """
        try:
            modalities = list(set(state.get("modalities", [])))
            enhanced_ctx = await self.osint.enhance_multimodal_context(
                state, modalities=modalities, priority=priority
            )

            if self._base_agent is not None:
                action_names = [
                    a.get("action", str(a)) if isinstance(a, dict) else str(a)
                    for a in available_actions
                ]
                try:
                    raw_result = self._base_agent.select_action(
                        enhanced_ctx, action_names
                    )
                    if isinstance(raw_result, dict):
                        base_action = raw_result.get("action", "wait")
                        base_explanation = raw_result.get("explanation", "")
                        base_confidence = raw_result.get("confidence", 0.7)
                    else:
                        base_action = str(raw_result)
                        base_explanation = f"base_agent selected: {base_action}"
                        base_confidence = 0.7
                except Exception as exc:
                    logger.warning(f"base_agent.select_action failed: {exc}")
                    base_action = "wait"
                    base_explanation = f"base_agent error: {exc}"
                    base_confidence = 0.3
            else:
                base_action = "wait"
                base_explanation = "No base_agent — OSINT context ready"
                base_confidence = 0.5

            return {
                "action": base_action,
                "explanation": base_explanation,
                "confidence": base_confidence,
                "osint_enhanced": True,
                "osint_data": enhanced_ctx.get("osint_data", {}),
                "risk_factors": enhanced_ctx.get("risk_factors", []),
                "confidence_score": enhanced_ctx.get("confidence_score", 0.0),
                "context": enhanced_ctx,
            }

        except Exception as exc:
            logger.error(f"OSINT async action selection error: {exc}")
            return {
                "action": "error",
                "explanation": str(exc),
                "confidence": 0.0,
                "osint_enhanced": False,
                "risk_factors": [],
                "confidence_score": 0.0,
                "context": state,
            }

    # Kept for backward compatibility
    async def select_action(
        self,
        state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            enhanced = await self.osint.enhance_decision_context(state)
            action = "wait"
            explanation = "OSINT enhanced context ready"
            if self._base_agent is not None:
                action_names = [
                    a.get("action", str(a)) if isinstance(a, dict) else str(a)
                    for a in available_actions
                ]
                try:
                    result = self._base_agent.select_action(enhanced, action_names)
                    action = result if isinstance(result, str) else result.get("action", "wait")
                    explanation = f"base_agent: {action}"
                except Exception as exc:
                    logger.warning(f"base_agent error in select_action: {exc}")
            return {
                "action": action,
                "explanation": explanation,
                "osint_enhanced": True,
                "risk_factors": enhanced.get("risk_factors", []),
                "confidence_score": enhanced.get("confidence_score", 0.0),
            }
        except Exception as exc:
            logger.error(f"OSINT action selection error: {exc}")
            return {"action": "error", "explanation": str(exc), "osint_enhanced": False}

    async def cleanup(self) -> None:
        await self.osint.cleanup()
