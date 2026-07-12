"""
OSINT интеграция для системы Legion через SpiderFoot API.
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
                'osint_data': search_results,
                'risk_factors': self._extract_risk_factors(search_results),
                'confidence_score': self._calculate_confidence(search_results),
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
        for key in ('entities', 'keywords', 'names'):
            queries.extend(context.get(key, []))
        return queries

    def _extract_risk_factors(self, results: Dict[str, Any]) -> List[str]:
        risk_factors: List[str] = []
        risk_types = {'sanction', 'blacklist', 'warning', 'risk'}
        for query, data in results.items():
            for item in data.get('data', []):
                if isinstance(item, dict) and item.get('type') in risk_types:
                    risk_factors.append(f"{query}: {item.get('value', 'Unknown')}")
        return risk_factors

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        total = sum(
            len(d.get('data', []))
            for d in results.values()
            if isinstance(d.get('data'), list)
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


class LegionOSINTEnhancedAgent:
    """Агент с OSINT обогащением контекста."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.osint = LegionOSINTIntegration(
            spiderfoot_api=cfg.get('spiderfoot_api', 'http://localhost:5009')
        )
        logger.info("LegionOSINTEnhancedAgent initialized")

    async def select_action(
        self,
        state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            enhanced = await self.osint.enhance_decision_context(state)
            # Sprint 2: подключить base_agent вместо заглушки
            return {
                'action': 'wait',
                'explanation': 'OSINT enhanced context ready',
                'osint_enhanced': True,
                'risk_factors': enhanced.get('risk_factors', []),
                'confidence_score': enhanced.get('confidence_score', 0.0),
            }
        except Exception as exc:
            logger.error(f"OSINT action selection error: {exc}")
            return {'action': 'error', 'explanation': str(exc), 'osint_enhanced': False}

    async def cleanup(self) -> None:
        await self.osint.cleanup()
