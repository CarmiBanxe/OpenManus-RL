"""
Unit-тесты для LegionOSINTIntegration (Sprint 2) — под РЕАЛЬНЫЙ API.

Реальные методы: __init__(spiderfoot_api), enhance_decision_context (async),
_generate_queries, _extract_risk_factors, _calculate_confidence, cleanup (async).
"""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.integration.legion_osint_integration import LegionOSINTIntegration


class TestLegionOSINTIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.osint = LegionOSINTIntegration(spiderfoot_api="http://localhost:5009")

    async def asyncTearDown(self) -> None:
        await self.osint.cleanup()

    def test_initialization(self) -> None:
        self.assertEqual(self.osint.spiderfoot_api, "http://localhost:5009")
        self.assertEqual(self.osint.search_cache, {})
        self.assertIsNotNone(self.osint.client)

    def test_generate_queries(self) -> None:
        context = {
            "entities": ["Acme Corp"],
            "keywords": ["fraud"],
            "names": ["John Doe"],
        }
        queries = self.osint._generate_queries(context)
        self.assertEqual(queries, ["Acme Corp", "fraud", "John Doe"])

    def test_generate_queries_empty(self) -> None:
        self.assertEqual(self.osint._generate_queries({"text": "no entities"}), [])

    def test_extract_risk_factors(self) -> None:
        results = {
            "Acme Corp": {
                "data": [
                    {"type": "sanction", "value": "OFAC list"},
                    {"type": "clean", "value": "ignore me"},
                ]
            }
        }
        factors = self.osint._extract_risk_factors(results)
        self.assertEqual(factors, ["Acme Corp: OFAC list"])

    def test_extract_risk_factors_none(self) -> None:
        results = {"q": {"data": [{"type": "clean", "value": "x"}]}}
        self.assertEqual(self.osint._extract_risk_factors(results), [])

    def test_calculate_confidence_tiers(self) -> None:
        self.assertEqual(self.osint._calculate_confidence({}), 0.0)
        self.assertEqual(
            self.osint._calculate_confidence({"q": {"data": [1, 2, 3]}}), 0.5
        )
        self.assertEqual(
            self.osint._calculate_confidence({"q": {"data": list(range(10))}}), 0.7
        )
        self.assertEqual(
            self.osint._calculate_confidence({"q": {"data": list(range(25))}}), 0.9
        )

    async def test_enhance_decision_context_empty(self) -> None:
        # Нет entities/keywords/names -> нет сетевых вызовов, детерминировано.
        context = {"text": "hello"}
        result = await self.osint.enhance_decision_context(context)
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["osint_data"], {})
        self.assertEqual(result["risk_factors"], [])
        self.assertEqual(result["confidence_score"], 0.0)

    async def test_enhance_decision_context_with_data(self) -> None:
        osint_payload = {
            "Acme Corp": {"data": [{"type": "sanction", "value": "OFAC list"}]}
        }
        with patch.object(
            self.osint, "_search_osint", new_callable=AsyncMock, return_value=osint_payload
        ):
            result = await self.osint.enhance_decision_context({"entities": ["Acme Corp"]})
        self.assertEqual(result["osint_data"], osint_payload)
        self.assertEqual(result["risk_factors"], ["Acme Corp: OFAC list"])
        self.assertEqual(result["confidence_score"], 0.5)

    async def test_enhance_decision_context_error_returns_original(self) -> None:
        with patch.object(
            self.osint, "_search_osint", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            context = {"entities": ["X"]}
            result = await self.osint.enhance_decision_context(context)
        # При ошибке возвращается исходный контекст без обогащения.
        self.assertEqual(result, context)
        self.assertNotIn("osint_data", result)


if __name__ == "__main__":
    unittest.main()
