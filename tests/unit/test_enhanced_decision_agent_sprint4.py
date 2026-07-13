"""
Unit-тесты для EnhancedDecisionAgent — методы Sprint 4 — под РЕАЛЬНЫЙ API.

Реальные методы:
  async analyze_multi_agent_scenario(scenario_config) -> {mfg_solution, nash_equilibrium,
        market_simulation, scenario_config}  (или {error} если MFG выключен/недоступен)
  async optimize_performance() -> {current_stats, recommendations}  (или {error})
  _generate_performance_recommendations(stats) -> List[str]

Конструктор EnhancedDecisionAgent(base_agent=None, config=None) тяжёлый (грузит voice/memory/
qwen3/osint). Для изолированного юнит-теста Sprint-4 методов создаём bare-instance через __new__
и подставляем только нужные атрибуты + реальные Sprint-4 зависимости.
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent
from openmanus_rl.decision.mean_field_games import MeanFieldGames
from openmanus_rl.optimization.performance_optimizer import PerformanceOptimizer


def _bare_agent() -> EnhancedDecisionAgent:
    """Экземпляр без тяжёлого __init__ — только для тестирования Sprint-4 методов."""
    return EnhancedDecisionAgent.__new__(EnhancedDecisionAgent)


class TestEnhancedDecisionAgentSprint4(unittest.TestCase):
    def test_analyze_multi_agent_scenario(self) -> None:
        agent = _bare_agent()
        agent.enable_mean_field_games = True
        agent.mean_field_games = MeanFieldGames(
            {"num_agents": 10, "state_dim": 2, "max_iterations": 3, "time_horizon": 5}
        )
        agent.memory = None  # пропустить store_episode

        result = asyncio.run(agent.analyze_multi_agent_scenario({"scenario": "market_competition"}))

        self.assertNotIn("error", result)
        self.assertIn("mfg_solution", result)
        self.assertIn("nash_equilibrium", result)
        self.assertIn("market_simulation", result)
        self.assertEqual(result["scenario_config"], {"scenario": "market_competition"})
        self.assertTrue(result["mfg_solution"]["success"])

    def test_analyze_multi_agent_scenario_disabled(self) -> None:
        agent = _bare_agent()
        agent.enable_mean_field_games = False
        agent.mean_field_games = None
        result = asyncio.run(agent.analyze_multi_agent_scenario({"scenario": "x"}))
        self.assertIn("error", result)

    def test_optimize_performance(self) -> None:
        agent = _bare_agent()
        agent.enable_performance_optimization = True
        agent.performance_optimizer = PerformanceOptimizer({})
        self.addCleanup(agent.performance_optimizer.cleanup)

        result = asyncio.run(agent.optimize_performance())

        self.assertNotIn("error", result)
        self.assertIn("current_stats", result)
        self.assertIn("recommendations", result)
        self.assertIsInstance(result["recommendations"], list)

    def test_optimize_performance_disabled(self) -> None:
        agent = _bare_agent()
        agent.enable_performance_optimization = False
        agent.performance_optimizer = None
        result = asyncio.run(agent.optimize_performance())
        self.assertIn("error", result)

    def test_generate_performance_recommendations_high(self) -> None:
        agent = _bare_agent()
        stats = {
            "current_memory_usage": 0.9,
            "current_cpu_usage": 0.9,
            "current_gpu_memory_usage": 0.9,
            "processing_times_by_operation": {"inference": {"mean": 2.0}},
        }
        recs = agent._generate_performance_recommendations(stats)
        self.assertIsInstance(recs, list)
        self.assertGreaterEqual(len(recs), 4)
        text = " ".join(recs).lower()
        self.assertIn("memory", text)
        self.assertIn("cpu", text)

    def test_generate_performance_recommendations_low(self) -> None:
        agent = _bare_agent()
        stats = {
            "current_memory_usage": 0.1,
            "current_cpu_usage": 0.1,
            "current_gpu_memory_usage": 0.1,
            "processing_times_by_operation": {},
        }
        self.assertEqual(agent._generate_performance_recommendations(stats), [])


if __name__ == "__main__":
    unittest.main()
