"""
Интеграционные тесты продвинутых сценариев — под РЕАЛЬНЫЙ API.

Композитные потоки поверх реальных методов агента: select_action (решение),
analyze_multi_agent_scenario (MFG), optimize_performance (PerfOpt). Без фиктивного
process_input — драйвим настоящий пайплайн в sandbox.
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent

SANDBOX_CFG = {
    "qwen3_omni": {"sandbox_mode": True},
    "voice_pipeline": {"sandbox_mode": True},
    "mean_field_games": {"num_agents": 8, "state_dim": 2, "max_iterations": 2, "time_horizon": 3},
}
ACTIONS = ["hedge", "hold", "reduce", "wait"]


class TestAdvancedScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = EnhancedDecisionAgent(config=SANDBOX_CFG)
        if cls.agent.performance_optimizer is not None:
            cls.addClassCleanup(cls.agent.performance_optimizer.cleanup)

    def test_decision_then_multiagent_then_optimize(self) -> None:
        decision = asyncio.run(self.agent.select_action(
            {"text": "Simulate market crash impact on portfolio", "entities": ["portfolio"]}, ACTIONS))
        self.assertIn("action", decision)

        multi = asyncio.run(self.agent.analyze_multi_agent_scenario({"scenario": "financial_crisis"}))
        self.assertTrue(multi["mfg_solution"]["success"])

        perf = asyncio.run(self.agent.optimize_performance())
        self.assertIn("current_stats", perf)
        self.assertIn("recommendations", perf)

    def test_repeated_decisions_yield_episodes(self) -> None:
        ids = []
        for i in range(3):
            res = asyncio.run(self.agent.select_action({"text": f"Advanced query {i}"}, ACTIONS))
            self.assertIn("episode_id", res)
            ids.append(res["episode_id"])
        self.assertEqual(len(ids), 3)

    def test_sprint3_components_wired(self) -> None:
        # Продвинутые сценарии опираются на S3-компоненты — проверяем, что они на месте.
        self.assertIsNotNone(self.agent.deep_hedging)
        self.assertIsNotNone(self.agent.signature_methods)
        self.assertIsNotNone(self.agent.qwen3_omni)

    def test_priority_variation(self) -> None:
        for priority in (0.1, 0.5, 0.9):
            with self.subTest(priority=priority):
                res = asyncio.run(self.agent.select_action(
                    {"text": "Risk management framework"}, ACTIONS, priority=priority))
                self.assertIn("action", res)
                self.assertIsInstance(res["confidence"], (int, float))


if __name__ == "__main__":
    unittest.main()
