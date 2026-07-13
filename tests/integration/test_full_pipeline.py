"""
Интеграционные тесты полного пайплайна OpenManus — под РЕАЛЬНЫЙ API.

Реальная точка входа: EnhancedDecisionAgent.select_action(state, available_actions,
voice_input=None, priority=0.5) -> {action, confidence, explanation, osint_enhanced,
remizov_used, source, episode_id, timestamp}.
Конструктор: EnhancedDecisionAgent(base_agent=None, config=None). Sandbox через sub-configs.
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
ACTIONS = ["buy", "sell", "wait"]


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = EnhancedDecisionAgent(config=SANDBOX_CFG)
        if cls.agent.performance_optimizer is not None:
            cls.addClassCleanup(cls.agent.performance_optimizer.cleanup)

    def test_full_pipeline_construction(self) -> None:
        for attr in (
            "context_manager", "voice_pipeline", "memory", "osint_agent",
            "qwen3_omni", "deep_hedging", "signature_methods",
            "mean_field_games", "performance_optimizer",
        ):
            self.assertIsNotNone(getattr(self.agent, attr), f"{attr} is None")

    def test_text_input_pipeline(self) -> None:
        res = asyncio.run(self.agent.select_action({"text": "What is the risk of BTC?"}, ACTIONS))
        for key in ("action", "confidence", "explanation", "osint_enhanced", "episode_id", "timestamp"):
            self.assertIn(key, res)
        self.assertIn(res["action"], ACTIONS + ["wait", "error"])
        self.assertIsInstance(res["confidence"], (int, float))

    def test_osint_enhanced_pipeline(self) -> None:
        res = asyncio.run(self.agent.select_action({"text": "Analyze XYZ Corp", "entities": ["XYZ Corp"]}, ACTIONS))
        self.assertIn("osint_enhanced", res)
        self.assertIsInstance(res["osint_enhanced"], bool)

    def test_multi_agent_pipeline(self) -> None:
        res = asyncio.run(self.agent.analyze_multi_agent_scenario({"scenario": "market_competition"}))
        self.assertNotIn("error", res)
        for key in ("mfg_solution", "nash_equilibrium", "market_simulation", "scenario_config"):
            self.assertIn(key, res)
        self.assertTrue(res["mfg_solution"]["success"])

    def test_performance_optimization_pipeline(self) -> None:
        res = asyncio.run(self.agent.optimize_performance())
        self.assertIn("current_stats", res)
        self.assertIn("recommendations", res)
        self.assertIsInstance(res["recommendations"], list)

    def test_concurrent_processing(self) -> None:
        async def run_all():
            tasks = [
                self.agent.select_action({"text": f"Concurrent {i}"}, ACTIONS)
                for i in range(3)
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn("action", res)


if __name__ == "__main__":
    unittest.main()
