"""
Интеграционные тесты многоагентных сценариев — под РЕАЛЬНЫЙ API.

Реальный метод: analyze_multi_agent_scenario(scenario_config) ->
  {mfg_solution, nash_equilibrium, market_simulation, scenario_config}
Использует реальный MeanFieldGames (solve_mfg / compute_nash_equilibrium /
simulate_market_dynamics). При выключенном MFG -> {error}.
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

SCENARIOS = [
    {"scenario": "market_competition", "num_agents": 8},
    {"scenario": "resource_allocation", "num_agents": 8},
    {"scenario": "consensus_formation", "num_agents": 8},
    {"scenario": "network_diffusion", "num_agents": 8},
]


class TestMultiAgentScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = EnhancedDecisionAgent(config=SANDBOX_CFG)
        if cls.agent.performance_optimizer is not None:
            cls.addClassCleanup(cls.agent.performance_optimizer.cleanup)

    def _assert_coherent(self, res: dict, scenario_config: dict) -> None:
        self.assertNotIn("error", res)
        for key in ("mfg_solution", "nash_equilibrium", "market_simulation", "scenario_config"):
            self.assertIn(key, res)
        self.assertEqual(res["scenario_config"], scenario_config)
        self.assertTrue(res["mfg_solution"]["success"])
        self.assertTrue(res["nash_equilibrium"]["success"])
        self.assertTrue(res["market_simulation"]["success"])

    def test_scenarios(self) -> None:
        for cfg in SCENARIOS:
            with self.subTest(scenario=cfg["scenario"]):
                res = asyncio.run(self.agent.analyze_multi_agent_scenario(cfg))
                self._assert_coherent(res, cfg)

    def test_multi_agent_disabled(self) -> None:
        agent = EnhancedDecisionAgent(config={**SANDBOX_CFG, "enable_mean_field_games": False})
        try:
            res = asyncio.run(agent.analyze_multi_agent_scenario({"scenario": "x"}))
            self.assertIn("error", res)
        finally:
            if agent.performance_optimizer is not None:
                agent.performance_optimizer.cleanup()

    def test_nash_equilibrium_structure(self) -> None:
        res = asyncio.run(self.agent.analyze_multi_agent_scenario({"scenario": "market_competition"}))
        nash = res["nash_equilibrium"]
        for key in ("equilibrium_policy", "value_functions", "social_welfare", "price_of_anarchy"):
            self.assertIn(key, nash)


if __name__ == "__main__":
    unittest.main()
