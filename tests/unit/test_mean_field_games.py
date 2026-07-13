"""
Unit-тесты для MeanFieldGames (Sprint 4) — под РЕАЛЬНЫЙ API.

Реальные публичные методы (без аргументов, кроме simulate):
  solve_mfg() -> {success, iterations, convergence_error, convergence_history,
                  equilibrium_policy, mean_field_distribution{mean,std,shape}}
  compute_nash_equilibrium() -> {success, equilibrium_policy, value_functions,
                  social_welfare, price_of_anarchy, mean_field_distribution}
  simulate_market_dynamics(initial_states, num_steps=None) -> {success, num_steps,
                  price_trajectory, volatility_trajectory, final_distribution, market_metrics}
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.decision.mean_field_games import MeanFieldGames


class TestMeanFieldGames(unittest.TestCase):
    def setUp(self) -> None:
        # Малые размерности — быстрый детерминированный прогон.
        self.config = {
            "num_agents": 10,
            "state_dim": 2,
            "max_iterations": 3,
            "time_horizon": 5,
            "hidden_dim": 16,
        }
        self.mfg = MeanFieldGames(self.config)

    def test_initialization(self) -> None:
        self.assertEqual(self.mfg.num_agents, 10)
        self.assertEqual(self.mfg.state_dim, 2)
        self.assertEqual(self.mfg.max_iterations, 3)
        self.assertEqual(self.mfg.time_horizon, 5)

    def test_initialization_defaults(self) -> None:
        mfg = MeanFieldGames()
        self.assertEqual(mfg.num_agents, 100)
        self.assertEqual(mfg.state_dim, 2)
        self.assertEqual(mfg.time_horizon, 50)

    def test_solve_mfg(self) -> None:
        result = self.mfg.solve_mfg()
        self.assertTrue(result["success"])
        self.assertIn("iterations", result)
        self.assertIn("convergence_error", result)
        self.assertIn("equilibrium_policy", result)
        self.assertIn("mean_field_distribution", result)
        dist = result["mean_field_distribution"]
        self.assertIn("mean", dist)
        self.assertIn("std", dist)
        self.assertEqual(dist["shape"], [self.mfg.num_agents, self.mfg.state_dim])

    def test_compute_nash_equilibrium(self) -> None:
        result = self.mfg.compute_nash_equilibrium()
        self.assertTrue(result["success"])
        self.assertIn("equilibrium_policy", result)
        self.assertIn("value_functions", result)
        self.assertIn("social_welfare", result)
        self.assertIn("price_of_anarchy", result)

    def test_simulate_market_dynamics(self) -> None:
        initial_states = np.random.randn(self.mfg.num_agents, self.mfg.state_dim)
        result = self.mfg.simulate_market_dynamics(initial_states, num_steps=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["num_steps"], 5)
        self.assertEqual(len(result["price_trajectory"]), 5)
        self.assertEqual(len(result["volatility_trajectory"]), 5)
        self.assertIn("final_distribution", result)
        self.assertIn("market_metrics", result)
        for key in ("avg_price", "price_volatility", "max_drawdown", "sharpe_ratio"):
            self.assertIn(key, result["market_metrics"])

    def test_extract_equilibrium_policy(self) -> None:
        mu = np.random.randn(self.mfg.num_agents, self.mfg.state_dim)
        policy = self.mfg._extract_equilibrium_policy(mu)
        self.assertIn("optimal_action", policy)
        self.assertIn("mean_field_state", policy)
        self.assertIn("dispersion", policy)
        self.assertEqual(policy["strategy_type"], "mean_reverting")


if __name__ == "__main__":
    unittest.main()
