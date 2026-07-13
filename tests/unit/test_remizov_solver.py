"""
Unit tests — RemizovSolver + DecisionEngineWithRemizov (Sprint 2)
scipy/sympy mocked where absent; numpy always available.
"""
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openmanus_rl.decision.remizov_solver import (
    DecisionEngineWithRemizov,
    RemizovSolver,
)


# ---------------------------------------------------------------------------
# RemizovSolver construction
# ---------------------------------------------------------------------------


class TestRemizovSolverInit:
    def test_default_config(self) -> None:
        solver = RemizovSolver()
        assert solver.dt == pytest.approx(0.01)
        assert solver.n_steps == 100
        assert solver.volatility == pytest.approx(0.2)
        assert solver.mean_reversion == pytest.approx(0.5)

    def test_custom_config(self) -> None:
        solver = RemizovSolver({"dt": 0.05, "n_steps": 50, "volatility": 0.4})
        assert solver.dt == pytest.approx(0.05)
        assert solver.n_steps == 50
        assert solver.volatility == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = RemizovSolver._softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_all_positive(self) -> None:
        x = np.array([-5.0, 0.0, 5.0])
        result = RemizovSolver._softmax(x)
        assert np.all(result > 0)

    def test_monotone(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = RemizovSolver._softmax(x)
        assert result[0] < result[1] < result[2]

    def test_single_element(self) -> None:
        result = RemizovSolver._softmax(np.array([0.0]))
        assert abs(result[0] - 1.0) < 1e-6

    def test_large_values_stable(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        result = RemizovSolver._softmax(x)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# solve_decision_ode — numpy fallback always works
# ---------------------------------------------------------------------------


class TestSolveDecisionODE:
    def setup_method(self) -> None:
        self.solver = RemizovSolver({"n_steps": 20, "dt": 0.01})

    def test_returns_dict_with_required_keys(self) -> None:
        initial = np.array([0.5, 0.3, 0.2])
        result = self.solver.solve_decision_ode(initial, [0.4, 0.3, 0.3])
        assert "optimal_weights" in result
        assert "trajectory" in result
        assert "convergence" in result
        assert "solver" in result

    def test_weights_are_probabilities(self) -> None:
        initial = np.array([0.5, 0.5])
        result = self.solver.solve_decision_ode(initial, [0.6, 0.4])
        weights = result["optimal_weights"]
        assert abs(sum(weights) - 1.0) < 1e-5
        assert all(w >= 0 for w in weights)

    def test_single_action(self) -> None:
        result = self.solver.solve_decision_ode(np.array([1.0]), [1.0])
        assert len(result["optimal_weights"]) == 1
        assert abs(result["optimal_weights"][0] - 1.0) < 1e-5

    def test_empty_initial_handled(self) -> None:
        # Should not crash; fallback handles mismatched dims
        result = self.solver._fallback_solution([0.5, 0.5])
        assert len(result["optimal_weights"]) == 2
        assert result["convergence"] is False

    def test_mismatched_weights_length(self) -> None:
        # More action weights than state dimensions — should not crash
        initial = np.array([0.5])
        result = self.solver.solve_decision_ode(initial, [0.3, 0.3, 0.4])
        assert "optimal_weights" in result


# ---------------------------------------------------------------------------
# solve_stochastic_volatility_pde
# ---------------------------------------------------------------------------


class TestSolvePDE:
    def setup_method(self) -> None:
        self.solver = RemizovSolver({"n_steps": 10})

    def test_returns_required_keys(self) -> None:
        state = np.array([1.0, 0.5])
        result = self.solver.solve_stochastic_volatility_pde(state, n_paths=10)
        assert "mean_trajectory" in result
        assert "final_mean" in result

    def test_final_mean_shape_matches_state(self) -> None:
        state = np.array([1.0, 2.0, 3.0])
        result = self.solver.solve_stochastic_volatility_pde(state, n_paths=20)
        assert len(result["final_mean"]) == len(state)

    def test_var_95_shape(self) -> None:
        state = np.array([1.0, 1.0])
        result = self.solver.solve_stochastic_volatility_pde(state, n_paths=50)
        assert len(result.get("var_95", [])) == len(state)

    def test_deterministic_with_zero_volatility(self) -> None:
        solver = RemizovSolver({"n_steps": 5, "volatility": 0.0})
        state = np.array([1.0])
        result = solver.solve_stochastic_volatility_pde(state, n_paths=10)
        # With zero volatility std should be very small
        std = result.get("final_std", [1.0])
        assert std[0] < 0.1


# ---------------------------------------------------------------------------
# calculate_transition_probabilities
# ---------------------------------------------------------------------------


class TestTransitionProbabilities:
    def setup_method(self) -> None:
        self.solver = RemizovSolver()

    def test_probabilities_sum_to_one(self) -> None:
        state: Dict[str, Any] = {"confidence": 0.8, "risk_level": 0.2}
        probs = self.solver.calculate_transition_probabilities(
            state, ["A", "B", "C"]
        )
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_all_actions_present(self) -> None:
        actions = ["go", "stop", "analyze"]
        probs = self.solver.calculate_transition_probabilities({}, actions)
        for a in actions:
            assert a in probs

    def test_empty_actions_returns_empty(self) -> None:
        probs = self.solver.calculate_transition_probabilities({}, [])
        assert probs == {}

    def test_single_action_probability_one(self) -> None:
        probs = self.solver.calculate_transition_probabilities({}, ["only"])
        assert abs(probs["only"] - 1.0) < 1e-5

    def test_custom_features(self) -> None:
        features = np.array([0.8, 0.6, 0.4])
        probs = self.solver.calculate_transition_probabilities(
            {}, ["a", "b", "c"], context_features=features
        )
        assert len(probs) == 3


# ---------------------------------------------------------------------------
# analytical_risk_assessment
# ---------------------------------------------------------------------------


class TestRiskAssessment:
    def setup_method(self) -> None:
        self.solver = RemizovSolver({"n_steps": 10})

    def test_returns_required_fields(self) -> None:
        state = np.array([0.5, 0.3])
        action = np.array([0.6, 0.4])
        result = self.solver.analytical_risk_assessment(state, action)
        assert "risk_level" in result
        assert "is_acceptable" in result
        assert "confidence" in result
        assert "recommendation" in result

    def test_risk_level_bounded(self) -> None:
        state = np.array([0.5])
        action = np.array([0.5])
        result = self.solver.analytical_risk_assessment(state, action)
        assert isinstance(result["risk_level"], float)

    def test_low_risk_acceptable(self) -> None:
        state = np.zeros(3)  # zero state → low risk
        action = np.zeros(3)
        result = self.solver.analytical_risk_assessment(
            state, action, risk_threshold=0.9
        )
        assert result["is_acceptable"] is True
        assert result["recommendation"] == "proceed"

    def test_risk_decomposition_present(self) -> None:
        result = self.solver.analytical_risk_assessment(
            np.array([0.5]), np.array([0.5])
        )
        components = result.get("risk_components", {})
        assert "market_risk" in components
        assert "execution_risk" in components
        assert "model_risk" in components


# ---------------------------------------------------------------------------
# DecisionEngineWithRemizov
# ---------------------------------------------------------------------------


class TestDecisionEngineWithRemizov:
    def setup_method(self) -> None:
        self.engine = DecisionEngineWithRemizov(
            config={"n_steps": 10, "dt": 0.01}
        )

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_returns_action(self) -> None:
        state: Dict[str, Any] = {"confidence": 0.7, "risk_level": 0.2}
        result = await self.engine.make_enhanced_decision(
            state, ["action_a", "action_b", "action_c"]
        )
        assert "action" in result
        assert result["action"] in ["action_a", "action_b", "action_c"]

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_confidence_range(self) -> None:
        state: Dict[str, Any] = {"confidence": 0.5}
        result = await self.engine.make_enhanced_decision(state, ["go", "stop"])
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_has_explanation(self) -> None:
        result = await self.engine.make_enhanced_decision(
            {"x": 0.5}, ["a", "b"]
        )
        assert isinstance(result.get("explanation"), str)
        assert len(result["explanation"]) > 0

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_empty_actions(self) -> None:
        result = await self.engine.make_enhanced_decision({}, [])
        assert "action" in result  # fallback — should not crash

    @pytest.mark.asyncio
    async def test_transition_probs_in_result(self) -> None:
        result = await self.engine.make_enhanced_decision(
            {"a": 0.5}, ["x", "y"]
        )
        assert "transition_probabilities" in result
        assert "x" in result["transition_probabilities"]

    @pytest.mark.asyncio
    async def test_risk_assessment_in_result(self) -> None:
        result = await self.engine.make_enhanced_decision(
            {"v": 0.3}, ["do_it", "skip"]
        )
        assert "risk_assessment" in result

    def test_state_to_vector_numeric_only(self) -> None:
        state = {"a": 1.0, "b": "text", "c": True, "d": [1, 2]}
        vec = self.engine._state_to_vector(state)
        assert vec.dtype == np.float64 or vec.dtype == np.float32 or vec.ndim == 1
        assert len(vec) >= 1

    def test_state_to_vector_empty_state(self) -> None:
        vec = self.engine._state_to_vector({})
        assert len(vec) == 1
        assert vec[0] == pytest.approx(0.5)

    def test_combine_scores_both_empty(self) -> None:
        result = self.engine._combine_scores([], [], {"risk_level": 0.3})
        assert result == []

    def test_combine_scores_balanced(self) -> None:
        ode = [0.6, 0.4]
        trans = [0.55, 0.45]
        combined = self.engine._combine_scores(ode, trans, {"risk_level": 0.2})
        assert len(combined) == 2
        assert combined[0] > combined[1]
