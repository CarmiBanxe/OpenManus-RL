"""
Tests — DeepHedgingFramework + DeepHedgingNetwork (Sprint 3).
Используются маленькие данные чтобы тесты оставались быстрыми (<2s each).
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from openmanus_rl.decision.deep_hedging import DeepHedgingFramework, DeepHedgingNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_config() -> Dict[str, Any]:
    return {
        "input_dim": 11,
        "hidden_dim": 16,
        "output_dim": 1,
        "learning_rate": 0.01,
        "epochs": 2,
        "batch_size": 4,
        "risk_aversion": 0.5,
        "transaction_cost": 0.001,
    }


@pytest.fixture()
def framework(small_config: Dict[str, Any]) -> DeepHedgingFramework:
    return DeepHedgingFramework(small_config)


@pytest.fixture()
def price_data() -> np.ndarray:
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(30) * 0.5)
    return prices.astype(np.float64)


@pytest.fixture()
def option_payoffs(price_data: np.ndarray) -> np.ndarray:
    strike = 100.0
    return np.maximum(price_data - strike, 0.0)


# ---------------------------------------------------------------------------
# DeepHedgingNetwork
# ---------------------------------------------------------------------------


class TestDeepHedgingNetwork:
    def test_output_shape(self) -> None:
        net = DeepHedgingNetwork(input_dim=11, hidden_dim=16, output_dim=1)
        x = torch.randn(4, 11)
        out = net(x)
        assert out.shape == (4, 1)

    def test_batch_size_1(self) -> None:
        net = DeepHedgingNetwork(11, 16, 1)
        out = net(torch.randn(1, 11))
        assert out.shape == (1, 1)

    def test_larger_hidden(self) -> None:
        net = DeepHedgingNetwork(5, 64, 2)
        out = net(torch.randn(8, 5))
        assert out.shape == (8, 2)

    def test_forward_is_deterministic_eval_mode(self) -> None:
        net = DeepHedgingNetwork(11, 16, 1)
        net.eval()
        x = torch.randn(2, 11)
        a = net(x).detach().numpy()
        b = net(x).detach().numpy()
        np.testing.assert_array_equal(a, b)

    def test_dropout_active_in_train_mode(self) -> None:
        net = DeepHedgingNetwork(11, 32, 1)
        net.train()
        x = torch.randn(50, 11)
        a = net(x).detach().numpy()
        b = net(x).detach().numpy()
        # Dropout introduces randomness → outputs should differ
        assert not np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# DeepHedgingFramework — construction
# ---------------------------------------------------------------------------


class TestFrameworkInit:
    def test_default_config(self) -> None:
        fw = DeepHedgingFramework()
        assert fw.input_dim == 10
        assert fw.hidden_dim == 64
        assert fw.output_dim == 1
        assert fw.risk_aversion == 0.5

    def test_custom_config(self, small_config: Dict[str, Any]) -> None:
        fw = DeepHedgingFramework(small_config)
        assert fw.hidden_dim == 16
        assert fw.epochs == 2

    def test_model_on_device(self, framework: DeepHedgingFramework) -> None:
        params = list(framework.model.parameters())
        assert len(params) > 0

    def test_optimizer_initialized(self, framework: DeepHedgingFramework) -> None:
        assert framework.optimizer is not None


# ---------------------------------------------------------------------------
# _generate_features
# ---------------------------------------------------------------------------


class TestGenerateFeatures:
    def test_shape(self, framework: DeepHedgingFramework, price_data: np.ndarray) -> None:
        feats = framework._generate_features(price_data)
        assert feats.shape == (len(price_data), framework.input_dim)

    def test_early_rows_padded(self, framework: DeepHedgingFramework) -> None:
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        feats = framework._generate_features(prices)
        assert feats.shape[0] == 4

    def test_no_nan(self, framework: DeepHedgingFramework, price_data: np.ndarray) -> None:
        feats = framework._generate_features(price_data)
        assert not np.any(np.isnan(feats))

    def test_feature_error_fallback(self, framework: DeepHedgingFramework) -> None:
        result = framework._generate_features(np.array([]))
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# _create_dataset
# ---------------------------------------------------------------------------


class TestCreateDataset:
    def test_length(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        feats = framework._generate_features(price_data)
        ds = framework._create_dataset(price_data, option_payoffs, feats)
        assert len(ds) == len(price_data) - 1

    def test_targets_finite(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        feats = framework._generate_features(price_data)
        ds = framework._create_dataset(price_data, option_payoffs, feats)
        for _inputs, target in ds:
            assert np.isfinite(target)


# ---------------------------------------------------------------------------
# train_hedging_strategy
# ---------------------------------------------------------------------------


class TestTrainHedgingStrategy:
    def test_returns_success(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        result = framework.train_hedging_strategy(price_data, option_payoffs)
        assert result["success"] is True

    def test_training_history_length(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        result = framework.train_hedging_strategy(price_data, option_payoffs)
        assert len(result["training_history"]) == framework.epochs

    def test_loss_is_finite(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        result = framework.train_hedging_strategy(price_data, option_payoffs)
        for entry in result["training_history"]:
            assert np.isfinite(entry["loss"])

    def test_evaluation_keys_present(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        result = framework.train_hedging_strategy(price_data, option_payoffs)
        eval_keys = {"final_pnl", "total_cost", "sharpe_ratio", "max_drawdown"}
        assert eval_keys.issubset(result["evaluation"].keys())

    def test_model_parameters_returned(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        result = framework.train_hedging_strategy(price_data, option_payoffs)
        assert "model_parameters" in result
        assert result["model_parameters"]["input_dim"] == framework.input_dim

    def test_with_explicit_features(
        self,
        framework: DeepHedgingFramework,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
    ) -> None:
        feats = framework._generate_features(price_data)
        result = framework.train_hedging_strategy(price_data, option_payoffs, features=feats)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# generate_hedging_decision
# ---------------------------------------------------------------------------


class TestGenerateHedgingDecision:
    def test_returns_expected_keys(self, framework: DeepHedgingFramework) -> None:
        state = np.array([100.0, 100.0, 1.0, 99.0, 101.0, 0.0, 0.5, 100.0, 100.0, 50.0, 0.2])
        result = framework.generate_hedging_decision(state, 0.0)
        for key in ("hedge_ratio", "recommended_position", "transaction_cost", "confidence"):
            assert key in result

    def test_transaction_cost_nonneg(self, framework: DeepHedgingFramework) -> None:
        state = np.random.randn(11)
        result = framework.generate_hedging_decision(state, 1.0)
        assert result.get("transaction_cost", 0.0) >= 0.0

    def test_confidence_in_range(self, framework: DeepHedgingFramework) -> None:
        # state[10] is volatility — must be non-negative for confidence to stay in [0, 1]
        state = np.zeros(11)
        state[10] = 0.3
        result = framework.generate_hedging_decision(state, 0.0)
        conf = result.get("confidence", 0.5)
        assert 0.0 <= conf <= 1.0

    def test_short_state_fallback(self, framework: DeepHedgingFramework) -> None:
        state = np.random.randn(5)
        # Should not raise; confidence falls back to 0.5
        result = framework.generate_hedging_decision(state, 0.0)
        assert "hedge_ratio" in result or "error" in result

    def test_error_path_returns_fallback_position(self, framework: DeepHedgingFramework) -> None:
        # Must patch `forward` — PyTorch nn.Module.__call__ ignores instance-level patches
        with patch.object(framework.model, "forward", side_effect=RuntimeError("oops")):
            result = framework.generate_hedging_decision(np.zeros(11), 42.0)
        assert result.get("recommended_position") == 42.0
        assert "error" in result


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_var_positive(self, framework: DeepHedgingFramework) -> None:
        state = np.zeros(11)
        state[10] = 0.3
        m = framework._calculate_risk_metrics(state, 0.5)
        assert m["var_95"] > 0

    def test_sharpe_ratio(self, framework: DeepHedgingFramework) -> None:
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
        hedge_ratios = [0.5] * len(returns)
        sr = framework._calculate_sharpe_ratio(returns, hedge_ratios)
        assert np.isfinite(sr)

    def test_sharpe_ratio_empty(self, framework: DeepHedgingFramework) -> None:
        assert framework._calculate_sharpe_ratio(np.array([]), []) == 0.0

    def test_max_drawdown_nonneg(self, framework: DeepHedgingFramework) -> None:
        returns = np.array([0.05, -0.1, 0.02, -0.15, 0.08])
        hedge_ratios = [0.3] * len(returns)
        dd = framework._calculate_max_drawdown(returns, hedge_ratios)
        assert dd >= 0.0

    def test_max_drawdown_empty(self, framework: DeepHedgingFramework) -> None:
        assert framework._calculate_max_drawdown(np.array([]), []) == 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(
        self,
        framework: DeepHedgingFramework,
        tmp_path: Any,
    ) -> None:
        price_data = 100.0 + np.cumsum(np.random.randn(15) * 0.5)
        payoffs = np.maximum(price_data - 100.0, 0.0)
        framework.train_hedging_strategy(price_data, payoffs)

        path = str(tmp_path / "model.pt")
        assert framework.save_model(path) is True

        fw2 = DeepHedgingFramework({"input_dim": 11, "hidden_dim": 16, "output_dim": 1, "epochs": 2, "batch_size": 4})
        assert fw2.load_model(path) is True
        assert fw2.config["input_dim"] == 11

    def test_load_nonexistent(self, framework: DeepHedgingFramework) -> None:
        assert framework.load_model("/nonexistent/path.pt") is False

    def test_save_to_bad_path(self, framework: DeepHedgingFramework) -> None:
        assert framework.save_model("/root/no_permission/model.pt") is False
