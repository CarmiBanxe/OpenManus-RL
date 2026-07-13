"""
Tests — SignatureMethods (Sprint 3).
iisignature может отсутствовать — все тесты работают с fallback.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from openmanus_rl.decision.signature_methods import SignatureMethods


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sm() -> SignatureMethods:
    return SignatureMethods({"signature_depth": 2, "path_dimension": 2, "time_points": 10})


@pytest.fixture()
def price_path() -> np.ndarray:
    np.random.seed(0)
    return 100.0 + np.cumsum(np.random.randn(20) * 0.5)


@pytest.fixture()
def vol_path(price_path: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(price_path, prepend=price_path[0])) + 0.01


@pytest.fixture()
def decision_history() -> List[Dict[str, Any]]:
    return [
        {"action": "buy", "confidence": 0.8, "outcome": 1.0},
        {"action": "sell", "confidence": 0.6, "outcome": -0.5},
        {"action": "hold", "confidence": 0.5, "outcome": 0.0},
        {"action": "buy", "confidence": 0.9, "outcome": 0.8},
        {"action": "sell", "confidence": 0.7, "outcome": -0.2},
        {"action": "buy", "confidence": 0.85, "outcome": 1.2},
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSignatureMethodsInit:
    def test_defaults(self) -> None:
        s = SignatureMethods()
        assert s.signature_depth == 3
        assert s.path_dimension == 2
        assert s.time_points == 100

    def test_custom_config(self, sm: SignatureMethods) -> None:
        assert sm.signature_depth == 2

    def test_cache_empty_on_init(self, sm: SignatureMethods) -> None:
        assert len(sm.signature_cache) == 0


# ---------------------------------------------------------------------------
# _preprocess_path
# ---------------------------------------------------------------------------


class TestPreprocessPath:
    def test_1d_to_2d(self, sm: SignatureMethods) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        out = sm._preprocess_path(arr)
        assert out.ndim == 2
        assert out.shape[1] == 1

    def test_zero_std_column(self, sm: SignatureMethods) -> None:
        arr = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        out = sm._preprocess_path(arr)
        # Column 0 has std=0 → subtract mean (→ zeros)
        np.testing.assert_allclose(out[:, 0], 0.0)

    def test_normalized_column(self, sm: SignatureMethods) -> None:
        arr = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        out = sm._preprocess_path(arr)
        # Column 0: (0,1,2) → z-score mean≈0
        assert abs(np.mean(out[:, 0])) < 1e-6

    def test_float64_output(self, sm: SignatureMethods) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        out = sm._preprocess_path(arr)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# compute_path_signature
# ---------------------------------------------------------------------------


class TestComputePathSignature:
    def test_returns_ndarray(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        sig = sm.compute_path_signature(price_path.reshape(-1, 1))
        assert isinstance(sig, np.ndarray)

    def test_not_empty(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        sig = sm.compute_path_signature(price_path.reshape(-1, 1))
        assert len(sig) > 0

    def test_cached_result(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        path = price_path.reshape(-1, 1)
        sig1 = sm.compute_path_signature(path)
        assert len(sm.signature_cache) == 1
        sig2 = sm.compute_path_signature(path)
        np.testing.assert_array_equal(sig1, sig2)

    def test_depth_override(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        sig1 = sm.compute_path_signature(price_path.reshape(-1, 1), depth=1)
        sig2 = sm.compute_path_signature(price_path.reshape(-1, 1), depth=2)
        # Different depths produce different cache entries
        assert len(sm.signature_cache) == 2

    def test_2d_path(self, sm: SignatureMethods) -> None:
        path = np.column_stack([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
        sig = sm.compute_path_signature(path)
        assert isinstance(sig, np.ndarray)

    def test_single_point_path(self, sm: SignatureMethods) -> None:
        # Edge case: single-row path
        path = np.array([[1.0, 2.0]])
        sig = sm.compute_path_signature(path)
        assert isinstance(sig, np.ndarray)


# ---------------------------------------------------------------------------
# compute_rough_volatility_signature
# ---------------------------------------------------------------------------


class TestRoughVolatilitySignature:
    def test_returns_array(
        self,
        sm: SignatureMethods,
        price_path: np.ndarray,
        vol_path: np.ndarray,
    ) -> None:
        sig = sm.compute_rough_volatility_signature(price_path, vol_path)
        assert isinstance(sig, np.ndarray)

    def test_length_mismatch_handled(self, sm: SignatureMethods) -> None:
        prices = np.array([1.0, 2.0, 3.0])
        vols = np.array([0.1, 0.2])
        # np.column_stack will raise → caught → returns np.zeros(1)
        sig = sm.compute_rough_volatility_signature(prices, vols)
        assert isinstance(sig, np.ndarray)


# ---------------------------------------------------------------------------
# compute_decision_history_signature
# ---------------------------------------------------------------------------


class TestDecisionHistorySignature:
    def test_returns_array(
        self, sm: SignatureMethods, decision_history: List[Dict[str, Any]]
    ) -> None:
        sig = sm.compute_decision_history_signature(decision_history)
        assert isinstance(sig, np.ndarray)

    def test_empty_history(self, sm: SignatureMethods) -> None:
        sig = sm.compute_decision_history_signature([])
        assert isinstance(sig, np.ndarray)

    def test_single_decision(self, sm: SignatureMethods) -> None:
        sig = sm.compute_decision_history_signature(
            [{"action": "buy", "confidence": 0.9, "outcome": 1.0}]
        )
        assert isinstance(sig, np.ndarray)


# ---------------------------------------------------------------------------
# _extract_decision_features
# ---------------------------------------------------------------------------


class TestExtractDecisionFeatures:
    def test_shape(
        self, sm: SignatureMethods, decision_history: List[Dict[str, Any]]
    ) -> None:
        feats = sm._extract_decision_features(decision_history)
        assert feats.shape == (len(decision_history), 3)

    def test_buy_code(self, sm: SignatureMethods) -> None:
        history = [{"action": "buy", "confidence": 1.0, "outcome": 0.0}]
        feats = sm._extract_decision_features(history)
        assert feats[0, 0] == 1.0

    def test_sell_code(self, sm: SignatureMethods) -> None:
        history = [{"action": "sell", "confidence": 1.0, "outcome": 0.0}]
        feats = sm._extract_decision_features(history)
        assert feats[0, 0] == -1.0

    def test_hold_code(self, sm: SignatureMethods) -> None:
        history = [{"action": "hold", "confidence": 0.5, "outcome": 0.0}]
        feats = sm._extract_decision_features(history)
        assert feats[0, 0] == 0.0

    def test_empty_returns_zeros(self, sm: SignatureMethods) -> None:
        feats = sm._extract_decision_features([])
        assert feats.shape == (1, 3)


# ---------------------------------------------------------------------------
# _extract_signature_features
# ---------------------------------------------------------------------------


class TestExtractSignatureFeatures:
    def test_basic_stats(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        sig = sm.compute_path_signature(price_path.reshape(-1, 1))
        feats = sm._extract_signature_features(sig)
        for key in ("mean", "std", "min", "max", "length"):
            assert key in feats

    def test_length_correct(self, sm: SignatureMethods) -> None:
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feats = sm._extract_signature_features(sig)
        assert feats["length"] == 5

    def test_depth_features_present(self, sm: SignatureMethods) -> None:
        # depth_features only when chunk fits
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        feats = sm._extract_signature_features(sig)
        assert "depth_features" in feats

    def test_error_returns_error_key(self, sm: SignatureMethods) -> None:
        feats = sm._extract_signature_features(None)  # type: ignore[arg-type]
        assert "error" in feats


# ---------------------------------------------------------------------------
# analyze_path_dependent_option
# ---------------------------------------------------------------------------


class TestAnalyzePathDependentOption:
    def test_asian_call(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "asian", "strike": 100.0, "option_type": "call"}
        )
        assert "payoff" in result
        assert result["payoff"] >= 0.0
        assert result["option_type"] == "asian"

    def test_asian_put(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "asian", "strike": 100.0, "option_type": "put"}
        )
        assert result["payoff"] >= 0.0

    def test_barrier_up_call(self, sm: SignatureMethods) -> None:
        prices = np.array([90.0, 95.0, 105.0, 98.0, 92.0])
        result = sm.analyze_path_dependent_option(
            prices,
            {
                "type": "barrier",
                "strike": 90.0,
                "barrier": 100.0,
                "barrier_type": "up",
                "option_type": "call",
            },
        )
        assert result["barrier_hit"] is True
        assert result["payoff"] >= 0.0

    def test_barrier_not_hit(self, sm: SignatureMethods) -> None:
        prices = np.array([80.0, 85.0, 88.0, 84.0, 82.0])
        result = sm.analyze_path_dependent_option(
            prices,
            {
                "type": "barrier",
                "strike": 80.0,
                "barrier": 100.0,
                "barrier_type": "up",
                "option_type": "call",
            },
        )
        assert result["barrier_hit"] is False
        assert result["payoff"] == 0.0

    def test_lookback_call(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "lookback", "option_type": "call"}
        )
        assert result["payoff"] >= 0.0
        assert "max_price" in result
        assert "min_price" in result

    def test_lookback_put(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "lookback", "option_type": "put"}
        )
        assert result["payoff"] >= 0.0

    def test_unknown_option_type(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "exotic_unknown"}
        )
        assert "error" in result

    def test_signature_in_result(self, sm: SignatureMethods, price_path: np.ndarray) -> None:
        result = sm.analyze_path_dependent_option(
            price_path, {"type": "asian"}
        )
        assert "signature" in result
        assert isinstance(result["signature"], np.ndarray)

    def test_signature_features_in_result(
        self, sm: SignatureMethods, price_path: np.ndarray
    ) -> None:
        result = sm.analyze_path_dependent_option(price_path, {"type": "asian"})
        assert "signature_features" in result


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_key_differs_for_different_depth(
        self, sm: SignatureMethods
    ) -> None:
        path = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k1 = sm._signature_cache_key(path, 1)
        k2 = sm._signature_cache_key(path, 2)
        assert k1 != k2

    def test_cache_key_same_for_same_input(self, sm: SignatureMethods) -> None:
        path = np.array([[1.0, 2.0], [3.0, 4.0]])
        k1 = sm._signature_cache_key(path, 2)
        k2 = sm._signature_cache_key(path, 2)
        assert k1 == k2

    def test_cache_grows(self, sm: SignatureMethods) -> None:
        p1 = np.array([[1.0], [2.0], [3.0]])
        p2 = np.array([[4.0], [5.0], [6.0]])
        sm.compute_path_signature(p1)
        sm.compute_path_signature(p2)
        assert len(sm.signature_cache) == 2
