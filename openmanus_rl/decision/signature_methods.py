"""
Signature Methods для учёта истории принятия решений.
Sprint 3 | Path signatures via iisignature (Chen iterated integrals).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import iisignature  # type: ignore[import-not-found]

    _IISIGNATURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IISIGNATURE_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning("iisignature not installed — SignatureMethods will use fallback zeros.")

logger = logging.getLogger(__name__)


class SignatureMethods:
    """Реализация Signature Methods для учёта истории принятия решений."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.signature_depth: int = self.config.get("signature_depth", 3)
        self.path_dimension: int = self.config.get("path_dimension", 2)
        self.time_points: int = self.config.get("time_points", 100)

        self.signature_cache: Dict[str, np.ndarray] = {}
        logger.info("SignatureMethods initialized (iisignature=%s)", _IISIGNATURE_AVAILABLE)

    # ------------------------------------------------------------------
    # Public — path signature
    # ------------------------------------------------------------------

    def compute_path_signature(
        self,
        path: np.ndarray,
        depth: Optional[int] = None,
    ) -> np.ndarray:
        """
        Вычисление сигнатуры пути (Chen iterated integrals).

        Args:
            path: Временной ряд размерности (n, d).
            depth: Глубина сигнатуры (по умолчанию self.signature_depth).

        Returns:
            1-D numpy array — сигнатура пути.
        """
        if depth is None:
            depth = self.signature_depth

        try:
            cache_key = self._signature_cache_key(path, depth)
            if cache_key in self.signature_cache:
                return self.signature_cache[cache_key]

            processed = self._preprocess_path(path)

            if _IISIGNATURE_AVAILABLE:
                signature: np.ndarray = iisignature.sig(processed, depth)
            else:
                # Fallback: first-order differences as a minimal proxy
                signature = np.diff(processed, axis=0).flatten()

            self.signature_cache[cache_key] = signature
            return signature

        except Exception as exc:
            logger.error("Path signature computation error: %s", exc)
            return np.zeros(1)

    def compute_rough_volatility_signature(
        self,
        price_data: np.ndarray,
        volatility_data: np.ndarray,
        depth: Optional[int] = None,
    ) -> np.ndarray:
        """Сигнатура двумерного пути (цена, волатильность)."""
        try:
            path = np.column_stack((price_data, volatility_data))
            return self.compute_path_signature(path, depth)
        except Exception as exc:
            logger.error("Rough volatility signature error: %s", exc)
            return np.zeros(1)

    def compute_decision_history_signature(
        self,
        decision_history: List[Dict[str, Any]],
        depth: Optional[int] = None,
    ) -> np.ndarray:
        """Сигнатура пути, построенного из истории решений."""
        try:
            path_features = self._extract_decision_features(decision_history)
            return self.compute_path_signature(path_features, depth)
        except Exception as exc:
            logger.error("Decision history signature error: %s", exc)
            return np.zeros(1)

    # ------------------------------------------------------------------
    # Public — option analysis
    # ------------------------------------------------------------------

    def analyze_path_dependent_option(
        self,
        price_data: np.ndarray,
        option_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Анализ path-dependent опциона с использованием сигнатур."""
        try:
            path_signature = self.compute_path_signature(price_data.reshape(-1, 1))
            option_type = option_params.get("type", "asian")

            if option_type == "asian":
                return self._analyze_asian_option(price_data, option_params, path_signature)
            if option_type == "barrier":
                return self._analyze_barrier_option(price_data, option_params, path_signature)
            if option_type == "lookback":
                return self._analyze_lookback_option(price_data, option_params, path_signature)
            return {"error": f"Unknown option type: {option_type}", "signature": path_signature}

        except Exception as exc:
            logger.error("Path-dependent option analysis error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Private — path preprocessing
    # ------------------------------------------------------------------

    def _preprocess_path(self, path: np.ndarray) -> np.ndarray:
        """Z-score нормализация по каждой размерности."""
        try:
            if path.ndim == 1:
                path = path.reshape(-1, 1)
            normalized = np.zeros_like(path, dtype=np.float64)
            for i in range(path.shape[1]):
                col = path[:, i]
                std = np.std(col)
                normalized[:, i] = (col - np.mean(col)) / std if std > 0 else col - np.mean(col)
            return normalized
        except Exception as exc:
            logger.error("Path preprocessing error: %s", exc)
            return path.astype(np.float64)

    # ------------------------------------------------------------------
    # Private — feature extraction
    # ------------------------------------------------------------------

    def _extract_decision_features(
        self, decision_history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Конвертирует историю решений в числовой путь (action, confidence, outcome)."""
        if not decision_history:
            return np.zeros((1, 3))
        try:
            rows: List[List[float]] = []
            for d in decision_history:
                action = d.get("action", "unknown")
                code = 1.0 if action == "buy" else (-1.0 if action == "sell" else 0.0)
                rows.append([code, float(d.get("confidence", 0.5)), float(d.get("outcome", 0.0))])
            return np.array(rows, dtype=np.float64)
        except Exception as exc:
            logger.error("Decision features extraction error: %s", exc)
            return np.zeros((1, 3))

    def _extract_signature_features(self, signature: np.ndarray) -> Dict[str, Any]:
        """Базовая статистика + покомпонентный анализ по глубинам."""
        try:
            features: Dict[str, Any] = {
                "mean": float(np.mean(signature)),
                "std": float(np.std(signature)),
                "min": float(np.min(signature)),
                "max": float(np.max(signature)),
                "length": int(len(signature)),
            }
            depth_features: Dict[str, Dict[str, float]] = {}
            start = 0
            for d in range(1, self.signature_depth + 1):
                n = self.path_dimension**d
                if start + n <= len(signature):
                    chunk = signature[start : start + n]
                    depth_features[f"depth_{d}"] = {
                        "mean": float(np.mean(chunk)),
                        "std": float(np.std(chunk)),
                        "norm": float(np.linalg.norm(chunk)),
                    }
                    start += n
            features["depth_features"] = depth_features
            return features
        except Exception as exc:
            logger.error("Signature features extraction error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Private — option analysers
    # ------------------------------------------------------------------

    def _analyze_asian_option(
        self,
        price_data: np.ndarray,
        params: Dict[str, Any],
        path_signature: np.ndarray,
    ) -> Dict[str, Any]:
        try:
            strike = float(params.get("strike", np.mean(price_data)))
            call = params.get("option_type", "call") == "call"
            avg_price = float(np.mean(price_data))
            payoff = max(0.0, avg_price - strike) if call else max(0.0, strike - avg_price)
            return {
                "option_type": "asian",
                "strike": strike,
                "average_price": avg_price,
                "payoff": payoff,
                "signature": path_signature,
                "signature_features": self._extract_signature_features(path_signature),
            }
        except Exception as exc:
            logger.error("Asian option analysis error: %s", exc)
            return {"error": str(exc)}

    def _analyze_barrier_option(
        self,
        price_data: np.ndarray,
        params: Dict[str, Any],
        path_signature: np.ndarray,
    ) -> Dict[str, Any]:
        try:
            strike = float(params.get("strike", price_data[0]))
            barrier = float(params.get("barrier", strike * 1.2))
            barrier_type = params.get("barrier_type", "up")
            call = params.get("option_type", "call") == "call"
            barrier_hit = bool(
                np.any(price_data >= barrier)
                if barrier_type == "up"
                else np.any(price_data <= barrier)
            )
            final = float(price_data[-1])
            payoff = 0.0
            if barrier_hit:
                payoff = max(0.0, final - strike) if call else max(0.0, strike - final)
            return {
                "option_type": "barrier",
                "strike": strike,
                "barrier": barrier,
                "barrier_type": barrier_type,
                "barrier_hit": barrier_hit,
                "final_price": final,
                "payoff": payoff,
                "signature": path_signature,
                "signature_features": self._extract_signature_features(path_signature),
            }
        except Exception as exc:
            logger.error("Barrier option analysis error: %s", exc)
            return {"error": str(exc)}

    def _analyze_lookback_option(
        self,
        price_data: np.ndarray,
        params: Dict[str, Any],
        path_signature: np.ndarray,
    ) -> Dict[str, Any]:
        try:
            call = params.get("option_type", "call") == "call"
            max_price = float(np.max(price_data))
            min_price = float(np.min(price_data))
            final = float(price_data[-1])
            payoff = max(0.0, final - min_price) if call else max(0.0, max_price - final)
            return {
                "option_type": "lookback",
                "max_price": max_price,
                "min_price": min_price,
                "final_price": final,
                "payoff": payoff,
                "signature": path_signature,
                "signature_features": self._extract_signature_features(path_signature),
            }
        except Exception as exc:
            logger.error("Lookback option analysis error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Private — cache key
    # ------------------------------------------------------------------

    def _signature_cache_key(self, path: np.ndarray, depth: int) -> str:
        digest = hashlib.md5(path.tobytes() + str(depth).encode()).hexdigest()  # noqa: S324
        return digest
