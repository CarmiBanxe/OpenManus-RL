"""
Deep Hedging Framework для управления рисками.
Sprint 3 | SANDBOX ONLY — no live banking data, no real positions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class DeepHedgingNetwork(nn.Module):
    """Нейронная сеть для Deep Hedging (4-layer MLP с Dropout)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------


class DeepHedgingFramework:
    """Фреймворк Deep Hedging для управления рисками."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        self.input_dim: int = self.config.get("input_dim", 10)
        self.hidden_dim: int = self.config.get("hidden_dim", 64)
        self.output_dim: int = self.config.get("output_dim", 1)
        self.learning_rate: float = self.config.get("learning_rate", 0.001)
        self.epochs: int = self.config.get("epochs", 100)
        self.batch_size: int = self.config.get("batch_size", 32)
        self.risk_aversion: float = self.config.get("risk_aversion", 0.5)
        self.transaction_cost: float = self.config.get("transaction_cost", 0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepHedgingNetwork(
            self.input_dim, self.hidden_dim, self.output_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.training_history: List[Dict[str, Any]] = []
        logger.info("DeepHedgingFramework initialized")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_hedging_strategy(
        self,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Обучение хеджирующей стратегии."""
        try:
            if features is None:
                features = self._generate_features(price_data)

            dataset = self._create_dataset(price_data, option_payoffs, features)
            training_history: List[Dict[str, Any]] = []

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0

                for i in range(0, len(dataset), self.batch_size):
                    batch = dataset[i : i + self.batch_size]
                    inputs, targets = self._prepare_batch(batch)
                    loss = self._train_batch(inputs, targets)
                    epoch_loss += loss
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                training_history.append({"epoch": epoch, "loss": avg_loss})

                if epoch % 10 == 0:
                    logger.info("Epoch %d, Loss: %.4f", epoch, avg_loss)

            self.training_history = training_history
            evaluation = self._evaluate_hedging_strategy(
                price_data, option_payoffs, features
            )

            return {
                "success": True,
                "training_history": training_history,
                "evaluation": evaluation,
                "model_parameters": self._get_model_parameters(),
            }

        except Exception as exc:
            logger.error("Deep hedging training error: %s", exc)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_hedging_decision(
        self,
        current_state: np.ndarray,
        portfolio_position: float,
    ) -> Dict[str, Any]:
        """Генерация хеджирующего решения."""
        try:
            input_tensor = (
                torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                hedge_ratio: float = self.model(input_tensor).item()

            recommended_position = hedge_ratio * float(current_state[0])
            transaction_amount = recommended_position - portfolio_position
            transaction_cost = abs(transaction_amount) * self.transaction_cost

            return {
                "current_position": portfolio_position,
                "recommended_position": recommended_position,
                "hedge_ratio": hedge_ratio,
                "transaction_amount": transaction_amount,
                "transaction_cost": transaction_cost,
                "risk_metrics": self._calculate_risk_metrics(
                    current_state, hedge_ratio
                ),
                "confidence": self._calculate_confidence(current_state),
            }

        except Exception as exc:
            logger.error("Hedging decision generation error: %s", exc)
            return {
                "error": str(exc),
                "recommended_position": portfolio_position,
                "hedge_ratio": 0.0,
            }

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _generate_features(self, price_data: np.ndarray) -> np.ndarray:
        """Генерация признаков из ценовых данных."""
        try:
            features: List[List[float]] = []

            for i in range(len(price_data)):
                if i >= 5:
                    window = price_data[i - 5 : i + 1]
                    returns = np.diff(window) / window[:-1]

                    gains = returns[returns > 0]
                    losses = -returns[returns < 0]

                    if len(gains) > 0 and len(losses) > 0:
                        rs = np.mean(gains) / np.mean(losses)
                        rsi = 100.0 - (100.0 / (1.0 + rs))
                    else:
                        rsi = 50.0

                    features.append(
                        [
                            float(price_data[i]),
                            float(np.mean(window)),
                            float(np.std(window)),
                            float(np.min(window)),
                            float(np.max(window)),
                            float(np.mean(returns)),
                            float(np.std(returns)),
                            float(np.mean(window)),        # sma_5
                            float(np.mean(window[-3:])),   # sma_3
                            rsi,
                            float(np.std(returns)),        # volatility
                        ]
                    )
                else:
                    features.append([float(price_data[i])] + [0.0] * (self.input_dim - 1))

            return np.array(features)

        except Exception as exc:
            logger.error("Feature generation error: %s", exc)
            return np.zeros((len(price_data), self.input_dim))

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _create_dataset(
        self,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
        features: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        dataset: List[Tuple[np.ndarray, float]] = []
        for i in range(len(price_data) - 1):
            price_change = (price_data[i + 1] - price_data[i]) / price_data[i]
            optimal_hedge = -price_change * self.risk_aversion
            dataset.append((features[i], float(optimal_hedge)))
        return dataset

    def _prepare_batch(
        self, batch: List[Tuple[np.ndarray, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.FloatTensor(np.array([item[0] for item in batch])).to(self.device)
        targets = (
            torch.FloatTensor([item[1] for item in batch]).unsqueeze(1).to(self.device)
        )
        return inputs, targets

    def _train_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_hedging_strategy(
        self,
        price_data: np.ndarray,
        option_payoffs: np.ndarray,
        features: np.ndarray,
    ) -> Dict[str, Any]:
        try:
            portfolio_value = 0.0
            hedging_costs = 0.0
            hedge_ratios: List[float] = []

            for i in range(len(price_data) - 1):
                decision = self.generate_hedging_decision(features[i], portfolio_value)
                portfolio_value += option_payoffs[i] - decision["transaction_cost"]
                hedging_costs += decision["transaction_cost"]
                hedge_ratios.append(decision["hedge_ratio"])

            return {
                "final_pnl": portfolio_value,
                "total_cost": hedging_costs,
                "sharpe_ratio": self._calculate_sharpe_ratio(option_payoffs, hedge_ratios),
                "max_drawdown": self._calculate_max_drawdown(option_payoffs, hedge_ratios),
                "avg_hedge_ratio": float(np.mean(hedge_ratios)) if hedge_ratios else 0.0,
                "hedge_ratio_std": float(np.std(hedge_ratios)) if hedge_ratios else 0.0,
            }

        except Exception as exc:
            logger.error("Hedging strategy evaluation error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def _calculate_risk_metrics(
        self, current_state: np.ndarray, hedge_ratio: float
    ) -> Dict[str, Any]:
        try:
            volatility = float(current_state[10]) if len(current_state) >= 11 else 0.2
            return {
                "volatility": volatility,
                "var_95": 1.96 * volatility,
                "es_95": 2.33 * volatility,
                "risk_adjusted_return": -hedge_ratio * volatility * self.risk_aversion,
            }
        except Exception as exc:
            logger.error("Risk metrics calculation error: %s", exc)
            return {"error": str(exc)}

    def _calculate_confidence(self, current_state: np.ndarray) -> float:
        try:
            if len(current_state) >= 11:
                volatility = float(current_state[10])
                return max(0.1, 1.0 - min(1.0, volatility * 5.0))
            return 0.5
        except Exception as exc:
            logger.error("Confidence calculation error: %s", exc)
            return 0.5

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, hedge_ratios: List[float]
    ) -> float:
        try:
            if len(returns) < 2:
                return 0.0
            hedged = [
                returns[i] - hedge_ratios[i] * returns[i]
                for i in range(min(len(returns), len(hedge_ratios)))
            ]
            if len(hedged) < 2:
                return 0.0
            std = float(np.std(hedged))
            return float(np.mean(hedged)) / std if std > 0 else 0.0
        except Exception as exc:
            logger.error("Sharpe ratio calculation error: %s", exc)
            return 0.0

    def _calculate_max_drawdown(
        self, returns: np.ndarray, hedge_ratios: List[float]
    ) -> float:
        try:
            if len(returns) < 2:
                return 0.0
            hedged = np.array(
                [
                    returns[i] - hedge_ratios[i] * returns[i]
                    for i in range(min(len(returns), len(hedge_ratios)))
                ]
            )
            if len(hedged) < 2:
                return 0.0
            cumulative = np.cumprod(1.0 + hedged)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / np.where(peak != 0, peak, 1.0)
            return float(-np.min(drawdown))
        except Exception as exc:
            logger.error("Max drawdown calculation error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_model_parameters(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "risk_aversion": self.risk_aversion,
            "transaction_cost": self.transaction_cost,
        }

    def save_model(self, filepath: str) -> bool:
        try:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "training_history": self.training_history,
                },
                filepath,
            )
            logger.info("Model saved to %s", filepath)
            return True
        except Exception as exc:
            logger.error("Model saving error: %s", exc)
            return False

    def load_model(self, filepath: str) -> bool:
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.config = checkpoint["config"]
            self.training_history = checkpoint["training_history"]
            logger.info("Model loaded from %s", filepath)
            return True
        except Exception as exc:
            logger.error("Model loading error: %s", exc)
            return False
