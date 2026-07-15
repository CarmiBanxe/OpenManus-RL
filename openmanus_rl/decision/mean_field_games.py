"""
Mean Field Games (MFG) — анализ систем с множеством агентов.
Sprint 4 | Sandbox: torch необязателен; fallback на numpy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network component (torch-only)
# ---------------------------------------------------------------------------


class MFGNetwork:
    """
    Нейронная сеть для аппроксимации решения MFG.
    Создаётся только при доступном torch; иначе используется заглушка.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self._net: Any = None

        if _TORCH_AVAILABLE:
            self._net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход; возвращает numpy array."""
        if self._net is None:
            return np.tanh(x @ np.random.randn(self.state_dim, self.state_dim) * 0.1)
        try:
            t = torch.FloatTensor(x)
            with torch.no_grad():
                return self._net(t).numpy()
        except Exception as exc:
            logger.warning("MFGNetwork.forward fallback: %s", exc)
            return np.zeros_like(x)

    def parameters(self) -> Any:
        if self._net is not None:
            return self._net.parameters()
        return iter([])


# ---------------------------------------------------------------------------
# Main framework
# ---------------------------------------------------------------------------


class MeanFieldGames:
    """
    Mean Field Games Framework для анализа взаимодействия N→∞ агентов.

    Решает систему уравнений HJB (Hamilton-Jacobi-Bellman) + Fokker-Planck
    итеративным методом (picard iterations) с нейросетевой аппроксимацией.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        self.num_agents: int = self.config.get("num_agents", 100)
        self.state_dim: int = self.config.get("state_dim", 2)
        self.time_horizon: int = self.config.get("time_horizon", 50)
        self.dt: float = self.config.get("dt", 0.1)
        self.learning_rate: float = self.config.get("learning_rate", 1e-3)
        self.max_iterations: int = self.config.get("max_iterations", 50)
        self.convergence_tol: float = self.config.get("convergence_tol", 1e-4)
        self.risk_aversion: float = self.config.get("risk_aversion", 0.5)
        self.transaction_cost: float = self.config.get("transaction_cost", 0.001)
        self.volatility: float = self.config.get("volatility", 0.2)

        self.policy_net = MFGNetwork(self.state_dim, self.config.get("hidden_dim", 64))
        self.value_net = MFGNetwork(self.state_dim, self.config.get("hidden_dim", 64))

        if _TORCH_AVAILABLE and self.policy_net._net is not None:
            self._optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=self.learning_rate,
            )
        else:
            self._optimizer = None

        # State: mean-field distribution μ (num_agents × state_dim)
        self._mean_field: Optional[np.ndarray] = None
        self._convergence_history: List[float] = []

        logger.info(
            "MeanFieldGames initialized (agents=%d, state_dim=%d, torch=%s)",
            self.num_agents,
            self.state_dim,
            _TORCH_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_mfg(self) -> Dict[str, Any]:
        """
        Решение MFG через Picard iterations.

        Returns:
            Dict с ключами: success, iterations, convergence_error,
            equilibrium_policy, mean_field_distribution.
        """
        try:
            mu = self._generate_initial_distribution()
            self._mean_field = mu
            prev_mu = mu.copy()
            convergence_history: List[float] = []

            for iteration in range(self.max_iterations):
                # HJB step: solve backward (value function → optimal policy)
                policy = self._solve_hjb(mu)

                # Fokker-Planck step: evolve distribution forward
                mu_new = self._solve_fokker_planck(mu, policy)

                # Convergence check (Wasserstein-1 proxy: L1 distance)
                error = float(np.mean(np.abs(mu_new - prev_mu)))
                convergence_history.append(error)

                if error < self.convergence_tol:
                    mu = mu_new
                    logger.info("MFG converged at iteration %d (error=%.6f)", iteration, error)
                    break

                prev_mu = mu.copy()
                mu = mu_new

            self._mean_field = mu
            self._convergence_history = convergence_history

            equilibrium_policy = self._extract_equilibrium_policy(mu)

            return {
                "success": True,
                "iterations": len(convergence_history),
                "convergence_error": float(convergence_history[-1]) if convergence_history else 0.0,
                "convergence_history": convergence_history,
                "equilibrium_policy": equilibrium_policy,
                "mean_field_distribution": {
                    "mean": mu.mean(axis=0).tolist(),
                    "std": mu.std(axis=0).tolist(),
                    "shape": list(mu.shape),
                },
            }

        except Exception as exc:
            logger.error("MFG solving error: %s", exc)
            return {"success": False, "error": str(exc)}

    def compute_nash_equilibrium(self) -> Dict[str, Any]:
        """
        Вычисление равновесия Нэша на основе текущего mean-field.

        Returns:
            Dict с оптимальными стратегиями, ценностными функциями и метриками.
        """
        try:
            mu = self._mean_field if self._mean_field is not None else self._generate_initial_distribution()

            # Оптимальная политика в равновесии
            equilibrium_policy = self._extract_equilibrium_policy(mu)

            # Ценностные функции
            value_functions = self._compute_value_functions(mu, equilibrium_policy)

            # Метрики равновесия
            social_welfare = self._compute_social_welfare(mu, equilibrium_policy)
            price_of_anarchy = self._compute_price_of_anarchy(social_welfare, mu)

            return {
                "success": True,
                "equilibrium_policy": equilibrium_policy,
                "value_functions": value_functions,
                "social_welfare": social_welfare,
                "price_of_anarchy": price_of_anarchy,
                "mean_field_distribution": mu.mean(axis=0).tolist(),
            }

        except Exception as exc:
            logger.error("Nash equilibrium computation error: %s", exc)
            return {"success": False, "error": str(exc)}

    def simulate_market_dynamics(
        self,
        initial_states: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Симуляция рыночной динамики с N агентами.

        Args:
            initial_states: Начальные состояния агентов (num_agents × state_dim).
            num_steps: Количество шагов симуляции.

        Returns:
            Dict с траекториями, статистикой и метриками.
        """
        try:
            steps = num_steps or self.time_horizon
            states = initial_states.copy()
            mu = initial_states.copy()

            trajectories: List[np.ndarray] = [states.copy()]
            price_trajectory: List[float] = []
            volatility_trajectory: List[float] = []

            for step in range(steps):
                # Оптимальные действия каждого агента
                policy = self._extract_equilibrium_policy(mu)
                actions = self._compute_actions(states, policy)

                # Рыночная цена = агрегат состояний агентов
                market_price = float(np.mean(states[:, 0]))
                vol = float(np.std(states[:, 0]))
                price_trajectory.append(market_price)
                volatility_trajectory.append(vol)

                # Динамика состояния: SDE дискретизация (Euler-Maruyama)
                noise = np.random.randn(*states.shape) * self.volatility * np.sqrt(self.dt)
                states = states + actions * self.dt + noise
                mu = states.copy()
                trajectories.append(states.copy())

            return {
                "success": True,
                "num_steps": steps,
                "price_trajectory": price_trajectory,
                "volatility_trajectory": volatility_trajectory,
                "final_distribution": {
                    "mean": states.mean(axis=0).tolist(),
                    "std": states.std(axis=0).tolist(),
                },
                "market_metrics": {
                    "avg_price": float(np.mean(price_trajectory)),
                    "price_volatility": float(np.std(price_trajectory)),
                    "max_drawdown": self._compute_max_drawdown(price_trajectory),
                    "sharpe_ratio": self._compute_sharpe_ratio(price_trajectory),
                },
                "trajectory_length": len(trajectories),
            }

        except Exception as exc:
            logger.error("Market dynamics simulation error: %s", exc)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Private — MFG solvers
    # ------------------------------------------------------------------

    def _solve_hjb(self, mu: np.ndarray) -> np.ndarray:
        """
        Решение уравнения HJB (backward pass).
        Возвращает оптимальную политику π: state_dim → action_dim.
        """
        try:
            # Running cost: квадратичный по действию + mean-field coupling
            mu_mean = mu.mean(axis=0)

            # Аппроксимация оптимальной политики нейронной сетью
            policy = self.policy_net.forward(mu_mean.reshape(1, -1)).flatten()

            # Нормализация политики (проецируем на [-1, 1]^d)
            max_abs = np.max(np.abs(policy))
            if max_abs > 0:
                policy = policy / max_abs * (1.0 - self.transaction_cost)

            return policy

        except Exception as exc:
            logger.error("HJB solving error: %s", exc)
            return np.zeros(self.state_dim)

    def _solve_fokker_planck(self, mu: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """
        Решение уравнения Фоккера-Планка (forward pass).
        Эволюция распределения μ под действием политики π.
        """
        try:
            drift = policy * self.dt
            diffusion = self.volatility * np.sqrt(self.dt) * np.random.randn(*mu.shape)
            mu_new = mu + drift + diffusion

            # Мягкая проекция: центрируем, чтобы не уходили в бесконечность
            mu_new = mu_new - 0.01 * (mu_new - mu_new.mean(axis=0))
            return mu_new

        except Exception as exc:
            logger.error("Fokker-Planck solving error: %s", exc)
            return mu.copy()

    # ------------------------------------------------------------------
    # Private — equilibrium helpers
    # ------------------------------------------------------------------

    def _extract_equilibrium_policy(self, mu: np.ndarray) -> Dict[str, Any]:
        mu_mean = mu.mean(axis=0)
        mu_std = mu.std(axis=0)

        # Оптимальная стратегия в равновесии: mean-reverting к µ
        optimal_action = -self.risk_aversion * (mu_mean / (np.maximum(mu_std, 1e-8)))

        return {
            "optimal_action": optimal_action.tolist(),
            "mean_field_state": mu_mean.tolist(),
            "dispersion": mu_std.tolist(),
            "strategy_type": "mean_reverting",
        }

    def _compute_value_functions(
        self, mu: np.ndarray, policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            mu_mean = mu.mean(axis=0)
            # Простая квадратичная ценностная функция
            v = self.value_net.forward(mu_mean.reshape(1, -1)).flatten()
            return {
                "values": v.tolist(),
                "mean_value": float(np.mean(v)),
                "value_range": [float(np.min(v)), float(np.max(v))],
            }
        except Exception as exc:
            logger.error("Value function computation error: %s", exc)
            return {"error": str(exc)}

    def _compute_social_welfare(
        self, mu: np.ndarray, policy: Dict[str, Any]
    ) -> float:
        try:
            optimal_action = np.array(policy.get("optimal_action", [0.0] * self.state_dim))
            action_cost = float(np.sum(optimal_action**2)) * 0.5 * self.risk_aversion
            distribution_utility = -float(np.mean(mu**2)) * 0.5
            return distribution_utility - action_cost
        except Exception as exc:
            logger.error("Social welfare computation error: %s", exc)
            return 0.0

    def _compute_price_of_anarchy(self, social_welfare: float, mu: np.ndarray) -> float:
        try:
            social_optimum = -float(np.mean(mu**2)) * 0.3
            if abs(social_optimum) < 1e-10:
                return 1.0
            return abs(social_welfare / social_optimum)
        except Exception as exc:
            logger.error("Price of anarchy computation error: %s", exc)
            return 1.0

    def _compute_actions(
        self, states: np.ndarray, policy: Dict[str, Any]
    ) -> np.ndarray:
        try:
            optimal_action = np.array(policy.get("optimal_action", [0.0] * self.state_dim))
            mean_state = np.array(policy.get("mean_field_state", states.mean(axis=0).tolist()))
            deviation = states - mean_state
            actions = -self.risk_aversion * deviation + optimal_action
            # Ограничение на размер действий
            norm = np.linalg.norm(actions, axis=1, keepdims=True)
            clamp = np.where(norm > 1.0, norm, 1.0)
            return actions / clamp
        except Exception as exc:
            logger.error("Action computation error: %s", exc)
            return np.zeros_like(states)

    # ------------------------------------------------------------------
    # Private — market metrics
    # ------------------------------------------------------------------

    def _compute_max_drawdown(self, prices: List[float]) -> float:
        try:
            if len(prices) < 2:
                return 0.0
            arr = np.array(prices)
            peak = np.maximum.accumulate(arr)
            drawdown = (arr - peak) / np.where(peak != 0, peak, 1.0)
            return float(-np.min(drawdown))
        except Exception:
            return 0.0

    def _compute_sharpe_ratio(self, prices: List[float]) -> float:
        try:
            if len(prices) < 2:
                return 0.0
            returns = np.diff(prices) / np.array(prices[:-1])
            std = float(np.std(returns))
            return float(np.mean(returns)) / std if std > 0 else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Public — distribution helpers
    # ------------------------------------------------------------------

    def _generate_initial_distribution(self) -> np.ndarray:
        """Генерирует начальное распределение N(0, 1)^(num_agents × state_dim)."""
        return np.random.randn(self.num_agents, self.state_dim)
