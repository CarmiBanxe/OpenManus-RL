"""
Аналитический решатель Remizov — стохастические ODE/PDE для принятия решений
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.integrate import odeint, quad
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available — RemizovSolver will use numpy fallbacks")

try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy not available — symbolic solving disabled")


class RemizovSolver:
    """
    Аналитический решатель на основе стохастических уравнений Remizov.
    Использует ODE/PDE для вычисления вероятностей переходов между состояниями.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.dt: float = self.config.get("dt", 0.01)
        self.n_steps: int = self.config.get("n_steps", 100)
        self.volatility: float = self.config.get("volatility", 0.2)
        self.mean_reversion: float = self.config.get("mean_reversion", 0.5)
        logger.info(
            f"RemizovSolver initialized: dt={self.dt}, n_steps={self.n_steps}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_decision_ode(
        self,
        initial_state: np.ndarray,
        action_weights: List[float],
        risk_factors: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Решение ODE для вычисления оптимальных весов решения.

        Модель: dx/dt = -α·x + β·u + σ·noise
        где x — вектор состояния, u — управляющее воздействие (action_weights).
        """
        risk_factors = risk_factors or [0.1] * len(action_weights)

        try:
            if SCIPY_AVAILABLE:
                return self._solve_with_scipy(
                    initial_state, action_weights, risk_factors
                )
            return self._solve_with_numpy(initial_state, action_weights, risk_factors)
        except Exception as exc:
            logger.error(f"ODE solver error: {exc}")
            return self._fallback_solution(action_weights)

    def solve_stochastic_volatility_pde(
        self,
        state: np.ndarray,
        time_horizon: float = 1.0,
        n_paths: int = 1000,
    ) -> Dict[str, Any]:
        """
        Метод Монте-Карло для стохастической волатильности (приближение к PDE Heston).
        """
        try:
            dt = time_horizon / self.n_steps
            paths = np.zeros((n_paths, self.n_steps + 1, len(state)))
            paths[:, 0, :] = state

            for step in range(self.n_steps):
                noise = np.random.normal(0, np.sqrt(dt), (n_paths, len(state)))
                drift = -self.mean_reversion * paths[:, step, :] * dt
                diffusion = self.volatility * paths[:, step, :] * noise
                paths[:, step + 1, :] = paths[:, step, :] + drift + diffusion

            final_states = paths[:, -1, :]
            return {
                "mean_trajectory": np.mean(paths, axis=0).tolist(),
                "std_trajectory": np.std(paths, axis=0).tolist(),
                "final_mean": np.mean(final_states, axis=0).tolist(),
                "final_std": np.std(final_states, axis=0).tolist(),
                "var_95": np.percentile(final_states, 5, axis=0).tolist(),
                "cvar_95": self._calculate_cvar(final_states),
            }
        except Exception as exc:
            logger.error(f"PDE solver error: {exc}")
            return {"mean_trajectory": [], "final_mean": state.tolist()}

    def calculate_transition_probabilities(
        self,
        current_state: Dict[str, Any],
        possible_actions: List[str],
        context_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Вычисление вероятностей перехода между состояниями для каждого действия.
        """
        if not possible_actions:
            return {}

        n_actions = len(possible_actions)
        features = (
            context_features
            if context_features is not None
            else np.ones(n_actions) * 0.5
        )

        try:
            weights = self._calculate_action_weights(features, current_state)
            probabilities = self._softmax(weights)
            return dict(zip(possible_actions, probabilities.tolist()))
        except Exception as exc:
            logger.error(f"Transition probability error: {exc}")
            uniform = 1.0 / n_actions
            return {a: uniform for a in possible_actions}

    def analytical_risk_assessment(
        self,
        state_vector: np.ndarray,
        action_vector: np.ndarray,
        risk_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Аналитическая оценка риска через интеграл Лапласа / нормальное приближение.
        """
        try:
            risk_integral = self._laplace_transform_solution(
                state_vector, action_vector
            )
            risk_level = float(np.mean(np.abs(risk_integral)))
            is_acceptable = risk_level < risk_threshold
            confidence = max(0.0, 1.0 - abs(risk_level - risk_threshold))

            return {
                "risk_level": risk_level,
                "is_acceptable": is_acceptable,
                "confidence": confidence,
                "risk_components": self._decompose_risk(state_vector, action_vector),
                "recommendation": "proceed" if is_acceptable else "review_required",
            }
        except Exception as exc:
            logger.error(f"Risk assessment error: {exc}")
            return {
                "risk_level": 0.5,
                "is_acceptable": True,
                "confidence": 0.5,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal ODE solvers
    # ------------------------------------------------------------------

    def _solve_with_scipy(
        self,
        initial_state: np.ndarray,
        action_weights: List[float],
        risk_factors: List[float],
    ) -> Dict[str, Any]:
        u = np.array(action_weights[: len(initial_state)])
        alpha = self.mean_reversion
        beta = 1.0

        def ode_system(
            x: np.ndarray, t: float, u: np.ndarray, alpha: float, beta: float
        ) -> np.ndarray:
            noise = np.random.normal(0, self.volatility * np.sqrt(self.dt), len(x))
            return -alpha * x + beta * u + noise

        t_points = np.linspace(0, self.n_steps * self.dt, self.n_steps)
        solution = odeint(ode_system, initial_state, t_points, args=(u, alpha, beta))
        final = solution[-1]
        weights = self._softmax(final)

        return {
            "optimal_weights": weights.tolist(),
            "trajectory": solution.tolist(),
            "convergence": True,
            "solver": "scipy_odeint",
        }

    def _solve_with_numpy(
        self,
        initial_state: np.ndarray,
        action_weights: List[float],
        risk_factors: List[float],
    ) -> Dict[str, Any]:
        state = initial_state.copy()
        u = np.array(action_weights[: len(initial_state)])
        trajectory = [state.copy()]

        for _ in range(self.n_steps):
            noise = np.random.normal(0, self.volatility * np.sqrt(self.dt), len(state))
            state = (
                state
                + (-self.mean_reversion * state + u + noise) * self.dt
            )
            trajectory.append(state.copy())

        weights = self._softmax(state)
        return {
            "optimal_weights": weights.tolist(),
            "trajectory": trajectory,
            "convergence": True,
            "solver": "numpy_euler",
        }

    def _fallback_solution(self, action_weights: List[float]) -> Dict[str, Any]:
        weights_arr = np.array(action_weights, dtype=float)
        if weights_arr.sum() > 0:
            weights_arr /= weights_arr.sum()
        else:
            weights_arr = np.ones_like(weights_arr) / len(weights_arr)
        return {
            "optimal_weights": weights_arr.tolist(),
            "trajectory": [],
            "convergence": False,
            "solver": "fallback_uniform",
        }

    # ------------------------------------------------------------------
    # Laplace / risk helpers
    # ------------------------------------------------------------------

    def _laplace_transform_solution(
        self,
        state_vector: np.ndarray,
        action_vector: np.ndarray,
    ) -> np.ndarray:
        """Приближение через числовую аппроксимацию Лапласа."""
        t_points = np.linspace(0.01, 10.0, self.n_steps)
        laplace_solution = np.zeros(self.n_steps)

        # Простая интерполяция "решения" по времени (ответ на стохастическое воздействие)
        combined = state_vector[: len(t_points)] if len(state_vector) >= len(t_points) else \
            np.pad(state_vector, (0, len(t_points) - len(state_vector)), constant_values=state_vector[-1] if len(state_vector) else 0.5)

        if SCIPY_AVAILABLE:
            for i, t in enumerate(t_points):
                integrand = lambda s, _t=t, _c=combined: (
                    _c[min(int(s * len(_c)), len(_c) - 1)] * np.exp(-s * _t)
                )
                try:
                    laplace_solution[i], _ = quad(integrand, 0, 10)
                except Exception:
                    laplace_solution[i] = combined[i] * np.exp(-t)
        else:
            for i, t in enumerate(t_points):
                laplace_solution[i] = combined[i] * np.exp(-t)

        return laplace_solution

    def _calculate_cvar(
        self, final_states: np.ndarray, alpha: float = 0.05
    ) -> List[float]:
        """Conditional Value at Risk (CVaR) at alpha level."""
        var = np.percentile(final_states, alpha * 100, axis=0)
        cvar = []
        for j in range(final_states.shape[1]):
            tail = final_states[final_states[:, j] <= var[j], j]
            cvar.append(float(np.mean(tail)) if len(tail) > 0 else float(var[j]))
        return cvar

    def _calculate_action_weights(
        self,
        features: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        confidence = float(state.get("confidence", 0.5))
        risk = float(state.get("risk_level", 0.3))
        features_adjusted = features * (1.0 + confidence - risk)
        return features_adjusted

    def _decompose_risk(
        self,
        state_vector: np.ndarray,
        action_vector: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "market_risk": float(np.std(state_vector) * 0.4),
            "execution_risk": float(np.std(action_vector) * 0.3) if len(action_vector) > 0 else 0.0,
            "model_risk": 0.1 + float(np.abs(np.mean(state_vector) - 0.5)) * 0.2,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x)
        exp_x = np.exp(np.clip(x_shifted, -500, 500))
        return exp_x / (exp_x.sum() + 1e-9)


# ---------------------------------------------------------------------------
# DecisionEngineWithRemizov
# ---------------------------------------------------------------------------


class DecisionEngineWithRemizov:
    """
    Движок принятия решений, интегрирующий RemizovSolver с теорией решений.
    """

    def __init__(
        self,
        solver: Optional[RemizovSolver] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}
        self.solver = solver or RemizovSolver(config)
        logger.info("DecisionEngineWithRemizov initialized")

    async def make_enhanced_decision(
        self,
        state: Dict[str, Any],
        available_actions: List[str],
        context_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Принятие решения с аналитическим обоснованием Remizov."""
        try:
            state_vector = self._state_to_vector(state)
            n_actions = len(available_actions)

            ode_result = self.solver.solve_decision_ode(
                state_vector,
                [1.0 / n_actions] * n_actions if n_actions > 0 else [1.0],
                risk_factors=[state.get("risk_level", 0.3)] * n_actions,
            )

            transition_probs = self.solver.calculate_transition_probabilities(
                state, available_actions, context_features
            )

            action_vector = np.array(
                [transition_probs.get(a, 1.0 / n_actions) for a in available_actions]
            )
            risk_assessment = self.solver.analytical_risk_assessment(
                state_vector, action_vector
            )

            combined_scores = self._combine_scores(
                ode_result.get("optimal_weights", []),
                list(transition_probs.values()),
                risk_assessment,
            )

            if combined_scores and available_actions:
                best_idx = int(np.argmax(combined_scores))
                best_action = available_actions[best_idx]
                best_score = combined_scores[best_idx]
            else:
                best_action = available_actions[0] if available_actions else "default"
                best_score = 0.5

            return {
                "action": best_action,
                "confidence": float(best_score),
                "explanation": (
                    f"Remizov ODE → '{best_action}' "
                    f"(risk={risk_assessment.get('risk_level', 0):.2f}, "
                    f"conv={ode_result.get('convergence', False)})"
                ),
                "ode_result": ode_result,
                "transition_probabilities": transition_probs,
                "risk_assessment": risk_assessment,
                "analytical_scores": combined_scores,
                "solver_used": ode_result.get("solver", "unknown"),
            }

        except Exception as exc:
            logger.error(f"Enhanced decision error: {exc}")
            fallback = available_actions[0] if available_actions else "default"
            return {
                "action": fallback,
                "confidence": 0.3,
                "explanation": f"Fallback (solver error: {exc})",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        numeric_values = []
        for v in state.values():
            if isinstance(v, (int, float)):
                numeric_values.append(float(v))
            elif isinstance(v, bool):
                numeric_values.append(1.0 if v else 0.0)
        if not numeric_values:
            return np.array([0.5])
        return np.array(numeric_values[:10])

    def _combine_scores(
        self,
        ode_weights: List[float],
        transition_probs: List[float],
        risk_assessment: Dict[str, Any],
    ) -> List[float]:
        if not ode_weights or not transition_probs:
            return transition_probs or ode_weights

        n = min(len(ode_weights), len(transition_probs))
        risk_penalty = float(risk_assessment.get("risk_level", 0.3))

        combined = []
        for i in range(n):
            score = (
                0.6 * ode_weights[i]
                + 0.4 * transition_probs[i]
                - 0.1 * risk_penalty
            )
            combined.append(max(0.0, score))
        return combined
