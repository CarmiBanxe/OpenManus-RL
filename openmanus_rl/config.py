"""
Конфигурация OpenManus — реальный load_config (P0-фундамент Спринта 6).

Возвращает ПЛОСКИЙ dict (а не класс), совместимый и с EnhancedDecisionAgent
(sub-configs qwen3_omni/voice_pipeline/mean_field_games + enable_* флаги),
и с сервисным слоем (host/port/cors/secret из env).

SECURITY (приватный Legion-контур, S-18 §1.2):
  - секреты только из env (никаких дефолтных production-ключей);
  - host по умолчанию 127.0.0.1 (не 0.0.0.0);
  - CORS по умолчанию localhost;
  - share/публичные туннели по умолчанию выключены.
"""
import os
from typing import Any, Dict

_ENVIRONMENTS = {"production", "development", "testing"}


def _base_defaults() -> Dict[str, Any]:
    return {
        # component enable flags (реальные — их читает агент)
        "enable_qwen3_omni": True,
        "enable_deep_hedging": True,
        "enable_signature_methods": True,
        "enable_mean_field_games": True,
        "enable_performance_optimization": True,
        # sandbox по умолчанию для sub-компонентов (без тяжёлых моделей)
        "qwen3_omni": {"sandbox_mode": True},
        "voice_pipeline": {"sandbox_mode": True},
        "mean_field_games": {"num_agents": 50, "state_dim": 2, "max_iterations": 10, "time_horizon": 10},
        # service layer (безопасные дефолты)
        "host": "127.0.0.1",
        "port": 8000,
        "cors_allow_origins": ["http://localhost", "http://127.0.0.1"],
        "gradio_share": False,
        "log_level": "INFO",
        "request_timeout": 60,
        "osint_rate_limit": 100,
    }


def _environment_overrides(env: str) -> Dict[str, Any]:
    if env == "development":
        return {"log_level": "DEBUG", "request_timeout": 30}
    if env == "testing":
        return {
            "log_level": "ERROR",
            "request_timeout": 10,
            "mean_field_games": {"num_agents": 8, "state_dim": 2, "max_iterations": 2, "time_horizon": 3},
        }
    return {}  # production — базовые дефолты


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def load_config(name: str = "production") -> Dict[str, Any]:
    """Загрузить конфигурацию по имени окружения или пути.

    name: 'production' | 'development' | 'testing' | путь вида 'config/testing.py'
          (из пути извлекается имя окружения).
    """
    env = name
    if name.endswith(".py") or "/" in name:
        env = os.path.splitext(os.path.basename(name))[0]
    if env not in _ENVIRONMENTS:
        env = "production"

    cfg = _base_defaults()
    cfg.update(_environment_overrides(env))
    cfg["environment"] = env

    # env-переопределения (секрет НЕ имеет небезопасного дефолта)
    cfg["secret_key"] = os.environ.get("OPENMANUS_SECRET_KEY")  # None -> auth откажет явно
    cfg["host"] = os.environ.get("OPENMANUS_HOST", cfg["host"])
    cfg["port"] = _int_env("OPENMANUS_PORT", cfg["port"])
    cfg["request_timeout"] = _int_env("OPENMANUS_REQUEST_TIMEOUT", cfg["request_timeout"])
    cfg["osint_rate_limit"] = _int_env("OPENMANUS_OSINT_RATE_LIMIT", cfg["osint_rate_limit"])
    cors_env = os.environ.get("OPENMANUS_CORS_ORIGINS")
    if cors_env:
        cfg["cors_allow_origins"] = [o.strip() for o in cors_env.split(",") if o.strip()]

    return cfg
