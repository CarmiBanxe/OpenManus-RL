"""LLM engine interfaces and factories.

This package provides lightweight wrappers around OpenAI-compatible
chat completion APIs and a simple factory used by tool modules.
"""

# Расширенная фабрика с интеграцией LiteLLM (Спринт 10). Импорт import-safe:
# опциональная зависимость `ollama` подтягивается лениво внутри фабрики.
from openmanus_rl.engines.enhanced_factory import (  # noqa: F401,E402
    BaseEngine,
    EnhancedEngineFactory,
    LiteLLMAdapter,
    create_engine,
    get_available_engines,
    get_engine_info,
)

__all__ = [
    "BaseEngine",
    "EnhancedEngineFactory",
    "LiteLLMAdapter",
    "create_engine",
    "get_available_engines",
    "get_engine_info",
]

