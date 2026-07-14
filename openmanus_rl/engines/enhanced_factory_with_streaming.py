"""
Фабрика движков с поддержкой streaming (+ опц. observability).

Отдельный слой, НЕ подменяет S10/S11. Реальные классы: LiteLLMAdapter (S10),
StreamingLiteLLMAdapter (S12), ChatOpenAI/create_optimized_ollama_engine/
create_llm_engine. `create_engine(..., stream=True)` авто-переключает litellm->streaming.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from openmanus_rl.engines.enhanced_factory import LiteLLMAdapter
from openmanus_rl.engines.openai import ChatOpenAI
from openmanus_rl.engines.optimized_ollama_engine import create_optimized_ollama_engine
from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter

try:
    from openmanus_rl.observability import get_health_checker, get_logger, get_metrics_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBSERVABILITY_AVAILABLE = False


def _create_ollama_engine(cfg: Dict[str, Any]):
    from openmanus_rl.engines.ollama_engine import create_llm_engine
    return create_llm_engine(cfg.get("model", "qwen2.5:7b-instruct"))


def _create_openai_engine(cfg: Dict[str, Any]) -> ChatOpenAI:
    return ChatOpenAI(model=cfg.get("model", "gpt-4o-mini"), base_url=cfg.get("base_url"),
                      api_key=cfg.get("api_key"), temperature=cfg.get("temperature", 0.0))


class StreamingLiteLLMFactory:
    """Фабрика с movками streaming/observability (opt-in)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = Path(config_path) if config_path else (
            self.project_root / "config" / "engines.toml")
        self.config = self._load_config()

        self._obs = OBSERVABILITY_AVAILABLE
        if self._obs:
            self.metrics = get_metrics_collector()
            self.logger = get_logger()
            self.health_checker = get_health_checker()
            self.health_checker.register_component(
                "StreamingLiteLLMFactory", self._health_check, interval=30.0)

        self._engines: Dict[str, Callable[..., Any]] = {
            "ollama": _create_ollama_engine,
            "openai": _create_openai_engine,
            "optimized_ollama": lambda cfg: create_optimized_ollama_engine(cfg),
            "litellm": LiteLLMAdapter,
            "streaming_litellm": StreamingLiteLLMAdapter,
        }
        self._engine_cache: Dict[str, Any] = {}

    def _default_config(self) -> Dict[str, Any]:
        base = {"base_url": "http://localhost:4000", "model": "gpt-3.5-turbo", "timeout": 60,
                "max_retries": 3, "fallback_models": ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
                "enable_observability": False}
        return {
            "default_engine": "litellm",
            "litellm": dict(base),
            "streaming_litellm": dict(base),
            "ollama": {"host": "localhost", "port": 11434, "model": "qwen2.5:7b-instruct", "timeout": 60},
            "openai": {"model": "gpt-3.5-turbo", "timeout": 60},
            "optimized_ollama": {"host": "localhost", "port": 11434,
                                 "model": "qwen2.5:7b-instruct", "timeout": 60},
        }

    def _load_config(self) -> Dict[str, Any]:
        cfg = self._default_config()
        if self.config_path.exists() and tomllib is not None:
            try:
                loaded = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка загрузки конфигурации ({exc}); дефолт")
                return cfg
            default_section = loaded.pop("default", {})
            if "default_engine" in default_section:
                cfg["default_engine"] = default_section["default_engine"]
            for section, values in loaded.items():
                if isinstance(values, dict):
                    cfg.setdefault(section, {}).update(values)
        return cfg

    def _health_check(self) -> Dict[str, Any]:
        available, unavailable = [], []
        for name in self._engines:
            try:
                engine = self.create_engine(name)
                if hasattr(engine, "is_available") and engine.is_available():
                    available.append(name)
                else:
                    unavailable.append(name)
            except Exception:  # noqa: BLE001
                unavailable.append(name)
        if unavailable:
            return {"status": "degraded",
                    "message": f"Some engines unavailable: {', '.join(unavailable)}",
                    "available_engines": available, "unavailable_engines": unavailable}
        return {"status": "healthy", "message": "All engines available", "available_engines": available}

    def register_engine(self, name: str, engine_class: Callable[..., Any]) -> None:
        self._engines[name] = engine_class

    def create_engine(self, engine_type: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None, stream: bool = False) -> Any:
        engine_type = engine_type or self.config["default_engine"]
        if stream and engine_type == "litellm":
            engine_type = "streaming_litellm"
        if engine_type not in self._engines:
            raise ValueError(f"Неизвестный тип движка: {engine_type}")
        cache_key = f"{engine_type}_{hash(json.dumps(config or {}, sort_keys=True))}_{stream}"
        if cache_key in self._engine_cache:
            return self._engine_cache[cache_key]
        engine_config = config if config is not None else self.config.get(engine_type, {})
        engine = self._engines[engine_type](engine_config)
        self._engine_cache[cache_key] = engine
        return engine

    def get_available_engines(self) -> List[str]:
        return list(self._engines.keys())

    def get_engine_info(self, engine_type: str) -> Dict[str, Any]:
        if engine_type not in self._engines:
            return {}
        eng = self._engines[engine_type]
        return {"name": engine_type, "class": getattr(eng, "__name__", str(eng)),
                "config": self.config.get(engine_type, {}),
                "streaming": engine_type == "streaming_litellm"}

    def _engine_status(self, name: str) -> Dict[str, Any]:
        try:
            engine = self.create_engine(name)
            return {"available": bool(hasattr(engine, "is_available") and engine.is_available()),
                    "metrics": engine.get_metrics() if hasattr(engine, "get_metrics") else {},
                    "streaming": name == "streaming_litellm"}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc), "streaming": name == "streaming_litellm"}

    def get_observability_summary(self) -> Dict[str, Any]:
        if not self._obs:
            return {"error": "Observability not available"}
        raw = self.metrics.get_metrics_summary()
        metrics_json = {name: [{"labels": dict(l), "value": v} for l, v in samples.items()]
                        for name, samples in raw.items()}
        return {"metrics": metrics_json, "health": self.health_checker.check_all(),
                "engines": {name: self._engine_status(name) for name in self._engines}}


factory = StreamingLiteLLMFactory()


def create_engine(engine_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                  stream: bool = False) -> Any:
    return factory.create_engine(engine_type, config, stream)


def create_streaming_engine(config: Optional[Dict[str, Any]] = None) -> StreamingLiteLLMAdapter:
    return factory.create_engine("streaming_litellm", config)


def get_available_engines() -> List[str]:
    return factory.get_available_engines()


def get_engine_info(engine_type: str) -> Dict[str, Any]:
    return factory.get_engine_info(engine_type)


def get_observability_summary() -> Dict[str, Any]:
    return factory.get_observability_summary()
