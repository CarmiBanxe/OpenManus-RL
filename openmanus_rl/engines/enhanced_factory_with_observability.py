"""
Observability-обёртка над движками OpenManus (опциональная, дефолт-выкл).

НЕ подменяет S10 `enhanced_factory.py` — отдельный слой. Реальные классы движков:
ChatOpenAI (kwargs), OptimizedOllamaEngine/create_optimized_ollama_engine (один dict),
create_llm_engine (ленивый, prompt->str). `ObservableLiteLLMAdapter` — единственный
полноценный движок (generate/chat/is_available/get_metrics) с опц. метриками/логами.
"""
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

# Реальные движки (импорт-safe): ChatOpenAI/openai guarded; optimized — только requests.
from openmanus_rl.engines.openai import ChatOpenAI
from openmanus_rl.engines.optimized_ollama_engine import (
    OptimizedOllamaEngine,  # noqa: F401  (реэкспорт/типизация)
    create_optimized_ollama_engine,
)

try:
    from openmanus_rl.observability import get_health_checker, get_logger, get_metrics_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBSERVABILITY_AVAILABLE = False


class ObservableLiteLLMAdapter:
    """LiteLLM-адаптер с опциональной observability (enable_observability=False)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.base_url = self.config.get("base_url", "http://localhost:4000")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.timeout = int(self.config.get("timeout", 60))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.fallback_models = list(self.config.get("fallback_models", []))
        self.master_key = self.config.get("master_key") or os.environ.get("LITELLM_MASTER_KEY", "")
        self.enable_observability = bool(self.config.get("enable_observability", False))

        self.session = requests.Session()
        if self.master_key:
            self.session.headers.update({"Authorization": f"Bearer {self.master_key}"})

        self.internal_metrics: Dict[str, Any] = {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_time": 0.0, "avg_response_time": 0.0, "tokens_per_second": 0.0,
            "fallback_used": 0,
        }
        self._available: Optional[bool] = None
        self._check_litellm_availability()

        self._obs = OBSERVABILITY_AVAILABLE and self.enable_observability
        if self._obs:
            self.metrics = get_metrics_collector()
            self.logger = get_logger()
            self.health_checker = get_health_checker()
            self.health_checker.register_component("LiteLLMAdapter", self._health_check, interval=30.0)

    def _check_litellm_availability(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code in (200, 401):  # 401 = up-but-auth
                self._available = True
                return True
        except Exception:  # noqa: BLE001
            pass
        self._available = False
        return False

    def is_available(self) -> bool:
        return self._available is True

    def _health_check(self) -> Dict[str, Any]:
        if self.is_available():
            return {"status": "healthy", "message": "LiteLLM is available"}
        return {"status": "unhealthy", "message": "LiteLLM is not available"}

    def _start_request(self, model: str, operation: str) -> Optional[str]:
        if not self._obs:
            return None
        request_id = str(uuid.uuid4())
        self.metrics.record_request_start("litellm", model)
        self.logger.request(engine_type="litellm", model=model,
                            operation=operation, request_id=request_id)
        return request_id

    def _end_request(self, request_id: Optional[str], model: str, operation: str,
                     status: str, duration: float, tokens: int = 0) -> None:
        if not (self._obs and request_id):
            return
        self.metrics.record_request_end("litellm", model, status, operation, duration, tokens)
        self.logger.response(engine_type="litellm", model=model, operation=operation,
                            request_id=request_id, status=status, duration=duration, tokens=tokens)

    def _log_error(self, request_id: Optional[str], operation: str, error: str,
                   error_type: str = "unknown") -> None:
        if not (self._obs and request_id):
            return
        self.metrics.record_error("litellm", error_type)
        self.logger.error(engine_type="litellm", operation=operation,
                         request_id=request_id, error=error, error_type=error_type)

    def _make_request(self, endpoint: str, payload: Dict[str, Any],
                      request_id: Optional[str]) -> Dict[str, Any]:
        start = time.time()
        original_model = payload.get("model", self.model)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Ошибка запроса к LiteLLM: {exc}") from exc

            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                if "total_tokens" in usage:
                    elapsed = time.time() - start
                    if elapsed > 0:
                        self.internal_metrics["tokens_per_second"] = usage["total_tokens"] / elapsed
                return result

            if attempt < self.max_retries and self.fallback_models:
                fallback_model = self.fallback_models[attempt % len(self.fallback_models)]
                payload["model"] = fallback_model
                self.internal_metrics["fallback_used"] += 1
                if self._obs and request_id:
                    self.metrics.record_fallback("litellm", original_model, fallback_model)
                    self.logger.fallback(engine_type="litellm", from_model=original_model,
                                        to_model=fallback_model, request_id=request_id)
                continue
            raise RuntimeError(f"Ошибка запроса к LiteLLM: {response.text}")
        raise RuntimeError("Неизвестная ошибка в _make_request")

    def _run(self, endpoint: str, payload: Dict[str, Any], operation: str) -> Dict[str, Any]:
        request_id = self._start_request(self.model, operation)
        start = time.time()
        try:
            result = self._make_request(endpoint, payload, request_id)
            duration = time.time() - start
            tokens = result.get("usage", {}).get("total_tokens", 0)
            self._end_request(request_id, self.model, operation, "success", duration, tokens)
            self.internal_metrics["total_requests"] += 1
            self.internal_metrics["total_time"] += duration
            self.internal_metrics["avg_response_time"] = (
                self.internal_metrics["total_time"] / self.internal_metrics["total_requests"])
            self.internal_metrics["successful_requests"] += 1
            return result
        except Exception as exc:
            duration = time.time() - start
            self._end_request(request_id, self.model, operation, "error", duration)
            self._log_error(request_id, operation, str(exc), "request_error")
            self.internal_metrics["total_requests"] += 1
            self.internal_metrics["failed_requests"] += 1
            raise

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model), "prompt": prompt, "stream": False,
            "temperature": kwargs.get("temperature", 0.7), "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        return self._run("/v1/completions", payload, "generate")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model), "messages": messages, "stream": False,
            "temperature": kwargs.get("temperature", 0.7), "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        return self._run("/v1/chat/completions", payload, "chat")

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.internal_metrics)

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as exc:  # noqa: BLE001
            if self._obs:
                self.logger.error_log("Error listing models", error=str(exc))
        return []

    def model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        for model in self.list_models():
            if model.get("id") == model_name:
                return model
        return {}


def _create_ollama_engine(cfg: Dict[str, Any]) -> Callable[[str], str]:
    """Ленивая обёртка над create_llm_engine (импорт `ollama` — только здесь)."""
    from openmanus_rl.engines.ollama_engine import create_llm_engine
    return create_llm_engine(cfg.get("model", "qwen2.5:7b-instruct"))


def _create_openai_engine(cfg: Dict[str, Any]) -> ChatOpenAI:
    """Адаптер dict->kwargs для реального ChatOpenAI (dict не принимает)."""
    return ChatOpenAI(
        model=cfg.get("model", "gpt-4o-mini"),
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.0),
    )


class ObservableEngineFactory:
    """Фабрика движков с опциональной observability (без подмены S10)."""

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
                "ObservableEngineFactory", self._health_check, interval=30.0)

        self._engines: Dict[str, Callable[..., Any]] = {
            "ollama": _create_ollama_engine,
            "openai": _create_openai_engine,
            "optimized_ollama": lambda cfg: create_optimized_ollama_engine(cfg),
            "litellm": ObservableLiteLLMAdapter,
        }
        self._engine_cache: Dict[str, Any] = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "default_engine": "litellm",
            "litellm": {"base_url": "http://localhost:4000", "model": "gpt-3.5-turbo",
                        "timeout": 60, "max_retries": 3,
                        "fallback_models": ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
                        "enable_observability": False},
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
                print(f"Ошибка загрузки конфигурации ({exc}); используются значения по умолчанию")
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
        return {"status": "healthy", "message": "All engines available",
                "available_engines": available}

    def register_engine(self, name: str, engine_class: Callable[..., Any]) -> None:
        self._engines[name] = engine_class

    def create_engine(self, engine_type: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None) -> Any:
        engine_type = engine_type or self.config["default_engine"]
        if engine_type not in self._engines:
            raise ValueError(f"Неизвестный тип движка: {engine_type}")
        cache_key = f"{engine_type}_{hash(repr(config))}"
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
                "config": self.config.get(engine_type, {})}

    def _engine_status(self, name: str) -> Dict[str, Any]:
        try:
            engine = self.create_engine(name)
            return {
                "available": bool(hasattr(engine, "is_available") and engine.is_available()),
                "metrics": engine.get_metrics() if hasattr(engine, "get_metrics") else {},
            }
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc)}

    def get_observability_summary(self) -> Dict[str, Any]:
        if not self._obs:
            return {"error": "Observability not available"}
        # JSON-safe метрики: tuple-ключи -> список {labels, value}.
        raw = self.metrics.get_metrics_summary()
        metrics_json = {
            name: [{"labels": dict(labels), "value": value} for labels, value in samples.items()]
            for name, samples in raw.items()
        }
        return {
            "metrics": metrics_json,
            "health": self.health_checker.check_all(),
            "engines": {name: self._engine_status(name) for name in self._engines},
        }


factory: Optional[ObservableEngineFactory] = None
if OBSERVABILITY_AVAILABLE:
    factory = ObservableEngineFactory()


def create_engine(engine_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    return factory.create_engine(engine_type, config)


def get_available_engines() -> List[str]:
    return factory.get_available_engines()


def get_engine_info(engine_type: str) -> Dict[str, Any]:
    return factory.get_engine_info(engine_type)


def get_observability_summary() -> Dict[str, Any]:
    return factory.get_observability_summary()
