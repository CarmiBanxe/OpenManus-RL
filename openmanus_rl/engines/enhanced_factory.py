"""
Расширенная фабрика движков с тонким адаптером поверх LiteLLM (:4000).

Адаптер НЕ дублирует routing/balancing/fallback LiteLLM — он даёт единый
интерфейс generate/chat + метрики производительности + graceful-availability.
Реальные классы движков переиспользуются из engines/ (ChatOpenAI,
OptimizedOllamaEngine, create_optimized_ollama_engine, create_llm_engine).
Секреты — только из окружения (LITELLM_MASTER_KEY); в коде/конфиге не хранятся.
"""
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

# Реальные классы движков (импорт безопасен: openai-обёртка guarded внутри).
from openmanus_rl.engines.openai import ChatOpenAI
from openmanus_rl.engines.optimized_ollama_engine import (
    create_optimized_ollama_engine,
)


class BaseEngine(ABC):
    """Базовый контракт движка."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        ...


class LiteLLMAdapter(BaseEngine):
    """Тонкий адаптер поверх LiteLLM с метриками и graceful-availability."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.base_url = self.config.get("base_url", "http://localhost:4000")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.timeout = int(self.config.get("timeout", 60))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.fallback_models = list(self.config.get("fallback_models", []))
        # Секрет — только из env (или явно переданный вызывающим), не из репо.
        self.master_key = self.config.get("master_key") or os.environ.get("LITELLM_MASTER_KEY", "")

        self.session = requests.Session()
        if self.master_key:
            self.session.headers.update({"Authorization": f"Bearer {self.master_key}"})

        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "tokens_per_second": 0.0,
            "fallback_used": 0,
        }

        # Ленивая проверка: не бросает в __init__; 401 = "up, но требует auth".
        self._available: Optional[bool] = None
        self._check_litellm_availability()

    def _check_litellm_availability(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            # 200 = ok; 401 = сервис поднят, но требует ключ (тоже "доступен").
            if resp.status_code in (200, 401):
                self._available = True
                return True
        except Exception:  # noqa: BLE001
            pass
        self._available = False
        return False

    def is_available(self) -> bool:
        return self._available is True

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Запрос к LiteLLM с ретраями и переключением на fallback-модель."""
        start = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout
                )
            except Exception as exc:  # noqa: BLE001  (сетевая ошибка)
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
                        self.metrics["tokens_per_second"] = usage["total_tokens"] / elapsed
                return result

            # Не 200: пробуем fallback-модель, если есть и остались попытки.
            if attempt < self.max_retries and self.fallback_models:
                payload["model"] = self.fallback_models[attempt % len(self.fallback_models)]
                self.metrics["fallback_used"] += 1
                continue
            raise RuntimeError(f"Ошибка запроса к LiteLLM: {response.text}")

        raise RuntimeError("Неизвестная ошибка в _make_request")

    def _record_success(self, start: float) -> None:
        self.metrics["total_requests"] += 1
        self.metrics["total_time"] += time.time() - start
        self.metrics["avg_response_time"] = (
            self.metrics["total_time"] / self.metrics["total_requests"]
        )
        self.metrics["successful_requests"] += 1

    def _record_failure(self) -> None:
        self.metrics["total_requests"] += 1
        self.metrics["failed_requests"] += 1

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "stream": False,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        start = time.time()
        try:
            result = self._make_request("/v1/completions", payload)
            self._record_success(start)
            return result
        except Exception:
            self._record_failure()
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        start = time.time()
        try:
            result = self._make_request("/v1/chat/completions", payload)
            self._record_success(start)
            return result
        except Exception:
            self._record_failure()
            raise

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка получения списка моделей: {exc}")
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
    """Адаптер dict->kwargs для реального ChatOpenAI (он не принимает dict)."""
    return ChatOpenAI(
        model=cfg.get("model", "gpt-4o-mini"),
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.0),
    )


class EnhancedEngineFactory:
    """Фабрика движков с поддержкой LiteLLM (реальные классы, tomllib-конфиг)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = Path(config_path) if config_path else (
            self.project_root / "config" / "engines.toml"
        )
        self.config = self._load_config()

        self._engines: Dict[str, Callable[..., Any]] = {
            "ollama": _create_ollama_engine,
            "openai": _create_openai_engine,
            "optimized_ollama": lambda cfg: create_optimized_ollama_engine(cfg),
            "litellm": LiteLLMAdapter,
        }
        self._engine_cache: Dict[str, Any] = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "default_engine": "litellm",
            "litellm": {
                "base_url": "http://localhost:4000",
                "model": "gpt-3.5-turbo",
                "timeout": 60,
                "max_retries": 3,
                "fallback_models": ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
            },
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "model": "qwen2.5:7b-instruct",
                "timeout": 60,
            },
            "openai": {"model": "gpt-3.5-turbo", "timeout": 60},
            "optimized_ollama": {
                "host": "localhost",
                "port": 11434,
                "model": "qwen2.5:7b-instruct",
                "timeout": 60,
            },
        }

    def _load_config(self) -> Dict[str, Any]:
        cfg = self._default_config()
        if self.config_path.exists() and tomllib is not None:
            try:
                loaded = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка загрузки конфигурации ({exc}); используются значения по умолчанию")
                return cfg
            # Секция [default] — служебная: поднимаем default_engine на верхний уровень.
            default_section = loaded.pop("default", {})
            if "default_engine" in default_section:
                cfg["default_engine"] = default_section["default_engine"]
            # Остальные секции — конфиги движков: сливаем поверх дефолтов.
            for section, values in loaded.items():
                if isinstance(values, dict):
                    cfg.setdefault(section, {}).update(values)
        return cfg

    def register_engine(self, name: str, engine_class: Callable[..., Any]) -> None:
        self._engines[name] = engine_class

    def create_engine(
        self, engine_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Any:
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
        return {
            "name": engine_type,
            "class": getattr(eng, "__name__", str(eng)),
            "config": self.config.get(engine_type, {}),
        }


# Экземпляр фабрики + функциональный фасад (без сетевых вызовов при импорте).
factory = EnhancedEngineFactory()


def create_engine(engine_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    return factory.create_engine(engine_type, config)


def get_available_engines() -> List[str]:
    return factory.get_available_engines()


def get_engine_info(engine_type: str) -> Dict[str, Any]:
    return factory.get_engine_info(engine_type)


if __name__ == "__main__":
    print("Доступные движки:", ", ".join(get_available_engines()))
    eng = create_engine("litellm")
    print("LiteLLM доступен:" , eng.is_available())
    if eng.is_available():
        models = eng.list_models()
        print(f"Моделей: {len(models)}")
