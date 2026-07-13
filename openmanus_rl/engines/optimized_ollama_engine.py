"""
Оптимизированный Ollama engine wrapper для OpenManus.

Исправлено vs черновик: НЕТ тяжёлых сайд-эффектов в __init__ (не пуллит и не создаёт
модель — это ломало каждый init). GPU-слои задаются на ЗАПРОС через options.num_gpu
(валидный Ollama-параметр), не через несуществующий Modelfile PARAMETER num_gpu_layers.
Проверка доступности — ленивая (is_available), не бросает на конструкторе.
"""
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class OptimizedOllamaEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.host = cfg.get("host", "127.0.0.1")
        self.port = int(cfg.get("port", 11434))
        self.model = cfg.get("model", "qwen2.5:7b-instruct")
        self.timeout = int(cfg.get("timeout", 60))
        self.gpu_layers = int(cfg.get("gpu_layers", 0))
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
        self._sem = threading.Semaphore(int(cfg.get("max_concurrent_requests", 2)))
        self.metrics = {"total_requests": 0, "successful_requests": 0, "failed_requests": 0,
                        "total_time": 0.0, "avg_response_time": 0.0, "tokens_per_second": 0.0}

    def is_available(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("models", [])

    def _options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        opts = {"temperature": kwargs.get("temperature", 0.7), "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 2048)}
        if self.gpu_layers > 0:
            opts["num_gpu"] = self.gpu_layers   # валидный Ollama-параметр (не num_gpu_layers)
        return opts

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._sem:
            start = time.time()
            self.metrics["total_requests"] += 1
            try:
                r = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
                elapsed = time.time() - start
                self.metrics["total_time"] += elapsed
                self.metrics["avg_response_time"] = self.metrics["total_time"] / self.metrics["total_requests"]
                if r.status_code != 200:
                    self.metrics["failed_requests"] += 1
                    raise RuntimeError(f"{endpoint} -> {r.status_code}: {r.text[:200]}")
                self.metrics["successful_requests"] += 1
                result = r.json()
                dur = result.get("eval_duration", 0)
                if dur > 0:
                    toks = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    self.metrics["tokens_per_second"] = toks / (dur / 1e9)
                return result
            except requests.RequestException as exc:
                self.metrics["failed_requests"] += 1
                raise RuntimeError(f"{endpoint} error: {exc}") from exc

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": self._options(kwargs)}
        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        return self._post("/api/generate", payload)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return self._post("/api/chat", {"model": self.model, "messages": messages,
                                        "stream": False, "options": self._options(kwargs)})

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)


def create_optimized_ollama_engine(config: Optional[Dict[str, Any]] = None) -> OptimizedOllamaEngine:
    if config is None:
        cfg_path = Path(__file__).resolve().parents[2] / "config" / "performance.toml"
        if cfg_path.exists():
            try:
                import tomllib
                config = tomllib.loads(cfg_path.read_text(encoding="utf-8")).get("ollama", {})
            except Exception:  # noqa: BLE001
                config = {}
    return OptimizedOllamaEngine(config)
