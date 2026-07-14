"""
Провайдеры эмбеддингов для семантической памяти (S14).

Дефолт — локальный Ollama `nomic-embed-text` (768-dim, :11434/api/embeddings):
без gateway-зависимости и без внешних ключей. Абстракция EmbeddingProvider
позволяет подменять источник (напр. gateway /v1/embeddings, если появится).
"""
from abc import ABC, abstractmethod
from typing import List

import requests


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Вернуть вектор эмбеддинга для текста (или []) при ошибке — на усмотрение реализации."""
        ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Эмбеддинги через локальный Ollama (nomic-embed-text по умолчанию)."""

    def __init__(self, model: str = "nomic-embed-text", host: str = "localhost",
                 port: int = 11434, timeout: int = 30) -> None:
        self.model = model
        self.timeout = timeout
        self.url = f"http://{host}:{port}/api/embeddings"
        self.session = requests.Session()

    def embed(self, text: str) -> List[float]:
        resp = self.session.post(self.url, json={"model": self.model, "prompt": text},
                                 timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def is_available(self) -> bool:
        try:
            return bool(self.embed("ping"))
        except Exception:  # noqa: BLE001
            return False
