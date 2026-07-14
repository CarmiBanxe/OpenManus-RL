"""Единая конфигурация агента (все слои в одном месте)."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentConfig:
    # LLM / шлюз
    model: str = "smart"
    base_url: str = "http://localhost:4000"
    master_key: Optional[str] = None  # None -> из env LITELLM_MASTER_KEY
    session_id: str = "default"
    # память (S13)
    memory: bool = True
    memory_db: str = ":memory:"
    max_turns: int = 20
    memory_summarize: bool = False
    # RAG (S14)
    rag: bool = False
    rag_k: int = 5
    embed_model: str = "nomic-embed-text"
    # инструменты (S15)
    tools: bool = False
    max_tool_iters: int = 5
    # observability (S11)
    enable_observability: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def engine_config(self) -> Dict[str, Any]:
        """Конфиг для нижележащих адаптеров (S10/S12/S15)."""
        cfg = {
            "model": self.model, "base_url": self.base_url,
            "enable_observability": self.enable_observability,
            "max_tool_iters": self.max_tool_iters, **self.extra,
        }
        if self.master_key is not None:
            cfg["master_key"] = self.master_key
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
