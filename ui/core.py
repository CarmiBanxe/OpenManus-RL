"""
Общее ядро UI — под РЕАЛЬНЫЙ API (select_action). Без импорта streamlit/gradio,
поэтому полностью тестируемо headless.
"""
import asyncio
from typing import Any, Dict, List, Optional

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent
from openmanus_rl.config import load_config

_agent: Optional[EnhancedDecisionAgent] = None


def get_agent(config_name: str = "development") -> EnhancedDecisionAgent:
    """Ленивая инициализация агента (sandbox-конфиг из load_config)."""
    global _agent
    if _agent is None:
        _agent = EnhancedDecisionAgent(config=load_config(config_name))
    return _agent


def run_query(text: str, available_actions: Optional[List[str]] = None,
              priority: float = 0.5) -> Dict[str, Any]:
    """Синхронная обёртка над реальным select_action — удобна для UI-колбэков."""
    actions = available_actions or ["proceed", "wait"]
    agent = get_agent()
    return asyncio.run(agent.select_action({"text": text}, actions, priority=priority))


def shutdown() -> None:
    """Освободить ресурсы агента (monitoring-поток PerformanceOptimizer и т.д.)."""
    global _agent
    if _agent is not None:
        try:
            asyncio.run(_agent.cleanup())
        except Exception:  # noqa: BLE001
            pass
        _agent = None
