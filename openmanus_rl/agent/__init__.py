"""Единый агент-фасад Legion (S17) — композиция S10–S16."""
from .config import AgentConfig
from .legion_agent import LegionAgent, create_agent

__all__ = ["AgentConfig", "LegionAgent", "create_agent"]
