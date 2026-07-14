"""
OpenAI-style tool-calling слой (S15) — ОТДЕЛЬНО от OctoTools (openmanus_rl/tools/).

OctoTools — тяжёлые research-инструменты (metadata + execute). Здесь — лёгкий
function-calling для LiteLLM chat (JSON schema + tool_calls loop). Мост
wrap_octotool позволяет вызывать OctoTools-инструменты через этот слой.
"""
from .registry import Tool, ToolRegistry
from .executor import ToolExecutor
from .builtins import register_builtins, safe_eval
from .octotools_bridge import wrap_octotool

__all__ = ["Tool", "ToolRegistry", "ToolExecutor", "register_builtins", "safe_eval", "wrap_octotool"]
