"""
Мост: OctoTools BaseTool (openmanus_rl/tools/) -> Tool (function-calling слой).

Позволяет вызывать богатые OctoTools-инструменты (wikipedia/arxiv/…) через
OpenAI tool-calling, не дублируя их. Тяжёлые зависимости не импортируются здесь —
вызывающий передаёт УЖЕ инстанцированный BaseTool.
"""
from typing import Any, Dict

from .registry import Tool


def _input_types_to_schema(input_types: Any) -> Dict[str, Any]:
    """OctoTools input_types (dict имя->описание) -> JSON Schema object."""
    props: Dict[str, Any] = {}
    if isinstance(input_types, dict):
        for name, desc in input_types.items():
            props[name] = {"type": "string", "description": str(desc)}
    return {"type": "object", "properties": props}


def wrap_octotool(base_tool: Any) -> Tool:
    """Обернуть инстанс OctoTools BaseTool в Tool для реестра function-calling."""
    meta = base_tool.get_metadata() if hasattr(base_tool, "get_metadata") else {}
    name = meta.get("tool_name") or getattr(base_tool, "tool_name", base_tool.__class__.__name__)
    description = meta.get("tool_description") or getattr(base_tool, "tool_description", "") or name
    parameters = _input_types_to_schema(meta.get("input_types") or getattr(base_tool, "input_types", {}))

    def _call(**kwargs: Any) -> Any:
        return base_tool.execute(**kwargs)

    return Tool(name=name, description=description, parameters=parameters, func=_call)
