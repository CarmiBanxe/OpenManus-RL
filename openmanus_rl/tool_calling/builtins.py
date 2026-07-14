"""
Безопасные встроенные инструменты (S-18: без shell/файл/сеть по умолчанию).

calculator — арифметика через ast (НЕ eval/exec). current_time, echo — тривиальные.
Опасные инструменты (shell/файлы/сеть) регистрируются вызывающим кодом ЯВНО.
"""
import ast
import operator
import time
from typing import Any

from .registry import ToolRegistry

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Вычислить арифметическое выражение безопасно (только числа и операторы)."""
    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsafe or unsupported expression")
    return _eval(ast.parse(expression, mode="eval").body)


def _calculator(expression: str) -> str:
    return str(safe_eval(expression))


def _current_time() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _echo(text: str) -> str:
    return text


def register_builtins(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        "calculator", "Evaluate an arithmetic expression (numbers and + - * / // % ** only).",
        {"type": "object", "properties": {"expression": {"type": "string"}},
         "required": ["expression"]}, _calculator)
    registry.register(
        "current_time", "Return the current UTC time (ISO 8601).",
        {"type": "object", "properties": {}}, _current_time)
    registry.register(
        "echo", "Echo back the provided text.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        _echo)
    return registry
