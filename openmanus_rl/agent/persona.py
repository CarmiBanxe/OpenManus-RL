"""
Persona / prompt-слой (S22): system-prompt шаблоны, роли, операционные guardrails.

ВАЖНО (S-18): Legion — приватный uncensored-контур. Guardrails здесь ОПЕРАЦИОННЫЕ
(лимит длины ввода, opt-in deny-patterns от оператора), НЕ навязанная цензура
контента — по умолчанию deny-list пуст.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional


class GuardrailError(ValueError):
    """Ввод нарушил операционный guardrail."""


class PromptTemplate:
    """Шаблон system-prompt с {переменными} (пропущенные -> пустая строка)."""

    def __init__(self, template: str) -> None:
        self.template = template

    def render(self, **variables: object) -> str:
        return self.template.format_map(defaultdict(str, variables))


# Встроенные персоны (system-prompt). Оператор может передать свой system_prompt.
PERSONAS = {
    "assistant": PromptTemplate("You are a helpful, accurate assistant."),
    "concise": PromptTemplate("Answer briefly and directly. No preamble, no filler."),
    "coder": PromptTemplate(
        "You are an expert software engineer. Provide correct, minimal, runnable code; "
        "explain only when asked."),
    "analyst": PromptTemplate(
        "You are a rigorous analyst. Be precise, cite assumptions, avoid speculation."),
}


def resolve_system_prompt(persona: Optional[str], system_prompt: Optional[str]) -> Optional[str]:
    """Явный system_prompt приоритетнее; иначе — встроенная персона; иначе None."""
    if system_prompt:
        return system_prompt
    if persona and persona in PERSONAS:
        return PERSONAS[persona].render()
    return None


@dataclass
class Guardrails:
    """Операционные ограничения ввода (НЕ контент-цензура)."""
    max_input_chars: int = 100_000
    deny_patterns: List[str] = field(default_factory=list)  # opt-in оператором; дефолт пусто

    def check(self, message: str) -> None:
        if len(message) > self.max_input_chars:
            raise GuardrailError(
                f"input too long: {len(message)} > {self.max_input_chars} chars")
        low = message.lower()
        for pat in self.deny_patterns:
            if pat and pat.lower() in low:
                raise GuardrailError("input matched a configured deny pattern")
