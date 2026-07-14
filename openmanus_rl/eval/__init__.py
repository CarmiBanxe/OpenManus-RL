"""Eval-харнесс для компонентов движка (S16) — оценка S10–S15."""
from .harness import EvalCase, EvalHarness, EvalReport, EvalResult, EvalSuite
from .metrics import aggregate
from .reporter import to_json, to_markdown

__all__ = ["EvalCase", "EvalResult", "EvalSuite", "EvalHarness", "EvalReport",
           "aggregate", "to_markdown", "to_json"]
