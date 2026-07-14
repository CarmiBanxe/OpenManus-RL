"""
Ядро eval-харнесса (S16): прогон наборов оценочных кейсов с таймингом.

EvalCase.run() -> output; EvalCase.check(output) -> passed. Харнесс замеряет
латентность, ловит ошибки, агрегирует. Опционально пишет латентность в S11
observability (record_custom_metric). Детерминированные кейсы — зелёный гейт;
live-LLM кейсы регистрируются вызывающим кодом отдельно (могут быть под skip).
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .metrics import aggregate

try:
    from openmanus_rl.observability import get_metrics_collector
    _OBS = True
except ImportError:  # pragma: no cover
    _OBS = False


@dataclass
class EvalCase:
    name: str
    run: Callable[[], Any]
    check: Callable[[Any], bool]
    component: str = "general"


@dataclass
class EvalResult:
    name: str
    component: str
    passed: bool
    latency_s: float
    output: Any = None
    error: Optional[str] = None


@dataclass
class EvalSuite:
    name: str
    cases: List[EvalCase] = field(default_factory=list)

    def add(self, case: EvalCase) -> "EvalSuite":
        self.cases.append(case)
        return self


@dataclass
class EvalReport:
    results: List[EvalResult]

    @property
    def summary(self) -> dict:
        return aggregate(self.results)

    def by_component(self) -> dict:
        out: dict = {}
        for r in self.results:
            out.setdefault(r.component, []).append(r)
        return {c: aggregate(rs) for c, rs in out.items()}


class EvalHarness:
    """Прогоняет EvalSuite'ы, собирает EvalResult'ы с латентностью."""

    def __init__(self, record_metrics: bool = False) -> None:
        self.record_metrics = record_metrics and _OBS
        self._metrics = get_metrics_collector() if self.record_metrics else None

    def run_case(self, case: EvalCase) -> EvalResult:
        start = time.perf_counter()
        output, error, passed = None, None, False
        try:
            output = case.run()
            passed = bool(case.check(output))
        except Exception as exc:  # noqa: BLE001  (провал кейса не должен ронять прогон)
            error = str(exc)
            passed = False
        latency = time.perf_counter() - start
        if self._metrics is not None:
            self._metrics.record_custom_metric(
                "openmanus_eval_latency_seconds", latency,
                {"component": case.component, "case": case.name})
        return EvalResult(case.name, case.component, passed, latency, output, error)

    def run(self, suites: List[EvalSuite]) -> EvalReport:
        results: List[EvalResult] = []
        for suite in suites:
            for case in suite.cases:
                results.append(self.run_case(case))
        return EvalReport(results)
