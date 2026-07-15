"""Пороги регрессии для eval (S24) — гейт для CI по success-rate / латентности."""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class RegressionThresholds:
    min_success_rate: float = 1.0
    max_p95_latency_s: Optional[float] = None
    max_mean_latency_s: Optional[float] = None

    def check(self, report: Any) -> Tuple[bool, List[str]]:
        """-> (passed, violations). report — EvalReport-подобный (.summary)."""
        s = report.summary
        v: List[str] = []
        if s["success_rate"] < self.min_success_rate:
            v.append(f"success_rate {s['success_rate']:.3f} < {self.min_success_rate}")
        if self.max_p95_latency_s is not None and s["latency_p95"] > self.max_p95_latency_s:
            v.append(f"latency_p95 {s['latency_p95']:.3f}s > {self.max_p95_latency_s}s")
        if self.max_mean_latency_s is not None and s["latency_mean"] > self.max_mean_latency_s:
            v.append(f"latency_mean {s['latency_mean']:.3f}s > {self.max_mean_latency_s}s")
        return (len(v) == 0, v)
