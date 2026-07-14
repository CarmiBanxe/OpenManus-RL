"""Агрегация метрик оценки (success-rate, латентность p50/p95)."""
from typing import Any, Dict, List

import numpy as np


def aggregate(results: List[Any]) -> Dict[str, Any]:
    """results: список EvalResult-подобных объектов (.passed, .latency_s)."""
    n = len(results)
    if n == 0:
        return {"count": 0, "passed": 0, "success_rate": 0.0,
                "latency_mean": 0.0, "latency_p50": 0.0, "latency_p95": 0.0}
    passed = sum(1 for r in results if r.passed)
    lat = np.array([r.latency_s for r in results], dtype=float)
    return {
        "count": n,
        "passed": passed,
        "success_rate": passed / n,
        "latency_mean": float(lat.mean()),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
    }
