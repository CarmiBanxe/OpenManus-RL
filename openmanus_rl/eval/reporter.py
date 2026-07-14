"""Генерация отчётов о результатах оценки (Markdown / JSON)."""
import json
from typing import Any


def to_markdown(report: Any) -> str:
    s = report.summary
    lines = [
        "# Eval report", "",
        f"- cases: {s['count']}  |  passed: {s['passed']}  |  "
        f"success: {s['success_rate'] * 100:.1f}%",
        f"- latency: mean {s['latency_mean'] * 1000:.1f}ms  "
        f"p50 {s['latency_p50'] * 1000:.1f}ms  p95 {s['latency_p95'] * 1000:.1f}ms",
        "", "## By component", "",
        "| component | cases | passed | success | p50 ms |",
        "|---|---|---|---|---|",
    ]
    for comp, agg in report.by_component().items():
        lines.append(
            f"| {comp} | {agg['count']} | {agg['passed']} | "
            f"{agg['success_rate'] * 100:.0f}% | {agg['latency_p50'] * 1000:.1f} |")
    lines += ["", "## Cases", "", "| component | case | passed | ms | error |",
              "|---|---|---|---|---|"]
    for r in report.results:
        lines.append(
            f"| {r.component} | {r.name} | {'✅' if r.passed else '❌'} | "
            f"{r.latency_s * 1000:.1f} | {(r.error or '')[:40]} |")
    return "\n".join(lines)


def to_json(report: Any) -> str:
    return json.dumps({
        "summary": report.summary,
        "by_component": report.by_component(),
        "results": [{"component": r.component, "name": r.name, "passed": r.passed,
                     "latency_s": r.latency_s, "error": r.error} for r in report.results],
    }, indent=2)
