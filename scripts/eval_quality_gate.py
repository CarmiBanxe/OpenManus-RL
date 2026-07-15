"""
Eval quality gate — exits non-zero if any threshold in eval_gate_config.yaml is breached.

Usage:
    python scripts/eval_quality_gate.py                          # run full suite + gate
    python scripts/eval_quality_gate.py --report eval_out.json  # gate against existing report
    python scripts/eval_quality_gate.py --update-baseline        # run + write new baseline

Exit codes:
    0 — all thresholds passed
    1 — one or more thresholds breached (regression / low pass rate / high latency)
    2 — configuration / environment error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "openmanus_rl" / "config" / "eval_gate_config.yaml"
_BASELINE_PATH = _REPO_ROOT / "scripts" / "eval_baseline.json"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.error("Config not found: %s", path)
        sys.exit(2)
    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Run eval suite
# ---------------------------------------------------------------------------

def run_eval_suite() -> dict[str, Any]:
    """Run the Legion eval harness and return a summary dict."""
    try:
        from openmanus_rl.eval.harness import EvalHarness  # type: ignore[import-not-found]
        from openmanus_rl.eval.suites import get_all_suites  # type: ignore[import-not-found]

        harness = EvalHarness()
        for suite in get_all_suites():
            harness.run_suite(suite)
        report = harness.get_report()
        return report.summary
    except ImportError:
        # Minimal fallback: run pytest on eval tests and parse output
        logger.warning("EvalHarness not importable; running pytest fallback")
        return _pytest_fallback()


def _pytest_fallback() -> dict[str, Any]:
    """Run pytest on tests/ and produce a minimal summary dict."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--no-header",
         "--json-report", "--json-report-file=/tmp/eval_report.json"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT)
    )
    try:
        with open("/tmp/eval_report.json") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        return {
            "total": total,
            "passed": passed,
            "failed": summary.get("failed", 0),
            "pass_rate": (passed / total) if total else 0.0,
            "p95_latency_s": 0.0,   # not available from pytest alone
            "by_component": {},
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Last resort: parse stdout
        passed = failed = 0
        for line in result.stdout.splitlines():
            if "passed" in line:
                for tok in line.split():
                    if tok.isdigit():
                        passed = int(tok)
                        break
            if "failed" in line:
                for tok in line.split():
                    if tok.isdigit():
                        failed = int(tok)
                        break
        total = passed + failed
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total) if total else 0.0,
            "p95_latency_s": 0.0,
            "by_component": {},
        }


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def check_gate(summary: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Return list of failure messages (empty = all passed)."""
    gate = cfg.get("gate", {})
    failures: list[str] = []

    total = summary.get("total", 0)
    min_count = gate.get("min_case_count", 5)
    if total < min_count:
        failures.append(f"Too few eval cases: {total} < {min_count} (min_case_count)")

    pass_rate = summary.get("pass_rate", 0.0)
    min_pr = gate.get("min_pass_rate", 0.80)
    if pass_rate < min_pr:
        failures.append(
            f"Pass rate {pass_rate:.1%} below threshold {min_pr:.1%} (min_pass_rate)"
        )

    p95 = summary.get("p95_latency_s", 0.0)
    max_p95 = gate.get("max_p95_latency_s", 10.0)
    if p95 and p95 > max_p95:
        failures.append(
            f"p95 latency {p95:.1f}s exceeds threshold {max_p95:.1f}s (max_p95_latency_s)"
        )

    # Per-component thresholds
    component_thresholds: dict[str, float] = gate.get("components", {})
    by_component: dict[str, Any] = summary.get("by_component", {})
    for component, threshold in component_thresholds.items():
        comp_data = by_component.get(component, {})
        comp_pr = comp_data.get("pass_rate", pass_rate)  # fall back to global if missing
        if comp_pr < threshold:
            failures.append(
                f"Component '{component}' pass rate {comp_pr:.1%} "
                f"below threshold {threshold:.1%}"
            )

    # Regression guard
    reg_cfg = cfg.get("regression", {})
    if reg_cfg.get("enabled", False) and _BASELINE_PATH.exists():
        try:
            with _BASELINE_PATH.open() as f:
                baseline = json.load(f)
            baseline_pr = baseline.get("pass_rate", 0.0)
            max_drop = reg_cfg.get("max_regression_fraction", 0.05)
            if pass_rate < baseline_pr - max_drop:
                failures.append(
                    f"Regression: pass rate dropped from {baseline_pr:.1%} "
                    f"to {pass_rate:.1%} (>{max_drop:.0%} drop)"
                )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not read baseline: %s", exc)

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Legion eval quality gate")
    parser.add_argument("--report", help="Path to existing JSON eval report (skip running suite)")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write passing summary as new baseline after successful gate")
    parser.add_argument("--config", default=str(_CONFIG_PATH),
                        help="Path to eval_gate_config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            logger.error("Report file not found: %s", report_path)
            return 2
        with report_path.open() as f:
            summary: dict[str, Any] = json.load(f)
    else:
        logger.info("Running eval suite…")
        summary = run_eval_suite()

    logger.info(
        "Eval summary: total=%d passed=%d pass_rate=%.1f%% p95=%.2fs",
        summary.get("total", 0),
        summary.get("passed", 0),
        summary.get("pass_rate", 0.0) * 100,
        summary.get("p95_latency_s", 0.0),
    )

    failures = check_gate(summary, cfg)
    if failures:
        logger.error("EVAL GATE FAILED — %d issue(s):", len(failures))
        for msg in failures:
            logger.error("  ✗ %s", msg)
        return 1

    logger.info("EVAL GATE PASSED")
    if args.update_baseline:
        _BASELINE_PATH.write_text(json.dumps(summary, indent=2))
        logger.info("Baseline updated: %s", _BASELINE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
