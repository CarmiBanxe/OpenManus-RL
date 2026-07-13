#!/usr/bin/env python3
"""
Реальный валидатор спринтов OpenManus.

Прогоняет pytest по наборам тестов и ПАДАЕТ (exit!=0) при провале — вместо os.path.exists.
Без плагина pytest-json-report: разбирает summary-строку pytest (exit-code + счётчики).
Интерпретатор по умолчанию — sys.executable (не хардкод пути).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SUITES: Dict[str, List[str]] = {
    "sprint5": [
        "tests/unit/test_enhanced_decision_agent_sprint4.py",
        "tests/unit/test_legion_osint_integration.py",
        "tests/unit/test_mean_field_games.py",
        "tests/unit/test_performance_optimizer.py",
        "tests/integration/test_full_pipeline.py",
        "tests/integration/test_voice_to_decision.py",
        "tests/integration/test_multi_agent_scenario.py",
        "tests/integration/test_advanced_scenarios.py",
    ],
    "sprint6-p0": [
        "tests/unit/test_config.py",
        "tests/integration/test_api_smoke.py",
    ],
    "sprint6-ui": [
        "tests/integration/test_ui_smoke.py",
    ],
    "sprint7": [
        "tests/unit/test_validate_sprint.py",
        "tests/unit/test_security_validator.py",
    ],
    "perf": [
        "tests/integration/test_performance_validator.py",
    ],
    "backup": [
        "tests/unit/test_backup.py",
    ],
    "metrics": [
        "tests/integration/test_metrics.py",
    ],
}

_COUNT_RE = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "error": re.compile(r"(\d+) error"),
}


class SprintValidator:
    def __init__(self, python_executable: str = None) -> None:
        self.python_executable = python_executable or sys.executable
        self.project_root = Path(__file__).resolve().parent.parent
        self.results: Dict[str, object] = {
            "total_tests": 0, "passed_tests": 0, "failed_tests": 0,
            "errors": [], "test_suites": {},
        }

    def run_test_suite(self, test_path: str, timeout: int = 600) -> Tuple[int, str, str]:
        cmd = [self.python_executable, "-m", "pytest", test_path, "-q", "--no-header"]
        try:
            r = subprocess.run(cmd, cwd=self.project_root, capture_output=True,
                               text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Timeout ({timeout}s): {test_path}"
        except Exception as exc:  # noqa: BLE001
            return 1, "", f"Error running {test_path}: {exc}"

    @staticmethod
    def parse_counts(output: str) -> Dict[str, int]:
        def g(key: str) -> int:
            m = _COUNT_RE[key].search(output)
            return int(m.group(1)) if m else 0
        passed = g("passed")
        failed = g("failed") + g("error")
        return {"passed": passed, "failed": failed, "total": passed + failed}

    def validate_group(self, name: str, suites: List[str]) -> bool:
        print(f"\n=== {name} ===")
        ok = True
        for suite in suites:
            path = self.project_root / suite
            if not path.exists():
                self.results["errors"].append(f"missing: {suite}")
                print(f"❌ missing {suite}")
                ok = False
                continue
            rc, out, err = self.run_test_suite(str(path))
            counts = self.parse_counts(out)
            self.results["test_suites"][suite] = counts
            self.results["passed_tests"] += counts["passed"]
            self.results["failed_tests"] += counts["failed"]
            self.results["total_tests"] += counts["total"]
            if rc != 0:
                ok = False
                self.results["errors"].append(f"failed: {suite}")
                print(f"❌ {suite}  ({counts['passed']} passed, {counts['failed']} failed)")
                if err.strip():
                    print("   " + err.strip().splitlines()[-1])
            else:
                print(f"✅ {suite}  ({counts['passed']} passed)")
        return ok

    def generate_report(self) -> str:
        r = self.results
        rate = (r["passed_tests"] / r["total_tests"] * 100) if r["total_tests"] else 0.0
        lines = [
            "# Sprint validation report", "",
            f"- total: {r['total_tests']}", f"- passed: {r['passed_tests']}",
            f"- failed: {r['failed_tests']}", f"- success: {rate:.1f}%", "",
        ]
        for suite, c in r["test_suites"].items():
            lines.append(f"- {suite}: {c['passed']}/{c['total']} passed")
        if r["errors"]:
            lines += ["", "## errors"] + [f"- {e}" for e in r["errors"]]
        return "\n".join(lines)

    def validate(self, sprint: str = "all") -> bool:
        groups = SUITES if sprint == "all" else {sprint: SUITES[sprint]}
        ok = True
        for name, suites in groups.items():
            if not self.validate_group(name, suites):
                ok = False
        report = self.generate_report()
        print("\n" + report)
        (self.project_root / "sprint_validation_report.md").write_text(report, encoding="utf-8")
        print("\n" + ("✅ VALIDATION PASSED" if ok else "❌ VALIDATION FAILED"))
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenManus sprint validator")
    parser.add_argument("--sprint", choices=["all", *SUITES.keys()], default="all")
    parser.add_argument("--python", default=None, help="python executable (default: current)")
    args = parser.parse_args()
    ok = SprintValidator(args.python).validate(args.sprint)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
