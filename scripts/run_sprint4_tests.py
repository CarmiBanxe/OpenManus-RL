#!/usr/bin/env python3
"""
Скрипт для запуска тестов Спринта 4.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Bootstrap: install missing core deps before anything else
# ---------------------------------------------------------------------------

_CORE_DEPS = ["numpy", "psutil", "pytest", "pytest-asyncio"]


def _ensure_deps() -> None:
    import importlib

    missing = []
    check_map = {
        "numpy": "numpy",
        "psutil": "psutil",
        "pytest": "pytest",
        "pytest-asyncio": "pytest_asyncio",
    }
    for pkg, mod in check_map.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing core deps: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )


_ensure_deps()


def run_command(command: str, description: str) -> bool:
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print("=" * 50)

    start = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)  # nosec B602 — internally-constructed trusted test commands
    elapsed = time.time() - start

    print(f"Elapsed: {elapsed:.2f}s | Return code: {result.returncode}")

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    return result.returncode == 0


def main() -> int:
    print("Sprint 4 Test Runner")
    print("=" * 50)

    tests = [
        (
            "python -m pytest tests/unit/test_qwen3_omni_integration.py -v --tb=short",
            "Qwen3-Omni Integration Tests",
        ),
        (
            "python -m pytest tests/unit/test_deep_hedging.py -v --tb=short",
            "Deep Hedging Tests",
        ),
        (
            "python -m pytest tests/unit/test_signature_methods.py -v --tb=short",
            "Signature Methods Tests",
        ),
        (
            (
                "python -c \""
                "from openmanus_rl.decision.mean_field_games import MeanFieldGames; "
                "mfg = MeanFieldGames({'num_agents': 10}); "
                "r = mfg.solve_mfg(); "
                "assert r['success'], r; "
                "print('Mean Field Games: OK')\""
            ),
            "Mean Field Games Import + Smoke Test",
        ),
        (
            (
                "python -c \""
                "from openmanus_rl.optimization.performance_optimizer import PerformanceOptimizer; "
                "p = PerformanceOptimizer(); "
                "s = p.get_performance_stats(); "
                "assert 'current_memory_usage' in s; "
                "p.cleanup(); "
                "print('Performance Optimizer: OK')\""
            ),
            "Performance Optimizer Import + Smoke Test",
        ),
        (
            (
                "python -c \""
                "from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent; "
                "agent = EnhancedDecisionAgent({'enable_mean_field_games': True, 'enable_performance_optimization': True}); "
                "print('Enhanced Decision Agent Sprint 4: OK')\""
            ),
            "Enhanced Decision Agent Sprint 4 Test",
        ),
    ]

    results: list[tuple[str, bool]] = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))

    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    for description, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {status}  {description}")
        if success:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} passed")

    if passed == len(results):
        print("\nAll tests passed. Sprint 4 is ready.")
        return 0
    else:
        print(f"\n{len(results) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
