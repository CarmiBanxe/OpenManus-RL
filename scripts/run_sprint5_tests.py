#!/usr/bin/env python3
"""Runner для интеграционных тестов Спринта 5 (реальный API).

--smoke : по одному тесту из каждого набора
--full  : все интеграционные тесты
--test X: конкретный путь
По умолчанию — smoke. Использует sys.executable (тот же интерпретатор, что запустил скрипт).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

FULL = [
    "tests/integration/test_full_pipeline.py",
    "tests/integration/test_voice_to_decision.py",
    "tests/integration/test_multi_agent_scenario.py",
    "tests/integration/test_advanced_scenarios.py",
]
SMOKE = [
    "tests/integration/test_full_pipeline.py::TestFullPipeline::test_text_input_pipeline",
    "tests/integration/test_voice_to_decision.py::TestVoiceToDecision::test_voice_select_action",
    "tests/integration/test_multi_agent_scenario.py::TestMultiAgentScenario::test_scenarios",
    "tests/integration/test_advanced_scenarios.py::TestAdvancedScenarios::test_decision_then_multiagent_then_optimize",
]


def _run(targets: list) -> int:
    cmd = [sys.executable, "-m", "pytest", "-v", *targets]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 5 integration test runner")
    parser.add_argument("--smoke", action="store_true", help="one test per suite")
    parser.add_argument("--full", action="store_true", help="all integration tests")
    parser.add_argument("--test", help="specific test path")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    if not os.path.isdir("tests/integration"):
        print("ERROR: tests/integration not found")
        return 1

    if args.test:
        return _run([args.test])
    if args.full:
        return _run(FULL)
    return _run(SMOKE)


if __name__ == "__main__":
    sys.exit(main())
