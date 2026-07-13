"""Тесты для scripts.validate_sprint.SprintValidator (реальные, mocked subprocess)."""
import os
import subprocess  # нужен для subprocess.TimeoutExpired (баг оригинала — отсутствовал импорт)
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.validate_sprint import SprintValidator


class TestSprintValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.v = SprintValidator()

    def test_init_uses_current_interpreter_by_default(self) -> None:
        self.assertEqual(self.v.python_executable, sys.executable)  # не хардкод
        self.assertTrue(self.v.project_root.exists())
        self.assertEqual(self.v.results["total_tests"], 0)

    def test_parse_counts(self) -> None:
        self.assertEqual(SprintValidator.parse_counts("13 passed in 1.2s"),
                         {"passed": 13, "failed": 0, "total": 13})
        self.assertEqual(SprintValidator.parse_counts("1 failed, 12 passed in 2s"),
                         {"passed": 12, "failed": 1, "total": 13})
        self.assertEqual(SprintValidator.parse_counts("2 error in 1s")["failed"], 2)
        self.assertEqual(SprintValidator.parse_counts("no summary here"),
                         {"passed": 0, "failed": 0, "total": 0})

    def test_run_test_suite_success(self) -> None:
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="3 passed", stderr="")):
            rc, out, err = self.v.run_test_suite("x")
        self.assertEqual((rc, out), (0, "3 passed"))

    def test_run_test_suite_timeout(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            rc, out, err = self.v.run_test_suite("x", timeout=1)
        self.assertEqual(rc, 1)
        self.assertIn("Timeout", err)

    def test_validate_group_counts_and_pass(self) -> None:
        # реальный существующий файл, но run_test_suite замокан -> быстро
        with patch.object(self.v, "run_test_suite", return_value=(0, "4 passed in 1s", "")):
            ok = self.v.validate_group("sprint6-ui", ["tests/integration/test_ui_smoke.py"])
        self.assertTrue(ok)
        self.assertEqual(self.v.results["passed_tests"], 4)
        self.assertEqual(self.v.results["failed_tests"], 0)

    def test_validate_group_missing_file_fails(self) -> None:
        ok = self.v.validate_group("x", ["tests/does_not_exist.py"])
        self.assertFalse(ok)
        self.assertTrue(any("missing" in e for e in self.v.results["errors"]))

    def test_validate_group_failure_propagates(self) -> None:
        with patch.object(self.v, "run_test_suite", return_value=(1, "1 failed, 0 passed", "boom")):
            ok = self.v.validate_group("sprint6-ui", ["tests/integration/test_ui_smoke.py"])
        self.assertFalse(ok)
        self.assertEqual(self.v.results["failed_tests"], 1)

    def test_generate_report(self) -> None:
        self.v.results["total_tests"] = 10
        self.v.results["passed_tests"] = 9
        self.v.results["failed_tests"] = 1
        self.v.results["test_suites"]["s"] = {"passed": 9, "failed": 1, "total": 10}
        report = self.v.generate_report()
        self.assertIn("total: 10", report)
        self.assertIn("passed: 9", report)


if __name__ == "__main__":
    unittest.main()
