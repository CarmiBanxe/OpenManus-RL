"""Детерминированные eval-наборы (tools/memory/rag) должны быть 100% зелёными."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.eval.harness import EvalHarness
from openmanus_rl.eval.suites import build_default_suites


class TestEvalSuites(unittest.TestCase):
    def setUp(self):
        self.report = EvalHarness().run(build_default_suites())

    def test_all_deterministic_pass(self):
        s = self.report.summary
        self.assertEqual(s["passed"], s["count"], msg=f"failures: "
                         f"{[r.name for r in self.report.results if not r.passed]}")
        self.assertEqual(s["success_rate"], 1.0)

    def test_components_present(self):
        bc = self.report.by_component()
        for comp in ("tools", "memory", "rag"):
            self.assertIn(comp, bc)
            self.assertEqual(bc[comp]["success_rate"], 1.0)

    def test_has_cases(self):
        self.assertGreaterEqual(self.report.summary["count"], 6)


if __name__ == "__main__":
    unittest.main()
