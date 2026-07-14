"""Тесты ядра eval-харнесса (harness/metrics/reporter)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.eval.harness import EvalCase, EvalHarness, EvalSuite
from openmanus_rl.eval.reporter import to_json, to_markdown


def _boom():
    raise ValueError("boom")


class TestEvalHarness(unittest.TestCase):
    def setUp(self):
        self.suite = EvalSuite("s", [
            EvalCase("passes", run=lambda: 42, check=lambda o: o == 42, component="c1"),
            EvalCase("fails", run=lambda: 1, check=lambda o: o == 2, component="c1"),
            EvalCase("errors", run=_boom, check=lambda o: True, component="c2"),
        ])
        self.report = EvalHarness().run([self.suite])

    def test_results_counts(self):
        s = self.report.summary
        self.assertEqual(s["count"], 3)
        self.assertEqual(s["passed"], 1)
        self.assertAlmostEqual(s["success_rate"], 1 / 3, places=5)

    def test_error_captured(self):
        err = next(r for r in self.report.results if r.name == "errors")
        self.assertFalse(err.passed)
        self.assertIn("boom", err.error)

    def test_latency_recorded(self):
        self.assertTrue(all(r.latency_s >= 0 for r in self.report.results))

    def test_by_component(self):
        bc = self.report.by_component()
        self.assertIn("c1", bc)
        self.assertIn("c2", bc)
        self.assertEqual(bc["c1"]["count"], 2)
        self.assertEqual(bc["c1"]["passed"], 1)

    def test_markdown_report(self):
        md = to_markdown(self.report)
        self.assertIn("# Eval report", md)
        self.assertIn("By component", md)
        self.assertIn("passes", md)

    def test_json_report(self):
        data = json.loads(to_json(self.report))
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["count"], 3)
        self.assertEqual(len(data["results"]), 3)


if __name__ == "__main__":
    unittest.main()
