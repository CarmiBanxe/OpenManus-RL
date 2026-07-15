"""Тесты порогов регрессии (S24) + структуры live-наборов (без прогона)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.eval import EvalHarness, RegressionThresholds
from openmanus_rl.eval.harness import EvalCase, EvalSuite
from openmanus_rl.eval.suites import (LIVE_COMPONENTS, build_agent_recall_suite,
                                      build_live_suites, build_rag_live_suite)


def _report(passes):
    cases = [EvalCase(f"c{i}", run=lambda: 1, check=(lambda o, p=p: p), component="x")
             for i, p in enumerate(passes)]
    return EvalHarness().run([EvalSuite("s", cases)])


class TestRegressionThresholds(unittest.TestCase):
    def test_success_rate_pass(self):
        ok, v = RegressionThresholds(min_success_rate=1.0).check(_report([True, True]))
        self.assertTrue(ok)
        self.assertEqual(v, [])

    def test_success_rate_fail(self):
        ok, v = RegressionThresholds(min_success_rate=1.0).check(_report([True, False]))
        self.assertFalse(ok)
        self.assertTrue(any("success_rate" in x for x in v))

    def test_success_rate_lenient(self):
        ok, _ = RegressionThresholds(min_success_rate=0.5).check(_report([True, False]))
        self.assertTrue(ok)

    def test_p95_latency_gate(self):
        rep = _report([True])
        self.assertFalse(RegressionThresholds(min_success_rate=0.0, max_p95_latency_s=0.0).check(rep)[0])
        self.assertTrue(RegressionThresholds(min_success_rate=0.0, max_p95_latency_s=10.0).check(rep)[0])

    def test_mean_latency_gate(self):
        rep = _report([True, True])
        self.assertTrue(RegressionThresholds(min_success_rate=0.0, max_mean_latency_s=10.0).check(rep)[0])


class TestLiveSuiteBuilders(unittest.TestCase):
    def test_agent_recall_suite_structure(self):
        s = build_agent_recall_suite("smart")
        self.assertEqual(s.name, "agent_recall")
        self.assertGreaterEqual(len(s.cases), 1)
        self.assertEqual(s.cases[0].component, "agent_recall")

    def test_rag_live_suite_structure(self):
        s = build_rag_live_suite()
        self.assertEqual(s.name, "rag_live")
        self.assertGreaterEqual(len(s.cases), 1)

    def test_build_live_suites(self):
        names = {s.name for s in build_live_suites("smart")}
        self.assertEqual(names, {"live_llm", "agent_recall", "rag_live"})
        self.assertTrue(names.issubset(LIVE_COMPONENTS))


if __name__ == "__main__":
    unittest.main()
