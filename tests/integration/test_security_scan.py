"""
Тест security-скана — РЕАЛЬНЫЙ прогон bandit + gitleaks (не моки).
Наш authored-код без HIGH; рабочее дерево (минус вендоренное) без секретов.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.security_scan import SecurityScanner


class TestSecurityScan(unittest.TestCase):
    def test_bandit_authored_no_high(self) -> None:
        r = SecurityScanner().run_bandit()
        self.assertTrue(r["passed"], f"HIGH findings in authored code: {r.get('high')}")

    def test_gitleaks_working_tree_no_secrets(self) -> None:
        r = SecurityScanner().run_gitleaks()
        # gitleaks может отсутствовать (graceful-skip) — тогда passed=True со skipped-нотой
        self.assertTrue(r["passed"], f"secrets in working tree: {r.get('leaks')}")

    def test_scan_overall_green(self) -> None:
        self.assertTrue(SecurityScanner().scan())


if __name__ == "__main__":
    unittest.main()
