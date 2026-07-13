"""Тесты network-валидатора — детерминированные (pure-функции на контролируемых входах)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.network_validator import classify_listeners, scan_config_egress, tor_available


class TestClassifyListeners(unittest.TestCase):
    def test_public_listener_flagged(self) -> None:
        ss = "LISTEN 0 128 0.0.0.0:4000 0.0.0.0:*\nLISTEN 0 128 127.0.0.1:8000 0.0.0.0:*\n"
        findings = classify_listeners(ss, watched={4000, 8000})
        self.assertEqual(findings, [{"port": 4000, "bind": "0.0.0.0"}])  # только 0.0.0.0:4000

    def test_localhost_only_clean(self) -> None:
        ss = "LISTEN 0 128 127.0.0.1:8000 0.0.0.0:*\nLISTEN 0 128 127.0.0.1:7860 0.0.0.0:*\n"
        self.assertEqual(classify_listeners(ss, watched={8000, 7860}), [])

    def test_ipv6_public_flagged(self) -> None:
        ss = "LISTEN 0 128 [::]:9090 [::]:*\n"
        findings = classify_listeners(ss, watched={9090})
        self.assertEqual(len(findings), 1)

    def test_unwatched_port_ignored(self) -> None:
        ss = "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n"
        self.assertEqual(classify_listeners(ss, watched={8000}), [])


class TestScanConfigEgress(unittest.TestCase):
    def test_localhost_endpoints_clean(self) -> None:
        cfg = 'base_url = "http://127.0.0.1:8080/v1"\nbase_url = "http://localhost:4000/v1"\n'
        self.assertEqual(scan_config_egress(cfg), [])

    def test_external_host_flagged(self) -> None:
        cfg = 'base_url = "https://api.evil.example/v1"\n'
        self.assertIn("api.evil.example", scan_config_egress(cfg))


class TestTorAvailable(unittest.TestCase):
    def test_returns_bool(self) -> None:
        self.assertIsInstance(tor_available(port=1, timeout=0.2), bool)  # закрытый порт -> False, но bool


if __name__ == "__main__":
    unittest.main()
