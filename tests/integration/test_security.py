"""Тесты hardening REST-сервиса (S19): rate-limit, audit-redact, SecurityAudit."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.api.security import (RateLimiter, SecurityAudit, _redact_client,
                                       install_security)


class TestRateLimiter(unittest.TestCase):
    def test_allows_up_to_limit_then_denies(self):
        rl = RateLimiter(limit=3, window_s=60)
        self.assertEqual([rl.check("c")[0] for _ in range(3)], [True, True, True])
        allowed, retry = rl.check("c")
        self.assertFalse(allowed)
        self.assertGreater(retry, 0)

    def test_per_client_isolation(self):
        rl = RateLimiter(limit=1, window_s=60)
        self.assertTrue(rl.check("a")[0])
        self.assertTrue(rl.check("b")[0])
        self.assertFalse(rl.check("a")[0])

    def test_reset(self):
        rl = RateLimiter(limit=1, window_s=60)
        rl.check("c")
        rl.reset()
        self.assertTrue(rl.check("c")[0])


class TestRedact(unittest.TestCase):
    def test_redacts_key_keeps_ip(self):
        self.assertEqual(_redact_client("sk-abcdef1234567890"), "sk-abcde…")
        self.assertEqual(_redact_client("127.0.0.1"), "127.0.0.1")


class TestSecurityAudit(unittest.TestCase):
    def test_safe_localhost_passes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGION_API_KEY", None)
            findings = SecurityAudit.run({}, bind_host="127.0.0.1")
        self.assertTrue(SecurityAudit.summary(findings)["passed"])

    def test_hardcoded_master_key_fails(self):
        findings = SecurityAudit.run({"master_key": "sk-xxx"}, bind_host="127.0.0.1")
        s = SecurityAudit.summary(findings)
        self.assertFalse(s["passed"])
        self.assertFalse(next(f for f in s["checks"] if f["check"] == "no_hardcoded_master_key")["passed"])

    def test_public_bind_without_key_fails(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGION_API_KEY", None)
            findings = SecurityAudit.run({}, bind_host="0.0.0.0")
        s = SecurityAudit.summary(findings)
        self.assertFalse(s["passed"])
        self.assertFalse(next(f for f in s["checks"] if f["check"] == "bind_localhost")["passed"])


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _FASTAPI = True
except ImportError:
    _FASTAPI = False


@unittest.skipIf(not _FASTAPI, "FastAPI not available")
class TestRateLimitMiddleware(unittest.TestCase):
    def test_429_after_limit(self):
        app = FastAPI()

        @app.get("/ping")
        def ping():
            return {"ok": True}

        install_security(app, rate_limit=3, audit=False)
        client = TestClient(app)
        codes = [client.get("/ping").status_code for _ in range(4)]
        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertEqual(codes[3], 429)

    def test_security_audit_endpoint(self):
        from openmanus_rl.api.agent_server import create_agent_app
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGION_API_KEY", None)
            client = TestClient(create_agent_app({
                "model": "smart", "tools": False, "rag": False,
                "memory": True, "memory_db": ":memory:"}))
            r = client.get("/security/audit")
        self.assertEqual(r.status_code, 200)
        self.assertIn("passed", r.json())
        self.assertIn("checks", r.json())


if __name__ == "__main__":
    unittest.main()
