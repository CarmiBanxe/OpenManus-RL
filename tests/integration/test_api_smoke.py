"""
Smoke-тест FastAPI-сервера OpenManus — реальный startup + auth + select_action.

Использует fastapi TestClient (поднимает приложение in-process, без живого сервера).
Проверяет security-контур: /query без токена -> 401/403; с токеном -> реальный select_action.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# env до входа в lifespan (агент строится в testing-sandbox)
os.environ["OPENMANUS_CONFIG_FILE"] = "testing"
os.environ["OPENMANUS_SECRET_KEY"] = "test-secret-key"
os.environ["OPENMANUS_ADMIN_USER"] = "admin"
os.environ["OPENMANUS_ADMIN_PASSWORD"] = "secret123"

from fastapi.testclient import TestClient  # noqa: E402

from openmanus_rl.api.server import app  # noqa: E402


class TestApiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()  # триггерит lifespan startup (строит агента)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_ctx.__exit__(None, None, None)  # lifespan shutdown -> agent.cleanup()

    def _token(self) -> str:
        r = self.client.post("/auth/login", json={"username": "admin", "password": "secret123"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["access_token"]

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["agent"])
        self.assertEqual(r.json()["environment"], "testing")

    def test_query_requires_auth(self) -> None:
        r = self.client.post("/query", json={"text": "hi", "available_actions": ["buy", "sell"]})
        self.assertIn(r.status_code, (401, 403))  # без токена — отказ

    def test_login_bad_credentials(self) -> None:
        r = self.client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_query_authenticated_real_select_action(self) -> None:
        token = self._token()
        r = self.client.post(
            "/query",
            json={"text": "What is the risk of BTC?", "available_actions": ["buy", "sell", "wait"],
                  "entities": ["BTC"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "success")
        result = body["result"]
        for key in ("action", "confidence", "osint_enhanced", "episode_id", "timestamp"):
            self.assertIn(key, result)
        self.assertIn(result["action"], ["buy", "sell", "wait", "error"])

    def test_config_endpoint_authenticated(self) -> None:
        token = self._token()
        r = self.client.get("/config", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        cfg = r.json()
        self.assertEqual(cfg["host"], "127.0.0.1")       # security default
        self.assertFalse(cfg["gradio_share"])
        self.assertNotIn("*", cfg["cors_allow_origins"])


if __name__ == "__main__":
    unittest.main()
