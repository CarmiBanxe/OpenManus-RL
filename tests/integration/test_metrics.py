"""
Тест защищённого /metrics — реальные Prometheus-метрики, auth-gated (не публичный).
In-process через TestClient (без живого сервера).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["OPENMANUS_CONFIG_FILE"] = "testing"
os.environ["OPENMANUS_SECRET_KEY"] = "metrics-secret-key-at-least-32-bytes-xx"
os.environ["OPENMANUS_ADMIN_USER"] = "admin"
os.environ["OPENMANUS_ADMIN_PASSWORD"] = "secret123"

from fastapi.testclient import TestClient  # noqa: E402

from openmanus_rl.api.server import app  # noqa: E402


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ctx = TestClient(app)
        cls.client = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ctx.__exit__(None, None, None)

    def _token(self) -> str:
        return self.client.post("/auth/login",
                                json={"username": "admin", "password": "secret123"}).json()["access_token"]

    def test_metrics_requires_auth(self) -> None:
        r = self.client.get("/metrics")
        self.assertIn(r.status_code, (401, 403))  # красная линия: не публичный

    def test_metrics_authenticated_real_data(self) -> None:
        token = self._token()
        r = self.client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        body = r.text
        # реальные (не мок) метрики присутствуют
        self.assertIn("openmanus_requests_total", body)
        self.assertIn("openmanus_memory_usage_bytes", body)
        self.assertIn("openmanus_request_latency_seconds", body)

    def test_request_counter_increments(self) -> None:
        token = self._token()
        # делаем запрос, затем метрики должны отражать его
        self.client.get("/health")
        r = self.client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
        self.assertIn('openmanus_requests_total{', r.text)
        self.assertIn("/health", r.text)


if __name__ == "__main__":
    unittest.main()
