"""Интеграционные тесты health-check эндпоинта (FastAPI TestClient)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from fastapi.testclient import TestClient

    from openmanus_rl.api.health import create_health_app

    HEALTH_ENDPOINT_AVAILABLE = True
except ImportError:
    HEALTH_ENDPOINT_AVAILABLE = False


@unittest.skipIf(not HEALTH_ENDPOINT_AVAILABLE, "FastAPI or health endpoint not available")
class TestHealthEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_health_app())

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in ("status", "message", "components"):
            self.assertIn(key, data)

    def test_component_health_check(self) -> None:
        response = self.client.get("/health/ObservableEngineFactory")
        self.assertIn(response.status_code, (200, 404))
        if response.status_code == 200:
            data = response.json()
            for key in ("status", "message", "last_check"):
                self.assertIn(key, data)

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        text = response.text
        self.assertIn("openmanus_requests_total", text)
        self.assertIn("openmanus_request_duration_seconds", text)

    def test_engines_endpoint(self) -> None:
        response = self.client.get("/engines")
        self.assertEqual(response.status_code, 200)
        engines = response.json()["engines"]
        for name in ("ollama", "openai", "optimized_ollama", "litellm"):
            self.assertIn(name, engines)

    def test_observability_endpoint(self) -> None:
        response = self.client.get("/observability")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in ("metrics", "health", "engines"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
