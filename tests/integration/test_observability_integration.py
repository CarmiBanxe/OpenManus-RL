"""
Интеграционные тесты observability.

Изоляция: каждый тест использует СВОЙ MetricsCollector (не глобальный singleton),
чтобы счётчики были детерминированы (иначе конкурентный тест копил бы чужие запросы).
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.observability.health import get_health_checker  # noqa: E402
from openmanus_rl.observability.logging import get_logger  # noqa: E402
from openmanus_rl.observability.metrics import MetricsCollector  # noqa: E402


class TestObservabilityIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MetricsCollector(port=9093)  # свой экземпляр на каждый тест
        self.logger = get_logger("test", "DEBUG")
        self.health_checker = get_health_checker()

    def test_metrics_integration(self) -> None:
        self.metrics.record_request_start("litellm", "gpt-3.5-turbo")
        time.sleep(0.05)
        self.metrics.record_request_end("litellm", "gpt-3.5-turbo", "success", "generate", 0.05, 10)
        rc = self.metrics.get_metrics_summary().get("openmanus_requests_total", {})
        self.assertTrue(any(
            dict(labels).get("engine_type") == "litellm" and dict(labels).get("status") == "success"
            for labels in rc))

    def test_logging_integration(self) -> None:
        rid = "test-123"
        self.logger.request(engine_type="litellm", model="gpt-3.5-turbo",
                           operation="generate", request_id=rid, prompt="Hello")
        self.logger.response(engine_type="litellm", model="gpt-3.5-turbo", operation="generate",
                            request_id=rid, status="success", duration=0.1, tokens=10)

    def test_health_check_integration(self) -> None:
        self.health_checker.register_component(
            "test_engine", lambda: {"status": "healthy", "message": "Engine is available"}, 5.0)
        health = self.health_checker.check_component("test_engine")
        self.assertEqual(health.status.value, "healthy")
        self.assertEqual(health.message, "Engine is available")

    def test_concurrent_requests(self) -> None:
        def make_request(request_id: str) -> None:
            self.metrics.record_request_start("litellm", "gpt-3.5-turbo")
            time.sleep(0.02)
            self.logger.request(engine_type="litellm", model="gpt-3.5-turbo",
                               operation="generate", request_id=request_id, prompt="Hello")
            self.logger.response(engine_type="litellm", model="gpt-3.5-turbo", operation="generate",
                                request_id=request_id, status="success", duration=0.02, tokens=10)
            self.metrics.record_request_end("litellm", "gpt-3.5-turbo", "success", "generate", 0.02, 10)

        threads = [threading.Thread(target=make_request, args=(f"test-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rc = self.metrics.get_metrics_summary().get("openmanus_requests_total", {})
        total = sum(value for labels, value in rc.items() if dict(labels).get("engine_type") == "litellm")
        self.assertEqual(total, 5)


if __name__ == "__main__":
    unittest.main()
