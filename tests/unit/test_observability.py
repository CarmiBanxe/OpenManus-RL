"""Тесты observability модуля (метрики / логи / health)."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.observability.health import (  # noqa: E402
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    get_health_checker,
)
from openmanus_rl.observability.logging import OpenManusLogger, get_logger  # noqa: E402
from openmanus_rl.observability.metrics import MetricsCollector, get_metrics_collector  # noqa: E402


class TestMetricsCollector(unittest.TestCase):
    def setUp(self) -> None:
        self.collector = MetricsCollector(port=9091)

    def test_init(self) -> None:
        self.assertEqual(self.collector.port, 9091)
        self.assertIsNotNone(self.collector.request_count)
        self.assertIsNotNone(self.collector.request_duration)
        self.assertIsNotNone(self.collector.active_requests)
        self.assertIsNotNone(self.collector._registry)

    def test_record_request_start(self) -> None:
        self.collector.record_request_start("litellm", "gpt-3.5-turbo")
        active = self.collector.get_metrics_summary().get("openmanus_active_requests", {})
        self.assertTrue(any(dict(labels).get("engine_type") == "litellm" for labels in active))

    def test_record_request_end(self) -> None:
        self.collector.record_request_end("litellm", "gpt-3.5-turbo", "success", "generate", 1.5, 20)
        rc = self.collector.get_metrics_summary().get("openmanus_requests_total", {})
        self.assertTrue(any(
            dict(labels).get("engine_type") == "litellm"
            and dict(labels).get("model") == "gpt-3.5-turbo"
            and dict(labels).get("status") == "success"
            for labels in rc))

    def test_record_error(self) -> None:
        self.collector.record_error("litellm", "timeout")
        ec = self.collector.get_metrics_summary().get("openmanus_errors_total", {})
        self.assertTrue(any(
            dict(labels).get("engine_type") == "litellm" and dict(labels).get("error_type") == "timeout"
            for labels in ec))

    def test_record_fallback(self) -> None:
        self.collector.record_fallback("litellm", "gpt-4", "gpt-3.5-turbo")
        fc = self.collector.get_metrics_summary().get("openmanus_fallback_total", {})
        self.assertTrue(any(
            dict(labels).get("from_model") == "gpt-4" and dict(labels).get("to_model") == "gpt-3.5-turbo"
            for labels in fc))

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.Process")
    def test_update_system_metrics(self, mock_process, mock_memory, mock_cpu) -> None:
        mock_cpu.return_value = 25.5
        mock_memory.return_value.percent = 60.0
        inst = MagicMock()
        inst.memory_info.return_value.rss = 1024 * 1024 * 100
        mock_process.return_value = inst

        self.collector.update_system_metrics()
        summary = self.collector.get_metrics_summary()
        self.assertEqual(summary.get("openmanus_system_cpu_usage_percent"), {(): 25.5})
        self.assertEqual(summary.get("openmanus_system_memory_usage_percent"), {(): 60.0})
        self.assertEqual(summary.get("openmanus_process_memory_usage_bytes"), {(): 1024 * 1024 * 100})

    def test_registry_isolation(self) -> None:
        c1, c2 = MetricsCollector(port=9092), MetricsCollector(port=9093)
        c1.record_request_start("litellm", "gpt-3.5-turbo")
        a1 = c1.get_metrics_summary().get("openmanus_active_requests", {})
        a2 = c2.get_metrics_summary().get("openmanus_active_requests", {})
        self.assertTrue(any(dict(labels).get("engine_type") == "litellm" for labels in a1))
        self.assertFalse(any(dict(labels).get("engine_type") == "litellm" for labels in a2))


class TestOpenManusLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = OpenManusLogger("test", "DEBUG")

    def test_init(self) -> None:
        self.assertEqual(self.logger.name, "test")
        self.assertEqual(self.logger.log_level, "DEBUG")
        self.assertIsNotNone(self.logger._logger)

    def test_request_logging(self) -> None:
        with patch.object(self.logger._logger, "info") as mock_info:
            self.logger.request(engine_type="litellm", model="gpt-3.5-turbo",
                               operation="generate", request_id="test-123", prompt="Hello")
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            self.assertEqual(args[0], "request_started")
            self.assertEqual(kwargs.get("engine_type"), "litellm")
            self.assertEqual(kwargs.get("request_id"), "test-123")
            self.assertEqual(kwargs.get("prompt"), "Hello")

    def test_response_logging(self) -> None:
        with patch.object(self.logger._logger, "info") as mock_info:
            self.logger.response(engine_type="litellm", model="gpt-3.5-turbo", operation="generate",
                                request_id="test-123", status="success", duration=1.5, tokens=20)
            args, kwargs = mock_info.call_args
            self.assertEqual(args[0], "request_completed")
            self.assertEqual(kwargs.get("status"), "success")
            self.assertEqual(kwargs.get("tokens"), 20)

    def test_error_logging(self) -> None:
        with patch.object(self.logger._logger, "error") as mock_error:
            self.logger.error(engine_type="litellm", operation="generate", request_id="test-123",
                             error="Connection timeout", error_type="timeout")
            args, kwargs = mock_error.call_args
            self.assertEqual(args[0], "request_failed")
            self.assertEqual(kwargs.get("error_type"), "timeout")

    def test_fallback_logging(self) -> None:
        with patch.object(self.logger._logger, "warning") as mock_warning:
            self.logger.fallback(engine_type="litellm", from_model="gpt-4",
                                to_model="gpt-3.5-turbo", request_id="test-123")
            args, kwargs = mock_warning.call_args
            self.assertEqual(args[0], "fallback_used")
            self.assertEqual(kwargs.get("to_model"), "gpt-3.5-turbo")

    def test_system_event_logging(self) -> None:
        with patch.object(self.logger._logger, "info") as mock_info:
            self.logger.system_event("engine_created", engine_type="litellm", model="gpt-3.5-turbo")
            args, kwargs = mock_info.call_args
            self.assertEqual(args[0], "engine_created")
            self.assertEqual(kwargs.get("engine_type"), "litellm")


class TestHealthChecker(unittest.TestCase):
    def setUp(self) -> None:
        self.health_checker = HealthChecker()

    def test_init(self) -> None:
        self.assertEqual(len(self.health_checker._components), 0)
        self.assertEqual(self.health_checker._global_status, HealthStatus.UNKNOWN)

    def test_register_component(self) -> None:
        self.health_checker.register_component("test", lambda: {"status": "healthy", "message": "OK"}, 10.0)
        self.assertIn("test", self.health_checker._components)
        self.assertEqual(self.health_checker._components["test"]["interval"], 10.0)

    def test_unregister_component(self) -> None:
        self.health_checker.register_component("test", lambda: {"status": "healthy", "message": "OK"})
        self.health_checker.unregister_component("test")
        self.assertNotIn("test", self.health_checker._components)

    def test_check_component(self) -> None:
        self.health_checker.register_component(
            "test", lambda: {"status": "healthy", "message": "OK", "details": {"version": "1.0"}})
        health = self.health_checker.check_component("test")
        self.assertEqual(health.status, HealthStatus.HEALTHY)
        self.assertEqual(health.details["version"], "1.0")

    def test_check_component_with_error(self) -> None:
        def boom():
            raise Exception("Test error")
        self.health_checker.register_component("test", boom)
        health = self.health_checker.check_component("test")
        self.assertEqual(health.status, HealthStatus.UNHEALTHY)
        self.assertIn("Check failed", health.message)
        self.assertEqual(health.details["error"], "Test error")

    def test_check_component_not_registered(self) -> None:
        health = self.health_checker.check_component("unknown")
        self.assertEqual(health.status, HealthStatus.UNKNOWN)
        self.assertIn("not registered", health.message)

    def test_check_all(self) -> None:
        self.health_checker.register_component("healthy", lambda: {"status": "healthy", "message": "OK"})
        self.health_checker.register_component("degraded", lambda: {"status": "degraded", "message": "Slow"})
        self.health_checker.register_component("unhealthy", lambda: {"status": "unhealthy", "message": "Err"})
        result = self.health_checker.check_all()
        self.assertEqual(result["status"], "unhealthy")
        for name in ("healthy", "degraded", "unhealthy"):
            self.assertIn(name, result["components"])

    def test_check_all_healthy(self) -> None:
        self.health_checker.register_component("c1", lambda: {"status": "healthy", "message": "OK"})
        self.health_checker.register_component("c2", lambda: {"status": "healthy", "message": "OK"})
        result = self.health_checker.check_all()
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["message"], "All components healthy")

    def test_check_all_degraded(self) -> None:
        self.health_checker.register_component("h", lambda: {"status": "healthy", "message": "OK"})
        self.health_checker.register_component("d", lambda: {"status": "degraded", "message": "Slow"})
        result = self.health_checker.check_all()
        self.assertEqual(result["status"], "degraded")

    def test_get_component_health(self) -> None:
        self.health_checker.register_component("test", lambda: {"status": "healthy", "message": "OK"})
        self.health_checker.check_component("test")
        health = self.health_checker.get_component_health("test")
        self.assertEqual(health.status, HealthStatus.HEALTHY)

    def test_get_global_health(self) -> None:
        result = self.health_checker.get_global_health()
        self.assertEqual(result["status"], "unknown")


class TestGlobalFunctions(unittest.TestCase):
    def test_get_metrics_collector(self) -> None:
        self.assertIs(get_metrics_collector(9092), get_metrics_collector(9092))

    def test_get_logger(self) -> None:
        self.assertIs(get_logger("test", "DEBUG"), get_logger("test", "DEBUG"))

    def test_get_health_checker(self) -> None:
        self.assertIs(get_health_checker(), get_health_checker())


if __name__ == "__main__":
    unittest.main()
