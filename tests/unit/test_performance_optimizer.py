"""
Unit-тесты для PerformanceOptimizer (Sprint 4) — под РЕАЛЬНЫЙ API.

Реальные методы: __init__(config), get_performance_stats(), optimize_inference(model, inputs),
optimize_batch_processing(items, func, batch_size), _get_gpu_memory_usage(), cleanup().
Реальные атрибуты: max_memory_usage, max_cpu_usage, max_gpu_memory_usage, monitoring_active,
performance_stats{memory_usage, gpu_memory_usage, cpu_usage, processing_times}.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.optimization.performance_optimizer import PerformanceOptimizer


class TestPerformanceOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"max_memory_usage": 0.8, "max_cpu_usage": 0.9}
        self.optimizer = PerformanceOptimizer(self.config)

    def tearDown(self) -> None:
        self.optimizer.cleanup()

    def test_initialization(self) -> None:
        self.assertEqual(self.optimizer.max_memory_usage, 0.8)
        self.assertEqual(self.optimizer.max_cpu_usage, 0.9)
        self.assertEqual(self.optimizer.max_gpu_memory_usage, 0.8)  # default
        self.assertFalse(self.optimizer.monitoring_active)
        for key in ("memory_usage", "gpu_memory_usage", "cpu_usage", "processing_times"):
            self.assertIn(key, self.optimizer.performance_stats)

    def test_get_performance_stats(self) -> None:
        stats = self.optimizer.get_performance_stats()
        for key in (
            "current_memory_usage",
            "current_cpu_usage",
            "current_gpu_memory_usage",
            "processing_times",
            "processing_times_by_operation",
        ):
            self.assertIn(key, stats)
        self.assertIsInstance(stats["current_memory_usage"], float)
        self.assertGreaterEqual(stats["current_memory_usage"], 0.0)
        self.assertLessEqual(stats["current_memory_usage"], 1.0)

    def test_optimize_inference(self) -> None:
        model = lambda x: x * 2  # noqa: E731 — callable без .quantize -> else-ветка
        result = self.optimizer.optimize_inference(model, 5, use_quantization=False)
        self.assertEqual(result, 10)
        ops = [e["operation"] for e in self.optimizer.performance_stats["processing_times"]]
        self.assertIn("inference", ops)

    def test_optimize_batch_processing(self) -> None:
        result = self.optimizer.optimize_batch_processing(
            [1, 2, 3, 4], lambda x: x * 2, batch_size=2
        )
        self.assertEqual(sorted(result), [2, 4, 6, 8])

    def test_get_gpu_memory_usage_real(self) -> None:
        usage = self.optimizer._get_gpu_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertGreaterEqual(usage, 0.0)

    def test_get_gpu_memory_usage_mocked(self) -> None:
        mod = "openmanus_rl.optimization.performance_optimizer"
        with patch(f"{mod}._TORCH_AVAILABLE", True), \
             patch(f"{mod}.torch.cuda.is_available", return_value=True), \
             patch(f"{mod}.torch.cuda.memory_allocated", return_value=4000), \
             patch(f"{mod}.torch.cuda.max_memory_allocated", return_value=8000):
            self.assertEqual(self.optimizer._get_gpu_memory_usage(), 0.5)

    def test_cleanup(self) -> None:
        self.optimizer.cleanup()
        self.assertFalse(self.optimizer.monitoring_active)
        self.assertEqual(self.optimizer.performance_stats, {})


if __name__ == "__main__":
    unittest.main()
