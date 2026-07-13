"""
Тесты Спринта 9 (perf-оптимизация) — под КОРРЕКТНЫЙ API.
Реальные: детект ресурсов, config-математика, snapshot, Ollama-availability.
Мок: Ollama HTTP-generate (не гоняем 7B в тесте).
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.optimized_ollama_engine import OptimizedOllamaEngine
from scripts.optimized_rollout import OptimizedRollout, RolloutConfig
from scripts.performance_monitor import PerformanceMonitor, PerformanceSnapshot
from scripts.performance_optimizer import PerformanceOptimizer, detect_system_resources


class TestPerformanceOptimizer(unittest.TestCase):
    def test_detect_real_resources(self) -> None:
        r = detect_system_resources()
        self.assertGreater(r.cpu_count, 0)
        self.assertGreater(r.memory_total, 0)
        self.assertIsInstance(r.gpu_available, bool)

    def test_optimize_math_and_config_roundtrip(self) -> None:
        tmp = Path(tempfile.mkdtemp()) / "performance.toml"
        opt = PerformanceOptimizer(str(tmp))
        strat = opt.optimize()
        for k in ("model_distribution", "ollama", "rollout", "monitoring"):
            self.assertIn(k, strat)
        self.assertTrue(tmp.exists())  # config записан
        # перечитывается корректно (tomllib)
        opt2 = PerformanceOptimizer(str(tmp))
        self.assertIn("ollama", opt2.config)

    def test_ollama_config_bounds(self) -> None:
        cfg = PerformanceOptimizer(str(Path(tempfile.mkdtemp()) / "p.toml")).optimize_ollama_config()
        self.assertGreaterEqual(cfg["max_concurrent_requests"], 1)
        self.assertGreaterEqual(cfg["gpu_layers"], 0)


class TestOptimizedOllamaEngine(unittest.TestCase):
    def test_construct_no_side_effects(self) -> None:
        # __init__ не пуллит/не создаёт модель — просто конструируется
        e = OptimizedOllamaEngine({"model": "qwen2.5:7b-instruct", "gpu_layers": 20})
        self.assertEqual(e.model, "qwen2.5:7b-instruct")
        self.assertIn("num_gpu", e._options({}))
        self.assertEqual(e._options({})["num_gpu"], 20)  # валидный параметр

    def test_is_available_real(self) -> None:
        # Ollama реально поднят на :11434
        self.assertIsInstance(OptimizedOllamaEngine().is_available(), bool)

    def test_generate_metrics_mocked_http(self) -> None:
        e = OptimizedOllamaEngine()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"response": "hi", "eval_count": 5, "prompt_eval_count": 3,
                                  "eval_duration": 1_000_000_000}
        with patch.object(e.session, "post", return_value=resp):
            out = e.generate("hello")
        self.assertEqual(out["response"], "hi")
        self.assertEqual(e.get_metrics()["successful_requests"], 1)
        self.assertGreater(e.get_metrics()["tokens_per_second"], 0)


class TestOptimizedRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_chunk_tasks(self) -> None:
        ro = OptimizedRollout(RolloutConfig(output_dir=self.tmp))
        chunks = ro.chunk_tasks([{"id": i} for i in range(10)], chunk_size=3)
        self.assertEqual([len(c) for c in chunks], [3, 3, 3, 1])

    def test_execute_task_no_zerodiv(self) -> None:
        # прямой вызов execute_task (total_tasks=0) НЕ должен падать ZeroDivisionError
        ro = OptimizedRollout(RolloutConfig(output_dir=self.tmp),
                              step_fn=lambda t: {"success": True, "action": "buy"})
        r = ro.execute_task({"id": "t1", "instruction": "x"})
        self.assertEqual(r["task_id"], "t1")
        self.assertTrue(r["success"])

    def test_execute_task_error_captured(self) -> None:
        def boom(_):
            raise RuntimeError("env fail")
        ro = OptimizedRollout(RolloutConfig(output_dir=self.tmp), step_fn=boom)
        r = ro.execute_task({"id": "t2"})
        self.assertFalse(r["success"])
        self.assertEqual(r["error"], "env fail")

    def test_run_rollout_parallel(self) -> None:
        ro = OptimizedRollout(RolloutConfig(max_workers=2, output_dir=self.tmp),
                              step_fn=lambda t: {"success": True})
        results = ro.run_rollout([{"id": i} for i in range(5)])
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["success"] for r in results))


class TestPerformanceMonitor(unittest.TestCase):
    def test_take_snapshot_real(self) -> None:
        snap = PerformanceMonitor({}).take_snapshot()
        self.assertIsInstance(snap, PerformanceSnapshot)
        self.assertGreaterEqual(snap.cpu_percent, 0.0)
        self.assertGreaterEqual(snap.memory_percent, 0.0)

    def test_no_thread_on_init(self) -> None:
        mon = PerformanceMonitor({})  # __init__ не стартует поток
        self.assertIsNone(mon._thread)

    def test_average_and_report(self) -> None:
        mon = PerformanceMonitor({})
        now = time.time()
        for i in range(3):
            mon.history.append(PerformanceSnapshot(
                timestamp=now - 10 * i, cpu_percent=20 + i * 10, memory_percent=50.0,
                memory_used=0, gpu_memory_percent=30.0, gpu_memory_used=0, gpu_utilization=40.0,
                disk_usage=70.0, network_sent=0, network_recv=0))
        avg = mon.get_average_metrics(duration=60)
        self.assertAlmostEqual(avg["avg_memory_percent"], 50.0)
        self.assertIn("OpenManus performance", mon.generate_report(60))


if __name__ == "__main__":
    unittest.main()
