"""
Performance Optimizer — оптимизация ресурсов и производительности.
Sprint 4 | Мониторинг CPU/RAM/GPU, пулы потоков, автоматическая GC.
"""
from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Оптимизатор производительности для компонентов OpenManus."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        self.max_memory_usage: float = self.config.get("max_memory_usage", 0.8)
        self.max_gpu_memory_usage: float = self.config.get("max_gpu_memory_usage", 0.8)
        self.max_cpu_usage: float = self.config.get("max_cpu_usage", 0.8)

        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get("thread_workers", 4))
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.get("process_workers", 2))

        self.task_queue: queue.Queue[Any] = queue.Queue()

        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        self.performance_stats: Dict[str, List[Any]] = {
            "memory_usage": [],
            "gpu_memory_usage": [],
            "cpu_usage": [],
            "processing_times": [],
        }

        logger.info("PerformanceOptimizer initialized")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def optimize_model_loading(
        self,
        model_class: type,
        model_config: Dict[str, Any],
    ) -> Any:
        """Оптимизированная загрузка модели с проверкой памяти."""
        try:
            if not self._check_memory_availability():
                self._cleanup_memory()

            start = time.time()

            if self._is_large_model(model_config):
                model = self._load_model_in_thread(model_class, model_config)
            else:
                model = model_class(model_config)

            self.performance_stats["processing_times"].append(
                {"operation": "model_loading", "time": time.time() - start}
            )
            logger.info("Model loaded in %.2f s", time.time() - start)
            return model

        except Exception as exc:
            logger.error("Model loading optimization error: %s", exc)
            return model_class(model_config)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def optimize_batch_processing(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        batch_size: int = 32,
    ) -> List[Any]:
        """Параллельная пакетная обработка через ThreadPoolExecutor."""
        try:
            batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
            futures = [
                self.thread_pool.submit(self._process_batch, batch, process_func)
                for batch in batches
            ]
            results: List[Any] = []
            for fut in futures:
                results.extend(fut.result())
            return results

        except Exception as exc:
            logger.error("Batch processing optimization error: %s", exc)
            return [process_func(item) for item in items]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def optimize_inference(
        self,
        model: Any,
        inputs: Any,
        use_quantization: bool = True,
    ) -> Any:
        """Оптимизированное выполнение вывода."""
        try:
            if not self._check_memory_availability():
                self._cleanup_memory()

            start = time.time()

            if use_quantization and hasattr(model, "quantize"):
                result = model.quantize()(inputs)
            else:
                result = model(inputs)

            self.performance_stats["processing_times"].append(
                {"operation": "inference", "time": time.time() - start}
            )
            return result

        except Exception as exc:
            logger.error("Inference optimization error: %s", exc)
            return model(inputs)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Запуск фонового мониторинга ресурсов."""
        if self.monitoring_active:
            return
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True,
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started (interval=%.1f s)", interval)

    def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Агрегированная статистика производительности."""
        by_op: Dict[str, List[float]] = {}
        for entry in self.performance_stats["processing_times"]:
            by_op.setdefault(entry["operation"], []).append(entry["time"])

        aggregated = {
            op: {
                "count": len(times),
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            }
            for op, times in by_op.items()
        }

        return {
            "current_memory_usage": psutil.virtual_memory().percent / 100,
            "current_cpu_usage": psutil.cpu_percent() / 100,
            "current_gpu_memory_usage": self._get_gpu_memory_usage(),
            "processing_times": self.performance_stats["processing_times"],
            "processing_times_by_operation": aggregated,
        }

    # ------------------------------------------------------------------
    # Memory optimisation
    # ------------------------------------------------------------------

    def optimize_memory_usage(self, target_usage: float = 0.7) -> None:
        """Снизить использование RAM до target_usage при необходимости."""
        try:
            current = psutil.virtual_memory().percent / 100
            if current > target_usage:
                self._cleanup_memory()
                gc.collect()
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                after = psutil.virtual_memory().percent / 100
                logger.info("Memory: %.2f → %.2f", current, after)
        except Exception as exc:
            logger.error("Memory optimization error: %s", exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Освобождение всех ресурсов."""
        try:
            self.stop_monitoring()
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            self.performance_stats.clear()
            logger.info("PerformanceOptimizer cleaned up")
        except Exception as exc:
            logger.error("PerformanceOptimizer cleanup error: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_memory_availability(self) -> bool:
        return psutil.virtual_memory().percent / 100 < self.max_memory_usage

    def _get_gpu_memory_usage(self) -> float:
        try:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.max_memory_allocated()
                return allocated / total if total > 0 else 0.0
        except Exception:
            pass
        return 0.0

    def _cleanup_memory(self) -> None:
        if hasattr(self, "response_cache"):
            self.response_cache.clear()  # type: ignore[attr-defined]
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_large_model(self, model_config: Dict[str, Any]) -> bool:
        return model_config.get("hidden_dim", 0) * model_config.get("num_layers", 1) > 10_000

    def _load_model_in_thread(self, model_class: type, model_config: Dict[str, Any]) -> Any:
        future = self.thread_pool.submit(model_class, model_config)
        return future.result()

    def _process_batch(
        self, batch: List[Any], process_func: Callable[[Any], Any]
    ) -> List[Any]:
        return [process_func(item) for item in batch]

    def _monitor_resources(self, interval: float) -> None:
        max_history = 1_000
        while self.monitoring_active:
            try:
                mem = psutil.virtual_memory().percent / 100
                cpu = psutil.cpu_percent() / 100
                gpu = self._get_gpu_memory_usage()

                self.performance_stats["memory_usage"].append(mem)
                self.performance_stats["cpu_usage"].append(cpu)
                self.performance_stats["gpu_memory_usage"].append(gpu)

                for key in ("memory_usage", "cpu_usage", "gpu_memory_usage"):
                    if len(self.performance_stats[key]) > max_history:
                        self.performance_stats[key] = self.performance_stats[key][-max_history:]

                if mem > self.max_memory_usage:
                    self.optimize_memory_usage()

                time.sleep(interval)
            except Exception as exc:
                logger.error("Resource monitoring error: %s", exc)
                time.sleep(interval)
