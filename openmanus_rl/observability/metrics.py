"""
Модуль сбора Prometheus-метрик для OpenManus.

Каждый коллектор держит СВОЙ CollectorRegistry (изоляция — нет
Duplicated timeseries при нескольких экземплярах). HTTP-сервер метрик НЕ
стартует автоматически; при явном start_server биндится на 127.0.0.1 (S-18).
"""
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server


class MetricsCollector:
    """Коллектор Prometheus-метрик для движков OpenManus."""

    def __init__(self, port: int = 9090) -> None:
        self.port = port
        self._lock = threading.Lock()
        self._registry = CollectorRegistry()  # свой реестр — изоляция

        self.request_count = Counter(
            "openmanus_requests_total", "Total number of requests",
            ["engine_type", "model", "status"], registry=self._registry)
        self.request_duration = Histogram(
            "openmanus_request_duration_seconds", "Request duration in seconds",
            ["engine_type", "model", "operation"], registry=self._registry)
        self.active_requests = Gauge(
            "openmanus_active_requests", "Number of active requests",
            ["engine_type"], registry=self._registry)
        self.tokens_total = Counter(
            "openmanus_tokens_total", "Total number of tokens processed",
            ["engine_type", "model", "direction"], registry=self._registry)
        self.tokens_per_second = Gauge(
            "openmanus_tokens_per_second", "Tokens processed per second",
            ["engine_type", "model"], registry=self._registry)

        self.system_cpu_usage = Gauge(
            "openmanus_system_cpu_usage_percent", "System CPU usage percentage",
            registry=self._registry)
        self.system_memory_usage = Gauge(
            "openmanus_system_memory_usage_percent", "System memory usage percentage",
            registry=self._registry)
        self.process_memory_usage = Gauge(
            "openmanus_process_memory_usage_bytes", "Process memory usage in bytes",
            registry=self._registry)

        self.error_count = Counter(
            "openmanus_errors_total", "Total number of errors",
            ["engine_type", "error_type"], registry=self._registry)
        self.fallback_count = Counter(
            "openmanus_fallback_total", "Total number of fallbacks",
            ["engine_type", "from_model", "to_model"], registry=self._registry)

        self._token_windows: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=10))
        self._time_windows: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=10))
        self._custom_gauges: Dict[str, Gauge] = {}  # динамические gauge'ы (напр. streaming TTFT)
        self._server_started = False

    def start_server(self) -> None:
        """Явный запуск HTTP-сервера метрик (127.0.0.1, свой реестр)."""
        if self._server_started:
            return
        with self._lock:
            if not self._server_started:
                start_http_server(self.port, addr="127.0.0.1", registry=self._registry)
                self._server_started = True
                print(f"Prometheus metrics server started on 127.0.0.1:{self.port}")

    def record_request_start(self, engine_type: str, model: str) -> None:
        self.active_requests.labels(engine_type=engine_type).inc()

    def record_request_end(self, engine_type: str, model: str, status: str,
                           operation: str, duration: float, tokens: int = 0) -> None:
        self.active_requests.labels(engine_type=engine_type).dec()
        self.request_count.labels(engine_type=engine_type, model=model, status=status).inc()
        self.request_duration.labels(engine_type=engine_type, model=model, operation=operation).observe(duration)
        if tokens > 0:
            self.tokens_total.labels(engine_type=engine_type, model=model, direction="output").inc(tokens)
            self._update_tokens_per_second(engine_type, model, tokens, duration)

    def record_error(self, engine_type: str, error_type: str) -> None:
        self.error_count.labels(engine_type=engine_type, error_type=error_type).inc()

    def record_fallback(self, engine_type: str, from_model: str, to_model: str) -> None:
        self.fallback_count.labels(engine_type=engine_type, from_model=from_model, to_model=to_model).inc()

    def record_tokens(self, engine_type: str, model: str, tokens: int, direction: str = "output") -> None:
        """Инкремент счётчика токенов (используется streaming-адаптером)."""
        if tokens > 0:
            self.tokens_total.labels(engine_type=engine_type, model=model, direction=direction).inc(tokens)

    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Установить значение произвольного Gauge (напр. streaming TTFT / tokens_per_second).

        Gauge создаётся лениво в собственном реестре; при повторном имени переиспользуется.
        """
        labels = labels or {}
        with self._lock:
            gauge = self._custom_gauges.get(name)
            if gauge is None:
                gauge = Gauge(name, name, sorted(labels.keys()), registry=self._registry)
                self._custom_gauges[name] = gauge
        (gauge.labels(**labels) if labels else gauge).set(value)

    def update_system_metrics(self) -> None:
        self.system_cpu_usage.set(psutil.cpu_percent())
        self.system_memory_usage.set(psutil.virtual_memory().percent)
        self.process_memory_usage.set(psutil.Process().memory_info().rss)

    def _update_tokens_per_second(self, engine_type: str, model: str, tokens: int, duration: float) -> None:
        with self._lock:
            now = time.time()
            key = (engine_type, model)
            self._token_windows[key].append(tokens)
            self._time_windows[key].append(now)
            if len(self._token_windows[key]) > 1:
                total_tokens = sum(self._token_windows[key])
                span = now - self._time_windows[key][0]
                if span > 0:
                    self.tokens_per_second.labels(engine_type=engine_type, model=model).set(total_tokens / span)

    def get_metrics_summary(self) -> Dict[str, Any]:
        # Ключуем по sample.name (не family.name): prometheus_client отрезает
        # суффикс `_total` у family.name Counter'ов, а sample.name сохраняет его.
        with self._lock:
            summary: Dict[str, Any] = {}
            for family in self._registry.collect():
                for sample in family.samples:
                    summary.setdefault(sample.name, {})[
                        tuple(sorted(sample.labels.items()))] = sample.value
            return summary


_metrics_collector = None


def get_metrics_collector(port: int = 9090) -> MetricsCollector:
    """Глобальный singleton коллектора (сервер НЕ стартуется автоматически)."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(port)
    return _metrics_collector
