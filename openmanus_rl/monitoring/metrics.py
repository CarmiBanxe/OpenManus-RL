"""
Prometheus-метрики OpenManus — РЕАЛЬНЫЕ данные (psutil), не моки.

Экспортируются через защищённый /metrics (auth-required, localhost) в server.py.
Метрики:
  openmanus_requests_total{method,path,status}      — счётчик запросов
  openmanus_request_latency_seconds{method,path}    — гистограмма латентности
  openmanus_memory_usage_bytes                      — RSS процесса
  openmanus_cpu_usage_percent                       — CPU %
"""
from typing import Tuple

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "openmanus_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
LATENCY = Histogram(
    "openmanus_request_latency_seconds", "HTTP request latency (s)", ["method", "path"]
)
MEMORY = Gauge("openmanus_memory_usage_bytes", "Process RSS memory (bytes)")
CPU = Gauge("openmanus_cpu_usage_percent", "Process/host CPU usage (percent)")


def record_request(method: str, path: str, status: int, latency_s: float) -> None:
    REQUESTS.labels(method, path, str(status)).inc()
    LATENCY.labels(method, path).observe(latency_s)


def update_system_metrics() -> None:
    """Обновить системные gauge реальными значениями (psutil)."""
    try:
        import psutil
        MEMORY.set(psutil.Process().memory_info().rss)
        CPU.set(psutil.cpu_percent())
    except Exception:  # noqa: BLE001
        pass


def render() -> Tuple[bytes, str]:
    """Вернуть (тело, content-type) в формате Prometheus text exposition."""
    update_system_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST
