"""Примеры использования observability модуля."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.observability import (  # noqa: E402
    get_health_checker,
    get_logger,
    get_metrics_collector,
)


def example_metrics() -> None:
    print("Пример метрик")
    print("=" * 30)
    metrics = get_metrics_collector()
    for i in range(5):
        metrics.record_request_start("litellm", "gpt-3.5-turbo")
        time.sleep(0.05)
        metrics.record_request_end("litellm", "gpt-3.5-turbo", "success", "generate", 0.05, 20 + i)
    summary = metrics.get_metrics_summary()
    print(f"Всего запросов: {sum(summary.get('openmanus_requests_total', {}).values())}")
    metrics.update_system_metrics()
    cpu = summary.get("openmanus_system_cpu_usage_percent", {}).get((), 0)
    print(f"CPU (snapshot): {cpu}%\n")


def example_logging() -> None:
    print("Пример логирования")
    print("=" * 30)
    logger = get_logger("example", "INFO")
    for i in range(3):
        rid = f"example-{i}"
        logger.request(engine_type="litellm", model="gpt-3.5-turbo",
                       operation="generate", request_id=rid, prompt=f"Hello {i}")
        logger.response(engine_type="litellm", model="gpt-3.5-turbo", operation="generate",
                        request_id=rid, status="success", duration=0.1, tokens=20)
    logger.system_event("example_completed", requests_processed=3)
    print("Логи выведены выше (JSON).\n")


def example_health_check() -> None:
    print("Пример health-check")
    print("=" * 30)
    health = get_health_checker()
    health.register_component("litellm", lambda: {"status": "healthy", "message": "LiteLLM OK"}, 0.0)
    health.register_component("ollama", lambda: {"status": "degraded", "message": "slow"}, 0.0)
    status = health.check_all()
    print(f"Общий статус: {status['status']} — {status['message']}")
    for name, comp in status["components"].items():
        print(f"  {name}: {comp['status']} — {comp['message']}")
    print()


def main() -> None:
    print("Примеры observability")
    print("=" * 50)
    example_metrics()
    example_logging()
    example_health_check()
    print("Готово")


if __name__ == "__main__":
    main()
