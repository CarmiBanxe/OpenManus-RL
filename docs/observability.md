# Observability для OpenManus

Слой observability: Prometheus-метрики, структурированное логирование (structlog),
health-check эндпоинт (FastAPI). Интеграция в движки **опциональна** и по умолчанию
выключена — S10 (`enhanced_factory.py`) не затронут.

## Компоненты

- `MetricsCollector` — Prometheus-метрики в **изолированном** `CollectorRegistry`
  (нет `Duplicated timeseries`); HTTP-сервер не стартует автоматически, при явном
  `start_server()` биндится на **127.0.0.1**.
- `OpenManusLogger` — structlog/JSON: `request/response/error/fallback/system_event`.
- `HealthChecker` — регистрация компонентов, `check_component/check_all`.

### Метрики

`openmanus_requests_total`, `openmanus_request_duration_seconds`,
`openmanus_active_requests`, `openmanus_tokens_total`, `openmanus_tokens_per_second`,
`openmanus_errors_total`, `openmanus_fallback_total`,
`openmanus_system_cpu_usage_percent`, `openmanus_system_memory_usage_percent`,
`openmanus_process_memory_usage_bytes`.

## Использование

```python
from openmanus_rl.observability import get_metrics_collector, get_logger, get_health_checker

metrics = get_metrics_collector()          # сервер НЕ стартует автоматически
metrics.record_request_start("litellm", "gpt-3.5-turbo")
metrics.record_request_end("litellm", "gpt-3.5-turbo", "success", "generate", 1.5, 20)

logger = get_logger("openmanus", "INFO")
logger.request(engine_type="litellm", model="gpt-3.5-turbo",
               operation="generate", request_id="r-1")

health = get_health_checker()
health.register_component("litellm", lambda: {"status": "healthy", "message": "OK"})
print(health.check_all()["status"])
```

Метрики в движке (опционально, дефолт-выкл):

```python
from openmanus_rl.engines.enhanced_factory_with_observability import create_engine

engine = create_engine("litellm", {"enable_observability": True})
engine.generate("Hello")
```

## Health-check эндпоинт (FastAPI, 127.0.0.1)

`GET /health` · `GET /health/{component}` · `GET /metrics` · `GET /engines` · `GET /observability`

```bash
python -m openmanus_rl.api.health   # uvicorn на 127.0.0.1:8080
```

## Конфигурация

`config/observability.toml` — порты/уровни/интервалы; `[engines].enabled=false`.

## Валидация

```bash
python scripts/validate_sprint.py --sprint observability
python -m pytest tests/unit/test_observability.py tests/integration/test_observability_integration.py tests/integration/test_health_endpoint.py -v
```
