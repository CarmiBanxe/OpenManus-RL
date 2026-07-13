# Интеграция с внешними API через LiteLLM

Спринт 10 добавляет тонкий адаптер поверх работающего LiteLLM (:4000) для
унифицированного доступа к внешним провайдерам (OpenAI, Together, Ollama) с
метриками производительности и graceful-availability. Адаптер **не дублирует**
routing/balancing/fallback самого LiteLLM.

## Поддерживаемые провайдеры

- OpenAI, Together, Ollama и любые другие, настроенные в LiteLLM.

## Использование

```python
from openmanus_rl.engines.enhanced_factory import create_engine

engine = create_engine("litellm")
if engine.is_available():
    print(engine.generate("Hello, how are you?"))
    print(engine.chat([{"role": "user", "content": "Hi"}]))
```

Кастомный конфиг:

```python
from openmanus_rl.engines.enhanced_factory import EnhancedEngineFactory

factory = EnhancedEngineFactory()
engine = factory.create_engine("litellm", {
    "base_url": "http://localhost:4000",
    "model": "gpt-4",
    "max_retries": 3,
    "fallback_models": ["gpt-3.5-turbo"],
    "master_key": os.environ["LITELLM_MASTER_KEY"],
})
```

## Конфигурация

- `config/engines.toml` — конфиг движков (без секретов).
- `config/litellm_config.yaml` — **образец** конфига LiteLLM; `master_key`
  задаётся плейсхолдером `${LITELLM_MASTER_KEY}` (подставляется из окружения).
  Живой конфиг работающего LiteLLM — `~/MetaClaw/litellm/litellm-config.v2.yaml`.

## Переменные окружения

- `LITELLM_MASTER_KEY` — ключ доступа к LiteLLM (auth).
- `RUN_LITELLM_TESTS=1` — включает реальные тесты LiteLLM.

## Безопасность

- Секреты только из окружения; в репозитории — плейсхолдер.
- Проверка доступности: `/health` с `Authorization: Bearer`; **401 = сервис
  поднят, но требует ключ** (трактуется как «доступен»), не как «выключен».

## Метрики (`get_metrics()`)

`total_requests`, `successful_requests`, `failed_requests`, `total_time`,
`avg_response_time`, `tokens_per_second`, `fallback_used`.

## Тестирование

```bash
# Юнит/интеграция (моки HTTP):
python -m pytest tests/integration/test_external_api_integration.py -v

# Реальный LiteLLM:
export LITELLM_MASTER_KEY="..."; export RUN_LITELLM_TESTS=1
python -m pytest tests/integration/test_external_api_integration.py::TestRealLiteLLMIntegration -v

# Финальный гейт Спринта 10:
python -m pytest tests/integration/test_sprint10_gate.py -v

# Через валидатор спринтов:
python scripts/validate_sprint.py --sprint external-api
```
