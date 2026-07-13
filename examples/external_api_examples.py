"""
Примеры использования интеграции с внешними API через LiteLLM.

Запуск: python examples/external_api_examples.py
Требует поднятый LiteLLM на :4000 (и LITELLM_MASTER_KEY в окружении для auth).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.engines.enhanced_factory import (  # noqa: E402
    EnhancedEngineFactory,
    create_engine,
    get_available_engines,
    get_engine_info,
)


def basic_example() -> None:
    print("Базовый пример:")
    print("=" * 40)
    engine = create_engine("litellm")
    if not engine.is_available():
        print("❌ LiteLLM недоступен")
        return
    print("✅ LiteLLM доступен")

    response = engine.generate("Hello, how are you?")
    print(f"generate -> {response.get('choices', [{}])[0].get('text', '')}")

    response = engine.chat([{"role": "user", "content": "Hello, how are you?"}])
    print(f"chat -> {response.get('choices', [{}])[0].get('message', {}).get('content', '')}")

    metrics = engine.get_metrics()
    print(f"метрики: запросов={metrics['total_requests']} успешных={metrics['successful_requests']} "
          f"avg={metrics['avg_response_time']:.2f}s tok/s={metrics['tokens_per_second']:.2f}")


def advanced_example() -> None:
    print("\nПродвинутый пример (кастомный конфиг):")
    print("=" * 40)
    factory = EnhancedEngineFactory()
    config = {
        "base_url": "http://localhost:4000",
        "model": "gpt-4",
        "timeout": 60,
        "max_retries": 3,
        "fallback_models": ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
        "master_key": os.environ.get("LITELLM_MASTER_KEY"),
    }
    engine = factory.create_engine("litellm", config)
    if not engine.is_available():
        print("❌ LiteLLM недоступен")
        return
    response = engine.generate("Write a short poem about programming", temperature=0.8, max_tokens=100)
    print(f"generate -> {response.get('choices', [{}])[0].get('text', '')}")


def engines_info_example() -> None:
    print("\nИнформация о движках:")
    print("=" * 40)
    for name in get_available_engines():
        info = get_engine_info(name)
        print(f"  {name}: class={info['class']}")


def models_example() -> None:
    print("\nМодели LiteLLM:")
    print("=" * 40)
    engine = create_engine("litellm")
    if not engine.is_available():
        print("❌ LiteLLM недоступен")
        return
    models = engine.list_models()
    print(f"Доступно моделей: {len(models)}")
    for model in models[:5]:
        print(f"  - {model.get('id')}")


if __name__ == "__main__":
    try:
        basic_example()
        advanced_example()
        engines_info_example()
        models_example()
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}")
