"""Примеры использования streaming-адаптера (требует LiteLLM :4000 + LITELLM_MASTER_KEY)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.engines.streaming_adapter import create_streaming_adapter
from openmanus_rl.engines.enhanced_factory_with_streaming import create_engine


async def example_stream_generate() -> None:
    print("Streaming generate")
    print("=" * 30)
    adapter = create_streaming_adapter({"model": "gpt-3.5-turbo", "max_tokens": 80})
    if not adapter.is_available():
        print("❌ LiteLLM недоступен")
        await adapter.close()
        return
    try:
        print("Ответ: ", end="", flush=True)
        async for chunk in adapter.stream_generate("Расскажи кратко про Python"):
            print(chunk, end="", flush=True)
        print()
    finally:
        await adapter.close()


async def example_auto_streaming() -> None:
    print("\nАвто-переключение (stream=True)")
    print("=" * 30)
    engine = create_engine("litellm", {"model": "gpt-3.5-turbo", "max_tokens": 80}, stream=True)
    if not engine.is_available():
        print("❌ LiteLLM недоступен")
        await engine.close()
        return
    try:
        print("Ответ: ", end="", flush=True)
        async for chunk in engine.stream_generate("Что такое async/await?"):
            print(chunk, end="", flush=True)
        print()
    finally:
        await engine.close()


async def main() -> None:
    print("Примеры streaming")
    print("=" * 50)
    await example_stream_generate()
    await example_auto_streaming()
    print("\nГотово")


if __name__ == "__main__":
    asyncio.run(main())
