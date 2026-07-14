"""Интеграционные тесты streaming (адаптер + фабрика)."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter
from openmanus_rl.engines.enhanced_factory_with_streaming import create_engine, create_streaming_engine


def _content_iter(chunks):
    async def _gen(*args, **kwargs):  # Mock зовёт __aiter__(self) — принимаем лишний arg
        for c in chunks:
            yield c
    return _gen


class TestStreamingIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"base_url": "http://localhost:4000", "model": "gpt-3.5-turbo", "master_key": "test-key"}

    @patch("aiohttp.ClientSession.post")
    async def test_streaming_adapter_integration(self, mock_post):
        resp = AsyncMock(status=200)
        resp.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}', b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = resp

        adapter = StreamingLiteLLMAdapter(self.config)
        results = [c async for c in adapter.stream_generate("Hello")]
        self.assertEqual(results, ["Hello", " world"])
        await adapter.close()

    @patch("aiohttp.ClientSession.post")
    async def test_factory_streaming_integration(self, mock_post):
        resp = AsyncMock(status=200)
        resp.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            b'data: {"choices": [{"delta": {"content": " there"}}]}', b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = resp

        engine = create_streaming_engine(self.config)
        self.assertIsInstance(engine, StreamingLiteLLMAdapter)
        results = [c async for c in engine.stream_generate("Hi")]
        self.assertEqual(results, ["Hi", " there"])
        await engine.close()

    @patch("aiohttp.ClientSession.post")
    async def test_factory_auto_streaming(self, mock_post):
        resp = AsyncMock(status=200)
        resp.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"delta": {"content": "Auto"}}]}',
            b'data: {"choices": [{"delta": {"content": " streaming"}}]}', b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = resp

        engine = create_engine("litellm", self.config, stream=True)
        self.assertIsInstance(engine, StreamingLiteLLMAdapter)
        results = [c async for c in engine.stream_generate("Auto")]
        self.assertEqual(results, ["Auto", " streaming"])
        await engine.close()

    def test_backward_compatibility(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter.generate") as mock_generate:
            mock_generate.return_value = {"choices": [{"text": "Backward compatible"}]}
            engine = create_streaming_engine(self.config)
            self.assertEqual(engine.generate("Test"), {"choices": [{"text": "Backward compatible"}]})
            mock_generate.assert_called_once_with("Test")


if __name__ == "__main__":
    unittest.main()
