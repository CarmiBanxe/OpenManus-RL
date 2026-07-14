"""Тесты streaming-адаптера."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.streaming_adapter import StreamingLiteLLMAdapter, create_streaming_adapter


def _content_iter(chunks):
    async def _gen(*args, **kwargs):  # Mock зовёт __aiter__(self) — принимаем лишний arg
        for c in chunks:
            yield c
    return _gen


class TestStreamingLiteLLMAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = StreamingLiteLLMAdapter({
            "base_url": "http://localhost:4000", "model": "gpt-3.5-turbo", "master_key": "test-key"})

    def test_init(self):
        self.assertEqual(self.adapter.base_url, "http://localhost:4000")
        self.assertEqual(self.adapter.model, "gpt-3.5-turbo")
        self.assertEqual(self.adapter.master_key, "test-key")
        self.assertIsNone(self.adapter.session)

    def test_availability_logic(self):
        # _check_litellm_availability пере-проверяет; is_available отдаёт кэш.
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            self.assertTrue(self.adapter._check_litellm_availability())
            mock_get.return_value.status_code = 401
            self.assertTrue(self.adapter._check_litellm_availability())
            mock_get.side_effect = Exception()
            self.assertFalse(self.adapter._check_litellm_availability())
            self.assertFalse(self.adapter.is_available())

    def test_health_check(self):
        with patch.object(self.adapter, "is_available", return_value=True):
            self.assertEqual(self.adapter._health_check()["status"], "healthy")
        with patch.object(self.adapter, "is_available", return_value=False):
            self.assertEqual(self.adapter._health_check()["status"], "unhealthy")

    @patch("aiohttp.ClientSession.post")
    async def test_stream_generate(self, mock_post):
        mock_response = AsyncMock(status=200)
        mock_response.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = mock_response

        results = [c async for c in self.adapter.stream_generate("Hello")]
        self.assertEqual(results, ["Hello", " world"])

    @patch("aiohttp.ClientSession.post")
    async def test_stream_chat(self, mock_post):
        mock_response = AsyncMock(status=200)
        mock_response.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            b'data: {"choices": [{"delta": {"content": " there"}}]}',
            b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = mock_response

        results = [c async for c in self.adapter.stream_chat([{"role": "user", "content": "Hello"}])]
        self.assertEqual(results, ["Hi", " there"])

    @patch("aiohttp.ClientSession.post")
    async def test_stream_handles_text_field(self, mock_post):
        # completions-формат: choices[].text вместо delta.content
        mock_response = AsyncMock(status=200)
        mock_response.content.__aiter__ = _content_iter([
            b'data: {"choices": [{"text": "abc"}]}', b'data: [DONE]'])
        mock_post.return_value.__aenter__.return_value = mock_response
        results = [c async for c in self.adapter.stream_generate("x")]
        self.assertEqual(results, ["abc"])

    def test_generate_backward_compatibility(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter.generate") as mock_generate:
            mock_generate.return_value = {"choices": [{"text": "Hello"}]}
            self.assertEqual(self.adapter.generate("Hello"), {"choices": [{"text": "Hello"}]})
            mock_generate.assert_called_once_with("Hello")

    def test_chat_backward_compatibility(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter.chat") as mock_chat:
            mock_chat.return_value = {"choices": [{"message": {"content": "Hi"}}]}
            messages = [{"role": "user", "content": "Hello"}]
            self.assertEqual(self.adapter.chat(messages), {"choices": [{"message": {"content": "Hi"}}]})
            mock_chat.assert_called_once_with(messages)

    async def asyncTearDown(self):
        if self.adapter.session and not self.adapter.session.closed:
            await self.adapter.close()


class TestCreateStreamingAdapter(unittest.TestCase):
    def test_create_streaming_adapter(self):
        adapter = create_streaming_adapter({"base_url": "http://localhost:4000", "model": "gpt-3.5-turbo"})
        self.assertIsInstance(adapter, StreamingLiteLLMAdapter)
        self.assertEqual(adapter.base_url, "http://localhost:4000")


if __name__ == "__main__":
    unittest.main()
