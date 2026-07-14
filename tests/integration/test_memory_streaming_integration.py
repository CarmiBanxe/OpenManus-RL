"""Интеграция диалоговой памяти со стримингом (S12+S13)."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.memory_aware_streaming import MemoryAwareStreamingAdapter
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.sqlite_memory import SQLiteMemory


def _content_iter(chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c
    return _gen


def _mock_stream(chunks):
    resp = AsyncMock(status=200)
    resp.content.__aiter__ = _content_iter(chunks)
    return resp


class TestMemoryStreamingIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = SQLiteMemory(":memory:")
        self.memory = ConversationMemory(self.backend, session_id="s1", max_turns=20)
        self.adapter = MemoryAwareStreamingAdapter({"master_key": "test-key"}, memory=self.memory)

    def tearDown(self):
        self.backend.close()

    @patch("aiohttp.ClientSession.post")
    async def test_stream_persists_turns(self, mock_post):
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}', b'data: [DONE]'])

        results = [c async for c in self.adapter.stream_chat([{"role": "user", "content": "Hi"}])]
        self.assertEqual(results, ["Hello", " world"])
        await self.adapter.close()

        ctx = self.memory.get_context()
        self.assertIn({"role": "user", "content": "Hi"}, ctx)
        self.assertIn({"role": "assistant", "content": "Hello world"}, ctx)

    @patch("aiohttp.ClientSession.post")
    async def test_prior_context_injected_on_second_call(self, mock_post):
        captured = {}

        real_stream_chat = self.adapter.stream.stream_chat

        def spy_stream_chat(messages, **kwargs):
            captured["messages"] = list(messages)
            return real_stream_chat(messages, **kwargs)

        # первый обмен
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "A"}}]}', b'data: [DONE]'])
        _ = [c async for c in self.adapter.stream_chat([{"role": "user", "content": "first"}])]

        # второй обмен — перехватываем, что реально ушло в базовый стрим
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "B"}}]}', b'data: [DONE]'])
        with patch.object(self.adapter.stream, "stream_chat", side_effect=spy_stream_chat):
            _ = [c async for c in self.adapter.stream_chat([{"role": "user", "content": "second"}])]
        await self.adapter.close()

        # в базовый стрим ушли: прежний контекст (first + A) + текущее сообщение (second)
        contents = [m["content"] for m in captured["messages"]]
        self.assertEqual(contents, ["first", "A", "second"])


if __name__ == "__main__":
    unittest.main()
