"""Интеграция RAG-памяти со стримингом (S12+S13+S14)."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.memory_aware_streaming import MemoryAwareStreamingAdapter
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.embeddings import EmbeddingProvider
from openmanus_rl.memory.semantic_memory import SemanticMemory


class FakeEmbeddings(EmbeddingProvider):
    VECS = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "weather": [0.0, 0.0, 1.0]}

    def embed(self, text):
        t = text.lower()
        for key, vec in self.VECS.items():
            if key in t:
                return vec
        return [0.0, 0.0, 0.0]


def _content_iter(chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c
    return _gen


def _mock_stream(chunks):
    resp = AsyncMock(status=200)
    resp.content.__aiter__ = _content_iter(chunks)
    return resp


class TestRAGStreamingIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = SemanticMemory(FakeEmbeddings(), ":memory:")
        self.memory = ConversationMemory(self.backend, session_id="s1", max_turns=50)
        # seed: два несвязанных факта в памяти
        self.backend.add_turn("s1", "user", "I have a cat named Felix")
        self.backend.add_turn("s1", "user", "The weather is sunny today")
        self.adapter = MemoryAwareStreamingAdapter(
            {"master_key": "x", "rag": True, "rag_k": 1}, memory=self.memory)

    def tearDown(self):
        self.backend.close()

    @patch("aiohttp.ClientSession.post")
    async def test_rag_injects_relevant_not_recent(self, mock_post):
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "ok"}}]}', b'data: [DONE]'])

        captured = {}
        real = self.adapter.stream.stream_chat

        def spy(messages, **kwargs):
            captured["messages"] = list(messages)
            return real(messages, **kwargs)

        with patch.object(self.adapter.stream, "stream_chat", side_effect=spy):
            _ = [c async for c in self.adapter.stream_chat(
                [{"role": "user", "content": "tell me about my cat"}])]
        await self.adapter.close()

        contents = [m["content"] for m in captured["messages"]]
        # RAG подмешал релевантный (cat), а не последний по времени (weather)
        self.assertIn("I have a cat named Felix", contents)
        self.assertNotIn("The weather is sunny today", contents)
        self.assertEqual(contents[-1], "tell me about my cat")

    @patch("aiohttp.ClientSession.post")
    async def test_turns_persisted_with_embeddings(self, mock_post):
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "hi"}}]}', b'data: [DONE]'])
        _ = [c async for c in self.adapter.stream_chat(
            [{"role": "user", "content": "a dog question"}])]
        await self.adapter.close()
        # семантический поиск по новому turn'у находит его
        hits = self.backend.semantic_search("s1", "dog", limit=1)
        self.assertIn("dog", hits[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
