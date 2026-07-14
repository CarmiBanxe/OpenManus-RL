"""Тесты единого агент-фасада LegionAgent (S17)."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agent import AgentConfig, LegionAgent, create_agent
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.embeddings import EmbeddingProvider
from openmanus_rl.memory.semantic_memory import SemanticMemory


class FakeEmbeddings(EmbeddingProvider):
    VECS = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "weather": [0.0, 0.0, 1.0]}

    def embed(self, text):
        t = text.lower()
        for k, v in self.VECS.items():
            if k in t:
                return v
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


def _final(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _tool_call(name, args_json):
    return {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": name, "arguments": args_json}}]}}]}


class TestLegionAgentChat(unittest.TestCase):
    def test_chat_no_tools_persists_memory(self):
        agent = LegionAgent(AgentConfig(model="smart", tools=False, memory=True))
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [_final("Hi there")]
            result = agent.chat("Hello")
        self.assertEqual(result["content"], "Hi there")
        self.assertEqual(result["tools_used"], [])
        ctx = agent.memory.get_context()
        self.assertIn({"role": "user", "content": "Hello"}, ctx)
        self.assertIn({"role": "assistant", "content": "Hi there"}, ctx)

    def test_chat_with_tools(self):
        agent = LegionAgent(AgentConfig(model="smart", tools=True, memory=True))
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [_tool_call("calculator", '{"expression": "2+2"}'), _final("It is 4")]
            result = agent.chat("What is 2+2?")
        self.assertEqual(result["content"], "It is 4")
        self.assertEqual(len(result["tools_used"]), 1)
        self.assertEqual(result["tools_used"][0]["output"], "4")
        self.assertIn({"role": "assistant", "content": "It is 4"}, agent.memory.get_context())


class TestLegionAgentStream(unittest.IsolatedAsyncioTestCase):
    @patch("aiohttp.ClientSession.post")
    async def test_stream_persists(self, mock_post):
        agent = LegionAgent(AgentConfig(model="smart", memory=True))
        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "Hel"}}]}',
            b'data: {"choices": [{"delta": {"content": "lo"}}]}', b'data: [DONE]'])
        chunks = [c async for c in agent.stream("hi")]
        await agent.close()
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertIn({"role": "assistant", "content": "Hello"}, agent.memory.get_context())

    @patch("aiohttp.ClientSession.post")
    async def test_stream_rag_injects_relevant(self, mock_post):
        backend = SemanticMemory(FakeEmbeddings(), ":memory:")
        mem = ConversationMemory(backend, session_id="s1", max_turns=50)
        backend.add_turn("s1", "user", "I have a cat named Felix")
        backend.add_turn("s1", "user", "The weather is sunny")
        agent = LegionAgent(AgentConfig(model="smart", rag=True, rag_k=1), memory=mem)

        mock_post.return_value.__aenter__.return_value = _mock_stream([
            b'data: {"choices": [{"delta": {"content": "ok"}}]}', b'data: [DONE]'])
        captured = {}
        real = agent.stream_adapter.stream_chat

        def spy(messages, **kwargs):
            captured["messages"] = list(messages)
            return real(messages, **kwargs)

        with patch.object(agent.stream_adapter, "stream_chat", side_effect=spy):
            _ = [c async for c in agent.stream("tell me about my cat")]
        await agent.close()
        backend.close()

        contents = [m["content"] for m in captured["messages"]]
        self.assertIn("I have a cat named Felix", contents)
        self.assertNotIn("The weather is sunny", contents)
        self.assertEqual(contents[-1], "tell me about my cat")


class TestAgentConfig(unittest.TestCase):
    def test_roundtrip(self):
        cfg = AgentConfig(model="coding", rag=True, tools=True, max_turns=7)
        cfg2 = AgentConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg2.model, "coding")
        self.assertTrue(cfg2.rag and cfg2.tools)
        self.assertEqual(cfg2.max_turns, 7)

    def test_engine_config_master_key_from_env_when_none(self):
        cfg = AgentConfig(master_key=None)
        self.assertNotIn("master_key", cfg.engine_config())
        cfg2 = AgentConfig(master_key="k")
        self.assertEqual(cfg2.engine_config()["master_key"], "k")

    def test_create_agent_from_dict(self):
        agent = create_agent({"model": "smart", "tools": True})
        self.assertIsInstance(agent, LegionAgent)
        self.assertIsNotNone(agent.tool_adapter)


if __name__ == "__main__":
    unittest.main()
