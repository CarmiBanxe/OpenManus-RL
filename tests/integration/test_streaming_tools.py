"""Тесты streaming+tools в одном потоке (S20): resolve non-stream -> стрим финала."""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agent import AgentConfig, LegionAgent
from openmanus_rl.engines.streaming_tool_calling_adapter import StreamingToolCallingAdapter
from openmanus_rl.engines.tool_calling_adapter import ToolCallingAdapter


def _content_iter(chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c
    return _gen


def _mock_stream(chunks):
    resp = AsyncMock(status=200)
    resp.content.__aiter__ = _content_iter(chunks)
    return resp


def _tool_call(name, args_json):
    return {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": name, "arguments": args_json}}]}}]}


def _final(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


_MREQ = "openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request"
_POST = "aiohttp.ClientSession.post"


class TestResolve(unittest.TestCase):
    def test_resolve_returns_messages_and_tools_no_final(self):
        adapter = ToolCallingAdapter({"model": "smart"})
        with patch(_MREQ) as m:
            m.side_effect = [_tool_call("calculator", '{"expression": "6*7"}'), _final("IGNORED")]
            msgs, tools_used = adapter.resolve([{"role": "user", "content": "6*7?"}])
        self.assertEqual(len(tools_used), 1)
        self.assertEqual(tools_used[0]["output"], "42")
        # финал НЕ добавлен: последнее сообщение — tool-результат
        self.assertEqual(msgs[-1]["role"], "tool")
        self.assertEqual(msgs[-1]["content"], "42")
        self.assertFalse(any(m.get("content") == "IGNORED" for m in msgs))

    def test_resolve_no_tools(self):
        adapter = ToolCallingAdapter({"model": "smart"})
        with patch(_MREQ) as m:
            m.side_effect = [_final("just answer")]
            msgs, tools_used = adapter.resolve([{"role": "user", "content": "hi"}])
        self.assertEqual(tools_used, [])
        self.assertEqual([mm["content"] for mm in msgs], ["hi"])


class TestStreamingToolCallingAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_then_stream(self):
        adapter = StreamingToolCallingAdapter({"model": "smart"})
        with patch(_MREQ) as m, patch(_POST) as post:
            m.side_effect = [_tool_call("calculator", '{"expression": "6*7"}'), _final("x")]
            post.return_value.__aenter__.return_value = _mock_stream([
                b'data: {"choices": [{"delta": {"content": "The answer is "}}]}',
                b'data: {"choices": [{"delta": {"content": "42"}}]}', b'data: [DONE]'])
            chunks = [c async for c in adapter.stream_chat([{"role": "user", "content": "6*7?"}])]
        await adapter.close()
        self.assertEqual(chunks, ["The answer is ", "42"])
        self.assertEqual(adapter.last_tools_used[0]["name"], "calculator")


class TestLegionAgentStreamWithTools(unittest.IsolatedAsyncioTestCase):
    async def test_stream_uses_tools_then_streams(self):
        agent = LegionAgent(AgentConfig(model="smart", tools=True, memory=True))
        with patch(_MREQ) as m, patch(_POST) as post:
            m.side_effect = [_tool_call("calculator", '{"expression": "2+2"}'), _final("y")]
            post.return_value.__aenter__.return_value = _mock_stream([
                b'data: {"choices": [{"delta": {"content": "It is 4"}}]}', b'data: [DONE]'])
            chunks = [c async for c in agent.stream("What is 2+2?")]
        await agent.close()
        self.assertEqual(chunks, ["It is 4"])
        self.assertEqual(agent.last_tools_used[0]["output"], "4")
        ctx = agent.memory.get_context()
        self.assertIn({"role": "user", "content": "What is 2+2?"}, ctx)
        self.assertIn({"role": "assistant", "content": "It is 4"}, ctx)


if __name__ == "__main__":
    unittest.main()
