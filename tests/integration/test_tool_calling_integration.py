"""Интеграция агентик tool-calling лупа (mock LiteLLM _make_request)."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.tool_calling_adapter import ToolCallingAdapter
from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.sqlite_memory import SQLiteMemory


def _tool_call_response(name, arguments_json):
    return {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": name, "arguments": arguments_json}}]}}]}


def _final_response(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestToolCallingLoop(unittest.TestCase):
    def setUp(self):
        self.adapter = ToolCallingAdapter({"model": "smart", "master_key": "x"})

    def test_loop_executes_tool_then_answers(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [
                _tool_call_response("calculator", '{"expression": "6*7"}'),
                _final_response("The answer is 42"),
            ]
            result = self.adapter.run([{"role": "user", "content": "What is 6*7?"}])

        self.assertEqual(result["content"], "The answer is 42")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(len(result["tools_used"]), 1)
        self.assertEqual(result["tools_used"][0]["name"], "calculator")
        self.assertEqual(result["tools_used"][0]["output"], "42")

    def test_no_tool_call_returns_directly(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [_final_response("Just a plain answer")]
            result = self.adapter.run([{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "Just a plain answer")
        self.assertEqual(result["tools_used"], [])
        self.assertEqual(result["iterations"], 1)

    def test_unknown_tool_call_handled(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [
                _tool_call_response("does_not_exist", "{}"),
                _final_response("recovered"),
            ]
            result = self.adapter.run([{"role": "user", "content": "x"}])
        self.assertEqual(result["content"], "recovered")
        self.assertIn("unknown tool", result["tools_used"][0]["output"])

    def test_memory_persists_exchange(self):
        backend = SQLiteMemory(":memory:")
        mem = ConversationMemory(backend, session_id="s1", max_turns=50)
        adapter = ToolCallingAdapter({"model": "smart", "master_key": "x"}, memory=mem)
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as mreq:
            mreq.side_effect = [
                _tool_call_response("calculator", '{"expression": "2+2"}'),
                _final_response("It is 4"),
            ]
            adapter.run([{"role": "user", "content": "What is 2+2?"}])
        ctx = mem.get_context()
        self.assertIn({"role": "user", "content": "What is 2+2?"}, ctx)
        self.assertIn({"role": "assistant", "content": "It is 4"}, ctx)
        backend.close()


if __name__ == "__main__":
    unittest.main()
