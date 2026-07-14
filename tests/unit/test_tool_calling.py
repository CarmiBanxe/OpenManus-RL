"""Тесты tool-calling слоя (registry/executor/builtins/bridge)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.tool_calling.builtins import register_builtins, safe_eval
from openmanus_rl.tool_calling.executor import ToolExecutor
from openmanus_rl.tool_calling.octotools_bridge import wrap_octotool
from openmanus_rl.tool_calling.registry import Tool, ToolRegistry


class TestSafeEval(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_eval("6*7"), 42)
        self.assertEqual(safe_eval("2 ** 10"), 1024)
        self.assertEqual(safe_eval("(1+2)*3 - 4/2"), 7.0)

    def test_rejects_unsafe(self):
        with self.assertRaises(ValueError):
            safe_eval("__import__('os').system('ls')")
        with self.assertRaises(ValueError):
            safe_eval("len([1,2,3])")


class TestRegistry(unittest.TestCase):
    def test_register_and_schema(self):
        reg = ToolRegistry()
        reg.register("t", "desc", {"type": "object", "properties": {}}, lambda: "x")
        self.assertTrue(reg.has("t"))
        self.assertEqual(reg.names(), ["t"])
        sc = reg.schemas()[0]
        self.assertEqual(sc["type"], "function")
        self.assertEqual(sc["function"]["name"], "t")

    def test_get_unknown_raises(self):
        with self.assertRaises(KeyError):
            ToolRegistry().get("nope")

    def test_builtins_registered(self):
        reg = register_builtins(ToolRegistry())
        for name in ("calculator", "current_time", "echo"):
            self.assertIn(name, reg.names())


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.reg = register_builtins(ToolRegistry())
        self.ex = ToolExecutor(self.reg, timeout=5.0)

    def tearDown(self):
        self.ex.shutdown()

    def test_execute_calculator(self):
        self.assertEqual(self.ex.execute("calculator", {"expression": "47*89"}), "4183")

    def test_execute_echo(self):
        self.assertEqual(self.ex.execute("echo", {"text": "hi"}), "hi")

    def test_unknown_tool(self):
        self.assertIn("unknown tool", self.ex.execute("nope", {}))

    def test_tool_error_captured(self):
        # неверное выражение -> инструмент бросает, executor вернёт строку ошибки
        out = self.ex.execute("calculator", {"expression": "1/0"})
        self.assertTrue(out.startswith("Error"))

    def test_bad_kwargs_captured(self):
        out = self.ex.execute("calculator", {"wrong": "x"})
        self.assertTrue(out.startswith("Error"))


class TestOctotoolsBridge(unittest.TestCase):
    def test_wrap(self):
        class FakeOcto:
            def get_metadata(self):
                return {"tool_name": "wiki", "tool_description": "search wikipedia",
                        "input_types": {"query": "the search query"}}

            def execute(self, query):
                return f"result for {query}"

        tool = wrap_octotool(FakeOcto())
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "wiki")
        self.assertIn("query", tool.parameters["properties"])
        self.assertEqual(tool.func(query="pandas"), "result for pandas")


if __name__ == "__main__":
    unittest.main()
