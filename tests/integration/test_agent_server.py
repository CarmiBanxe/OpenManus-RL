"""Тесты REST-сервера LegionAgent (S18) + CLI."""
import argparse
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from fastapi.testclient import TestClient

    from openmanus_rl.api.agent_server import create_agent_app
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from openmanus_rl.agent.cli import build_agent, chat_once  # noqa: E402


def _final(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@unittest.skipIf(not _AVAILABLE, "FastAPI not available")
class TestAgentServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_agent_app({
            "model": "smart", "tools": False, "rag": False,
            "memory": True, "memory_db": ":memory:", "enable_observability": False}))

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())

    def test_chat(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as m:
            m.side_effect = [_final("Hi from server")]
            r = self.client.post("/chat", json={"message": "hello", "session_id": "s1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["content"], "Hi from server")

    def test_chat_persists_per_session(self):
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as m:
            m.side_effect = [_final("one"), _final("two")]
            self.client.post("/chat", json={"message": "first", "session_id": "s2"})
            self.client.post("/chat", json={"message": "second", "session_id": "s2"})
            r = self.client.post("/reset", json={"message": "", "session_id": "s2"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], "s2")

    def test_api_key_enforced(self):
        with patch.dict(os.environ, {"LEGION_API_KEY": "secret"}):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 401)

    def test_api_key_accepts_valid(self):
        with patch.dict(os.environ, {"LEGION_API_KEY": "secret"}), \
                patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as m:
            m.side_effect = [_final("ok")]
            r = self.client.post("/chat", json={"message": "hi", "session_id": "auth"},
                                 headers={"X-API-Key": "secret"})
        self.assertEqual(r.status_code, 200)


class TestCLI(unittest.TestCase):
    def test_chat_once(self):
        agent = build_agent(argparse.Namespace(model="smart", rag=False, tools=False, session="t"))
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as m:
            m.side_effect = [_final("cli answer")]
            res = chat_once(agent, "hi")
        self.assertEqual(res["content"], "cli answer")


if __name__ == "__main__":
    unittest.main()
