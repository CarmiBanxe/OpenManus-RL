"""Тесты SessionManager (TTL/лимит) + персистентности памяти через «рестарт» (S21)."""
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agent.session_manager import SessionManager


class _FakeAgent:
    def __init__(self, sid):
        self.sid = sid
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class TestSessionManager(unittest.TestCase):
    def test_get_reuses_same_agent(self):
        sm = SessionManager(_FakeAgent)
        self.assertIs(sm.get("a"), sm.get("a"))
        self.assertIsNot(sm.get("a"), sm.get("b"))
        self.assertEqual(sm.count(), 2)

    def test_ttl_eviction(self):
        sm = SessionManager(_FakeAgent, ttl_s=0.05)
        sm.get("x")
        self.assertEqual(sm.count(), 1)
        time.sleep(0.12)
        self.assertEqual(sm.count(), 0)

    def test_max_sessions_evicts_oldest(self):
        sm = SessionManager(_FakeAgent, max_sessions=2)
        sm.get("a")
        sm.get("b")
        sm.get("c")  # -> evicts oldest (a)
        self.assertEqual(sm.count(), 2)
        self.assertNotIn("a", sm.sessions())
        self.assertIn("c", sm.sessions())

    def test_reset(self):
        sm = SessionManager(_FakeAgent)
        agent = sm.get("z")
        self.assertTrue(sm.reset("z"))
        self.assertTrue(agent.reset_called)
        self.assertFalse(sm.reset("nope"))

    def test_drop(self):
        sm = SessionManager(_FakeAgent)
        sm.get("a")
        sm.drop("a")
        self.assertEqual(sm.count(), 0)


def _final(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


try:
    from fastapi.testclient import TestClient

    from openmanus_rl.agent import AgentConfig, LegionAgent
    from openmanus_rl.api.agent_server import create_agent_app
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


@unittest.skipIf(not _AVAILABLE, "FastAPI not available")
class TestMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "sessions.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_history_survives_server_restart(self):
        cfg = {"model": "smart", "tools": False, "rag": False,
               "memory": True, "memory_db": self.db}
        # «сервер 1»: сохраняем обмен в файловый db
        with patch("openmanus_rl.engines.enhanced_factory.LiteLLMAdapter._make_request") as m:
            m.side_effect = [_final("stored answer")]
            TestClient(create_agent_app(cfg)).post(
                "/chat", json={"message": "remember this", "session_id": "persist"})

        # «рестарт»: новый агент на ТОТ ЖЕ файл — история должна перечитаться
        agent2 = LegionAgent(AgentConfig(model="smart", memory=True,
                                         memory_db=self.db, session_id="persist"))
        ctx = agent2.memory.get_context()
        self.assertIn({"role": "user", "content": "remember this"}, ctx)
        self.assertIn({"role": "assistant", "content": "stored answer"}, ctx)


if __name__ == "__main__":
    unittest.main()
