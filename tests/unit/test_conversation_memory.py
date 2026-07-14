"""Тесты SQLiteMemory + ConversationMemory (диалоговый слой S13)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.memory.conversation_memory import ConversationMemory
from openmanus_rl.memory.sqlite_memory import SQLiteMemory


class TestSQLiteMemory(unittest.TestCase):
    def setUp(self):
        self.mem = SQLiteMemory(":memory:")

    def tearDown(self):
        self.mem.close()

    def test_add_and_get_turns_ordered(self):
        for role, content in [("user", "hi"), ("assistant", "hello"), ("user", "bye")]:
            self.mem.add_turn("s1", role, content)
        turns = self.mem.get_turns("s1")
        self.assertEqual([(t["role"], t["content"]) for t in turns],
                         [("user", "hi"), ("assistant", "hello"), ("user", "bye")])

    def test_get_turns_limit_returns_last_in_order(self):
        for i in range(5):
            self.mem.add_turn("s1", "user", f"m{i}")
        turns = self.mem.get_turns("s1", limit=2)
        self.assertEqual([t["content"] for t in turns], ["m3", "m4"])

    def test_session_isolation(self):
        self.mem.add_turn("s1", "user", "a")
        self.mem.add_turn("s2", "user", "b")
        self.assertEqual(self.mem.count("s1"), 1)
        self.assertEqual(self.mem.count("s2"), 1)
        self.assertEqual(self.mem.get_turns("s1")[0]["content"], "a")

    def test_search(self):
        self.mem.add_turn("s1", "user", "the quick brown fox")
        self.mem.add_turn("s1", "assistant", "a lazy dog")
        hits = self.mem.search("s1", "fox", limit=3)
        self.assertEqual(len(hits), 1)
        self.assertIn("fox", hits[0]["content"])

    def test_trim_to(self):
        for i in range(6):
            self.mem.add_turn("s1", "user", f"m{i}")
        removed = self.mem.trim_to("s1", keep=2)
        self.assertEqual(removed, 4)
        self.assertEqual([t["content"] for t in self.mem.get_turns("s1")], ["m4", "m5"])

    def test_clear(self):
        self.mem.add_turn("s1", "user", "x")
        self.mem.clear("s1")
        self.assertEqual(self.mem.count("s1"), 0)


class TestConversationMemory(unittest.TestCase):
    def setUp(self):
        self.backend = SQLiteMemory(":memory:")
        self.conv = ConversationMemory(self.backend, session_id="s1", max_turns=20)

    def tearDown(self):
        self.backend.close()

    def test_store_and_context(self):
        self.conv.store_turn("user", "hi")
        self.conv.store_turn("assistant", "hello")
        ctx = self.conv.get_context()
        self.assertEqual(ctx, [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}])

    def test_messages_with_context_prepends(self):
        self.conv.store_turn("user", "prior")
        merged = self.conv.get_messages_with_context([{"role": "user", "content": "now"}])
        self.assertEqual(merged[0]["content"], "prior")
        self.assertEqual(merged[-1]["content"], "now")

    def test_query(self):
        self.conv.store_turn("user", "talk about pandas")
        self.conv.store_turn("assistant", "pandas are bears")
        hits = self.conv.query("pandas", limit=5)
        self.assertTrue(any("pandas" in h["content"] for h in hits))

    def test_trim_truncate_no_summary(self):
        conv = ConversationMemory(self.backend, session_id="t", max_turns=3, summarize=False)
        for i in range(6):
            conv.store_turn("user", f"m{i}")
        self.assertTrue(conv.trim_if_needed())
        self.assertEqual(conv.count(), 3)
        self.assertEqual([t["content"] for t in self.backend.get_turns("t")], ["m3", "m4", "m5"])

    def test_trim_with_summary(self):
        calls = {}

        def fake_summarizer(steps, **kwargs):
            calls["steps"] = steps
            return "SUM"

        conv = ConversationMemory(self.backend, session_id="g", max_turns=4, keep_recent=2,
                                  summarize=True, summarizer=fake_summarizer)
        for i in range(6):
            conv.store_turn("user", f"m{i}")
        self.assertTrue(conv.trim_if_needed())
        turns = self.backend.get_turns("g")
        self.assertEqual(turns[0]["role"], "system")
        self.assertTrue(turns[0]["content"].startswith("[Summary of earlier conversation] SUM"))
        # последние keep_recent сохранены
        self.assertEqual([t["content"] for t in turns[1:]], ["m4", "m5"])
        self.assertIn("steps", calls)

    def test_no_trim_under_limit(self):
        self.conv.store_turn("user", "x")
        self.assertFalse(self.conv.trim_if_needed())


if __name__ == "__main__":
    unittest.main()
