"""Тесты SemanticMemory + EmbeddingProvider (RAG, S14)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.memory.embeddings import EmbeddingProvider
from openmanus_rl.memory.semantic_memory import SemanticMemory


class FakeEmbeddings(EmbeddingProvider):
    """Детерминированные орто-векторы по ключевому слову (для тестов)."""

    VECS = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "weather": [0.0, 0.0, 1.0]}

    def embed(self, text):
        t = text.lower()
        for key, vec in self.VECS.items():
            if key in t:
                return vec
        return [0.0, 0.0, 0.0]


class TestSemanticMemory(unittest.TestCase):
    def setUp(self):
        self.mem = SemanticMemory(FakeEmbeddings(), ":memory:")

    def tearDown(self):
        self.mem.close()

    def test_add_turn_stores_embedding(self):
        self.mem.add_turn("s1", "user", "I have a cat")
        with self.mem._lock:
            row = self.mem._conn.execute(
                "SELECT embedding FROM turns WHERE session_id='s1'").fetchone()
        self.assertIsNotNone(row["embedding"])

    def test_semantic_search_ranks_by_meaning(self):
        self.mem.add_turn("s1", "user", "I have a cat named Felix")
        self.mem.add_turn("s1", "user", "I own a dog")
        self.mem.add_turn("s1", "user", "The weather is nice")
        hits = self.mem.semantic_search("s1", "tell me about the cat", limit=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("cat", hits[0]["content"].lower())
        self.assertGreater(hits[0]["score"], 0.9)

    def test_semantic_search_orders_multiple(self):
        self.mem.add_turn("s1", "user", "dog stuff")
        self.mem.add_turn("s1", "user", "cat stuff")
        hits = self.mem.semantic_search("s1", "about a cat", limit=2)
        self.assertEqual(hits[0]["content"], "cat stuff")

    def test_fallback_without_provider(self):
        mem = SemanticMemory(provider=None, db_path=":memory:")
        mem.add_turn("s2", "user", "quick brown fox")
        mem.add_turn("s2", "user", "lazy dog")
        # без провайдера semantic_search -> substring search
        hits = mem.semantic_search("s2", "fox", limit=3)
        self.assertTrue(any("fox" in h["content"] for h in hits))
        mem.close()

    def test_inherits_sqlite_api(self):
        # унаследованные методы S13 работают (get_turns/count/trim_to)
        for i in range(4):
            self.mem.add_turn("s3", "user", f"m{i}")
        self.assertEqual(self.mem.count("s3"), 4)
        self.mem.trim_to("s3", keep=2)
        self.assertEqual([t["content"] for t in self.mem.get_turns("s3")], ["m2", "m3"])


if __name__ == "__main__":
    unittest.main()
