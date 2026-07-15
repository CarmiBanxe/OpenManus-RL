"""
Встроенные eval-наборы для компонентов S10–S15.

Детерминированные (tools/memory/rag) — без сети, зелёный гейт. Live-LLM набор
(adapter/tool-calling) ходит на живой шлюз — только по явному запросу (CLI --live),
не в юнит-гейте.
"""
from typing import List

from .harness import EvalCase, EvalSuite


def build_tools_suite() -> EvalSuite:
    from openmanus_rl.tool_calling import ToolExecutor, ToolRegistry, register_builtins
    ex = ToolExecutor(register_builtins(ToolRegistry()))
    suite = EvalSuite("tools")
    for expr, expected in [("6*7", "42"), ("2**10", "1024"), ("(3+4)*5", "35")]:
        suite.add(EvalCase(
            f"calc:{expr}",
            run=(lambda e=expr: ex.execute("calculator", {"expression": e})),
            check=(lambda out, ex_=expected: out == ex_),
            component="tools"))
    suite.add(EvalCase(
        "safety:reject_import",
        run=(lambda: ex.execute("calculator", {"expression": "__import__('os')"})),
        check=(lambda out: out.startswith("Error")),
        component="tools"))
    return suite


def build_memory_suite() -> EvalSuite:
    from openmanus_rl.memory import SQLiteMemory
    suite = EvalSuite("memory")

    def store_fetch():
        m = SQLiteMemory(":memory:")
        m.add_turn("s", "user", "hello")
        m.add_turn("s", "assistant", "hi")
        turns = m.get_turns("s")
        m.close()
        return turns

    def trim():
        m = SQLiteMemory(":memory:")
        for i in range(5):
            m.add_turn("s", "user", f"m{i}")
        m.trim_to("s", 2)
        turns = m.get_turns("s")
        m.close()
        return turns

    suite.add(EvalCase("store_fetch_2_turns", run=store_fetch,
                       check=lambda t: len(t) == 2 and t[0]["content"] == "hello",
                       component="memory"))
    suite.add(EvalCase("trim_to_last_2", run=trim,
                       check=lambda t: [x["content"] for x in t] == ["m3", "m4"],
                       component="memory"))
    return suite


def build_rag_suite() -> EvalSuite:
    from openmanus_rl.memory import SemanticMemory
    from openmanus_rl.memory.embeddings import EmbeddingProvider

    class _Fake(EmbeddingProvider):
        V = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "weather": [0.0, 0.0, 1.0]}

        def embed(self, text):
            t = text.lower()
            for k, v in self.V.items():
                if k in t:
                    return v
            return [0.0, 0.0, 0.0]

    def rank():
        m = SemanticMemory(_Fake(), ":memory:")
        m.add_turn("s", "user", "I have a cat")
        m.add_turn("s", "user", "I have a dog")
        m.add_turn("s", "user", "nice weather today")
        hits = m.semantic_search("s", "tell me about the cat", 1)
        m.close()
        return hits

    suite = EvalSuite("rag")
    suite.add(EvalCase("semantic_top1_is_cat", run=rank,
                       check=lambda h: bool(h) and "cat" in h[0]["content"].lower(),
                       component="rag"))
    return suite


def build_default_suites() -> List[EvalSuite]:
    """Детерминированные наборы (без сети) — основа гейта."""
    return [build_tools_suite(), build_memory_suite(), build_rag_suite()]


def build_live_llm_suite(model: str = "smart") -> EvalSuite:
    """Live-набор на живой шлюз (adapter + tool-calling). НЕ для юнит-гейта."""
    from openmanus_rl.engines.enhanced_factory import LiteLLMAdapter
    from openmanus_rl.engines.tool_calling_adapter import ToolCallingAdapter

    suite = EvalSuite("live_llm")
    suite.add(EvalCase(
        "adapter_chat_responds",
        run=(lambda: LiteLLMAdapter({"model": model}).chat(
            [{"role": "user", "content": "Say OK"}], model=model, max_tokens=200)),
        check=lambda r: isinstance(r, dict) and "choices" in r,
        component="live_llm"))
    suite.add(EvalCase(
        "tool_calling_calculator",
        run=(lambda: ToolCallingAdapter({"model": model}).run(
            [{"role": "user", "content": "What is 47*89? Use the calculator, then state the result."}],
            max_tokens=400)),
        check=lambda r: "4183" in (r.get("content") or "") or
        any(t["output"] == "4183" for t in r.get("tools_used", [])),
        component="live_llm"))
    return suite


def build_agent_recall_suite(model: str = "smart") -> EvalSuite:
    """Live: агент помнит факт между turn'ами (memory через фасад)."""
    from openmanus_rl.agent import create_agent

    def recall():
        agent = create_agent({"model": model, "memory": True, "extra": {"include_reasoning": True}})
        agent.chat("My name is Zara. Acknowledge.", max_tokens=200)
        return agent.chat("What is my name?", max_tokens=200)["content"]

    suite = EvalSuite("agent_recall")
    suite.add(EvalCase("recalls_name", run=recall,
                       check=lambda out: "zara" in (out or "").lower(), component="agent_recall"))
    return suite


def build_rag_live_suite() -> EvalSuite:
    """Live: семантич. retrieval на РЕАЛЬНЫХ Ollama-эмбеддингах (по смыслу, не substring)."""
    from openmanus_rl.memory.embeddings import OllamaEmbeddingProvider
    from openmanus_rl.memory.semantic_memory import SemanticMemory

    def semantic():
        m = SemanticMemory(OllamaEmbeddingProvider("nomic-embed-text"), ":memory:")
        m.add_turn("s", "user", "My favorite programming language is Python")
        m.add_turn("s", "user", "I enjoy hiking in the mountains on weekends")
        hits = m.semantic_search("s", "what do I write code with?", 1)
        m.close()
        return hits

    suite = EvalSuite("rag_live")
    suite.add(EvalCase("semantic_retrieval", run=semantic,
                       check=lambda h: bool(h) and "Python" in h[0]["content"], component="rag_live"))
    return suite


LIVE_COMPONENTS = {"live_llm", "agent_recall", "rag_live"}


def build_live_suites(model: str = "smart") -> List[EvalSuite]:
    return [build_live_llm_suite(model), build_agent_recall_suite(model), build_rag_live_suite()]
