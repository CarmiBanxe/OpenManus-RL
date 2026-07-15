"""Тесты persona/prompt/guardrails (S22)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agent import AgentConfig, LegionAgent
from openmanus_rl.agent.persona import (PERSONAS, GuardrailError, Guardrails,
                                        PromptTemplate, resolve_system_prompt)


class TestPromptTemplate(unittest.TestCase):
    def test_render_with_vars(self):
        self.assertEqual(PromptTemplate("Hi {name}").render(name="Zara"), "Hi Zara")

    def test_missing_var_is_empty(self):
        self.assertEqual(PromptTemplate("Hi {name}!").render(), "Hi !")


class TestPersonaResolve(unittest.TestCase):
    def test_builtin_personas(self):
        for name in ("assistant", "concise", "coder", "analyst"):
            self.assertIn(name, PERSONAS)

    def test_explicit_overrides_persona(self):
        self.assertEqual(resolve_system_prompt("coder", "custom prompt"), "custom prompt")

    def test_persona_used_when_no_explicit(self):
        self.assertIn("software engineer", resolve_system_prompt("coder", None))

    def test_none_when_neither(self):
        self.assertIsNone(resolve_system_prompt(None, None))
        self.assertIsNone(resolve_system_prompt("nonexistent", None))


class TestGuardrails(unittest.TestCase):
    def test_clean_passes(self):
        Guardrails(max_input_chars=100).check("hello")  # no raise

    def test_oversized_rejected(self):
        with self.assertRaises(GuardrailError):
            Guardrails(max_input_chars=5).check("way too long")

    def test_deny_pattern(self):
        with self.assertRaises(GuardrailError):
            Guardrails(deny_patterns=["forbidden"]).check("this is FORBIDDEN text")

    def test_empty_deny_no_censorship(self):
        Guardrails().check("anything goes on the uncensored engine")  # no raise


class TestLegionAgentPersona(unittest.TestCase):
    def test_system_prompt_prepended(self):
        agent = LegionAgent(AgentConfig(persona="coder", memory=False))
        msgs = agent._prepare("write a function")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("software engineer", msgs[0]["content"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "write a function"})

    def test_no_system_when_no_persona(self):
        agent = LegionAgent(AgentConfig(memory=False))
        msgs = agent._prepare("hi")
        self.assertTrue(all(m["role"] != "system" for m in msgs))

    def test_guardrail_rejects_oversized(self):
        agent = LegionAgent(AgentConfig(max_input_chars=10, memory=False))
        with self.assertRaises(GuardrailError):
            agent._prepare("x" * 50)


try:
    from fastapi.testclient import TestClient

    from openmanus_rl.api.agent_server import create_agent_app
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


@unittest.skipIf(not _AVAILABLE, "FastAPI not available")
class TestServerGuardrail(unittest.TestCase):
    def test_oversized_input_400(self):
        client = TestClient(create_agent_app({
            "model": "smart", "memory": True, "memory_db": ":memory:", "max_input_chars": 5}))
        r = client.post("/chat", json={"message": "way too long", "session_id": "g"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
