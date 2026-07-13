"""
Интеграционные тесты сквозного сценария voice-to-decision — под РЕАЛЬНЫЙ API.

Реальные пути:
  select_action(state, actions, voice_input=<bytes>) — голосовой ввод через пайплайн.
  process_voice_input_advanced(audio_data, base_context=None) ->
    {text_response, confidence, modalities_used, model, risk_analysis,
     signature_analysis, sandbox, metadata}
Voice-пайплайн деградирует корректно в sandbox (без реальной whisper-модели).
"""
import asyncio
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent

SANDBOX_CFG = {
    "qwen3_omni": {"sandbox_mode": True},
    "voice_pipeline": {"sandbox_mode": True},
    "mean_field_games": {"num_agents": 8, "state_dim": 2, "max_iterations": 2, "time_horizon": 3},
}
ACTIONS = ["buy", "sell", "wait"]


def _pcm_bytes(n: int = 1600) -> bytes:
    return np.random.randn(n).astype(np.float32).tobytes()


class TestVoiceToDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = EnhancedDecisionAgent(config=SANDBOX_CFG)
        if cls.agent.performance_optimizer is not None:
            cls.addClassCleanup(cls.agent.performance_optimizer.cleanup)

    def test_voice_select_action(self) -> None:
        res = asyncio.run(self.agent.select_action({"text": ""}, ACTIONS, voice_input=_pcm_bytes()))
        self.assertIn("action", res)
        self.assertIn("osint_enhanced", res)
        self.assertIn(res["action"], ACTIONS + ["wait", "error"])

    def test_process_voice_input_advanced(self) -> None:
        res = asyncio.run(self.agent.process_voice_input_advanced(_pcm_bytes()))
        for key in ("text_response", "confidence", "modalities_used", "risk_analysis",
                    "signature_analysis", "sandbox"):
            self.assertIn(key, res)

    def test_voice_advanced_then_decision(self) -> None:
        voice = asyncio.run(self.agent.process_voice_input_advanced(_pcm_bytes()))
        transcript = voice.get("text_response", "")
        decision = asyncio.run(self.agent.select_action({"text": transcript}, ACTIONS))
        self.assertIn("action", decision)
        self.assertIn("confidence", decision)

    def test_voice_input_ndarray(self) -> None:
        arr = np.random.randn(1600).astype(np.float32)
        res = asyncio.run(self.agent.process_voice_input_advanced(arr))
        self.assertIsInstance(res, dict)
        self.assertIn("text_response", res)


if __name__ == "__main__":
    unittest.main()
