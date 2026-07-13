"""
Tests — Qwen3OmniIntegration (Sprint 3).
Все тесты используют sandbox_mode=True (без загрузки реальной модели).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from openmanus_rl.integration.qwen3_omni_integration import Qwen3OmniIntegration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm_bytes(n_samples: int = 1_600) -> bytes:
    """16-bit signed PCM, ~0.1s at 16kHz."""
    arr = (np.random.randn(n_samples) * 1_000).astype(np.int16)
    return arr.tobytes()


def _make_float_array(n_samples: int = 1_600) -> np.ndarray:
    return np.random.randn(n_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestQwen3OmniInit:
    def test_defaults(self) -> None:
        q = Qwen3OmniIntegration()
        assert q.sandbox_mode is True
        assert q.sample_rate == 16_000
        assert q.max_new_tokens == 256
        assert q._model is None
        assert q._processor is None

    def test_custom_config(self) -> None:
        q = Qwen3OmniIntegration({"sample_rate": 8_000, "max_new_tokens": 128})
        assert q.sample_rate == 8_000
        assert q.max_new_tokens == 128

    def test_model_name_default(self) -> None:
        q = Qwen3OmniIntegration()
        assert "Qwen" in q.model_name

    def test_model_name_custom(self) -> None:
        q = Qwen3OmniIntegration({"qwen3_omni_model": "custom/model"})
        assert q.model_name == "custom/model"

    def test_device_default(self) -> None:
        q = Qwen3OmniIntegration()
        assert q.device == "cpu"


# ---------------------------------------------------------------------------
# _to_array
# ---------------------------------------------------------------------------


class TestToArray:
    def test_bytes_pcm_to_float32(self) -> None:
        q = Qwen3OmniIntegration()
        raw = _make_pcm_bytes(1_600)
        arr = q._to_array(raw)
        assert arr.dtype == np.float32
        assert len(arr) == 1_600
        assert arr.max() <= 1.0 + 1e-6

    def test_ndarray_passthrough(self) -> None:
        q = Qwen3OmniIntegration()
        inp = np.array([0.1, 0.2, 0.3])
        out = q._to_array(inp)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, inp.astype(np.float32))

    def test_empty_bytes(self) -> None:
        q = Qwen3OmniIntegration()
        out = q._to_array(b"")
        assert out.dtype == np.float32
        assert len(out) == 0

    def test_float64_ndarray_cast(self) -> None:
        q = Qwen3OmniIntegration()
        inp = np.array([1.0, 2.0], dtype=np.float64)
        out = q._to_array(inp)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Sandbox mode
# ---------------------------------------------------------------------------


class TestSandboxMode:
    def test_sandbox_response_bytes(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        raw = _make_pcm_bytes()
        arr = q._to_array(raw)
        result = q._sandbox_response(arr, None)
        assert "text_response" in result
        assert "SANDBOX" in result["text_response"]
        assert result["sandbox"] is True
        assert result["confidence"] == 0.5

    def test_sandbox_response_metadata(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        arr = np.zeros(3_200, dtype=np.float32)
        ctx: Dict[str, Any] = {"key1": "val1"}
        result = q._sandbox_response(arr, ctx)
        assert "audio_duration_s" in result["metadata"]
        assert "context_keys" in result["metadata"]
        assert "key1" in result["metadata"]["context_keys"]

    def test_sandbox_duration_calculation(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True, "sample_rate": 16_000})
        arr = np.zeros(16_000, dtype=np.float32)
        result = q._sandbox_response(arr, None)
        assert abs(result["metadata"]["audio_duration_s"] - 1.0) < 1e-3

    def test_sandbox_rms_silent(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        arr = np.zeros(100, dtype=np.float32)
        result = q._sandbox_response(arr, None)
        assert result["metadata"]["audio_rms"] == 0.0

    def test_sandbox_modalities_used(self) -> None:
        q = Qwen3OmniIntegration()
        arr = np.zeros(100, dtype=np.float32)
        result = q._sandbox_response(arr, None)
        assert "audio" in result["modalities_used"]


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_model_returns_false_in_sandbox(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        assert q._load_model() is False

    def test_load_model_returns_true_when_already_loaded(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": False})
        q._model = MagicMock()  # simulate already loaded
        assert q._load_model() is True

    def test_load_model_handles_import_error(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": False})
        with patch("builtins.__import__", side_effect=ImportError("no transformers")):
            result = q._load_model()
        assert result is False


# ---------------------------------------------------------------------------
# process_voice_input (async)
# ---------------------------------------------------------------------------


class TestProcessVoiceInput:
    def test_bytes_input_sandbox(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        raw = _make_pcm_bytes()
        result = asyncio.run(
            q.process_voice_input(raw, None)
        )
        assert "text_response" in result
        assert "error" not in result

    def test_ndarray_input_sandbox(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        arr = _make_float_array()
        result = asyncio.run(
            q.process_voice_input(arr, None)
        )
        assert result["confidence"] == 0.5

    def test_context_passed_to_metadata(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        ctx = {"user": "test", "session": "abc"}
        raw = _make_pcm_bytes()
        result = asyncio.run(
            q.process_voice_input(raw, ctx)
        )
        meta = result.get("metadata", {})
        assert "user" in meta.get("context_keys", [])

    def test_error_returns_error_key(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        with patch.object(q, "_to_array", side_effect=RuntimeError("oops")):
            result = asyncio.run(
                q.process_voice_input(b"bad", None)
            )
        assert "error" in result
        assert result["confidence"] == 0.0

    def test_model_stub_in_non_sandbox_with_mock(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": False})
        q._model = MagicMock()
        q._processor = MagicMock()

        async def fake_real_inference(arr: Any, ctx: Any) -> Dict[str, Any]:
            return {"text_response": "hello", "confidence": 0.85, "modalities_used": ["audio"], "model": "mock"}

        with patch.object(q, "_real_inference", side_effect=fake_real_inference):
            result = asyncio.run(
                q.process_voice_input(_make_float_array(), None)
            )
        assert result["confidence"] == 0.85

    def test_sample_rate_zero_no_division_error(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True, "sample_rate": 0})
        arr = np.zeros(100, dtype=np.float32)
        result = asyncio.run(
            q.process_voice_input(arr, None)
        )
        assert "text_response" in result


# ---------------------------------------------------------------------------
# _real_inference (mocked)
# ---------------------------------------------------------------------------


class TestRealInference:
    def test_real_inference_returns_expected_keys(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": False})
        q._model = MagicMock()
        q._processor = MagicMock()

        mock_proc = MagicMock()
        mock_proc.to.return_value = mock_proc
        q._processor.return_value = mock_proc

        generated = MagicMock()
        q._model.generate.return_value = generated
        q._processor.batch_decode.return_value = ["transcribed text"]

        arr = np.zeros(100, dtype=np.float32)
        result = asyncio.run(q._real_inference(arr, {}))
        assert "text_response" in result
        assert "confidence" in result
        assert result["model"] == q.model_name


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_long_audio(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        arr = np.random.randn(160_000).astype(np.float32)
        result = asyncio.run(
            q.process_voice_input(arr, None)
        )
        assert "text_response" in result
        assert "10.00s" in result["text_response"] or "audio" in result["text_response"]

    def test_none_context_handled(self) -> None:
        q = Qwen3OmniIntegration({"sandbox_mode": True})
        arr = np.zeros(100, dtype=np.float32)
        result = q._sandbox_response(arr, None)
        assert result["metadata"]["context_keys"] == []

    def test_high_amplitude_pcm(self) -> None:
        q = Qwen3OmniIntegration()
        arr = np.full(100, 32_767, dtype=np.int16)
        out = q._to_array(arr.tobytes())
        assert np.allclose(out, 1.0, atol=1e-3)
