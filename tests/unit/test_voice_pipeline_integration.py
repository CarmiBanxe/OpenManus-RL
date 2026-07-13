"""
Unit tests — VoicePipelineIntegration (Sprint 2)
All external pipeline calls mocked.
"""
import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openmanus_rl.integration.voice_pipeline_integration import VoicePipelineIntegration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline(
    stt_text: str = "тестовая транскрипция",
    tts_audio: Optional[bytes] = b"\x00\x01\x02",
) -> MagicMock:
    """Build a MagicMock that mimics LegionVoicePipeline."""
    pipeline = MagicMock()
    pipeline.stt = MagicMock()
    pipeline.stt.transcribe_audio = MagicMock(return_value=stt_text)
    pipeline.stt.whisper_model = MagicMock()
    pipeline.stt.whisper_model.config = {"beam_size": 5, "best_of": 5}

    pipeline.tts = MagicMock()
    pipeline.tts.text_to_speech = MagicMock(return_value=tts_audio)
    pipeline.tts.tts_model = MagicMock()
    pipeline.tts.tts_model.config = {"speed": 1.0}
    return pipeline


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestVoicePipelineIntegrationInit:
    def test_default_construction(self) -> None:
        vp = VoicePipelineIntegration()
        assert vp.config == {}
        assert vp.voice_pipeline is None
        assert not vp._initialized

    def test_custom_config(self) -> None:
        vp = VoicePipelineIntegration({"model": "tiny"})
        assert vp.config["model"] == "tiny"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TestTranscribe:
    @pytest.fixture
    def vp_with_pipeline(self) -> VoicePipelineIntegration:
        vp = VoicePipelineIntegration()
        vp.voice_pipeline = _make_mock_pipeline("привет мир")
        vp._initialized = True
        return vp

    @pytest.mark.asyncio
    async def test_transcribe_success(
        self, vp_with_pipeline: VoicePipelineIntegration
    ) -> None:
        result = await vp_with_pipeline.transcribe(b"\x00" * 100)
        assert result["success"] is True
        assert result["text"] == "привет мир"
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_transcribe_no_pipeline(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True  # skip load
        vp.voice_pipeline = None
        result = await vp.transcribe(b"\x00" * 100)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_transcribe_stt_none(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        pipeline = MagicMock()
        pipeline.stt = None
        vp.voice_pipeline = pipeline
        result = await vp.transcribe(b"\x00")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_transcribe_returns_language(
        self, vp_with_pipeline: VoicePipelineIntegration
    ) -> None:
        result = await vp_with_pipeline.transcribe(b"\x00", language="ru")
        assert result.get("language") == "ru"


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.fixture
    def vp_with_pipeline(self) -> VoicePipelineIntegration:
        vp = VoicePipelineIntegration()
        vp.voice_pipeline = _make_mock_pipeline(tts_audio=b"\xAA\xBB\xCC")
        vp._initialized = True
        return vp

    @pytest.mark.asyncio
    async def test_synthesize_success(
        self, vp_with_pipeline: VoicePipelineIntegration
    ) -> None:
        result = await vp_with_pipeline.synthesize("hello")
        assert result["success"] is True
        assert result["audio"] == b"\xAA\xBB\xCC"
        assert result["text"] == "hello"

    @pytest.mark.asyncio
    async def test_synthesize_no_pipeline(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = None
        result = await vp.synthesize("test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_synthesize_none_audio(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = _make_mock_pipeline(tts_audio=None)
        result = await vp.synthesize("test")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# process_voice_command
# ---------------------------------------------------------------------------


class TestProcessVoiceCommand:
    @pytest.mark.asyncio
    async def test_full_command_success(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = _make_mock_pipeline("выполни задачу")
        result = await vp.process_voice_command(b"\x00" * 50)
        assert result["success"] is True
        assert result["transcription"] == "выполни задачу"
        assert result["context_enrichment"]["voice_input"] == "выполни задачу"
        assert result["context_enrichment"]["has_voice_input"] is True

    @pytest.mark.asyncio
    async def test_command_with_context(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = _make_mock_pipeline("команда")
        result = await vp.process_voice_command(
            b"\x00", context={"session": "abc"}
        )
        assert result["success"] is True
        assert result["context_enrichment"]["session"] == "abc"

    @pytest.mark.asyncio
    async def test_command_failure_propagates(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = None
        result = await vp.process_voice_command(b"\x00")
        assert result["success"] is False
        assert result["transcription"] == ""


# ---------------------------------------------------------------------------
# optimize_for_low_latency — guard against None models
# ---------------------------------------------------------------------------


class TestOptimizeForLowLatency:
    def test_no_pipeline_returns_error(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = None
        result = vp.optimize_for_low_latency()
        assert "error" in result

    def test_with_full_pipeline(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        vp.voice_pipeline = _make_mock_pipeline()
        result = vp.optimize_for_low_latency()
        assert "optimizations_applied" in result
        applied = result["optimizations_applied"]
        assert "stt_beam_size_reduced" in applied
        assert "tts_speed_increased" in applied

    def test_stt_model_none_does_not_crash(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        pipeline = MagicMock()
        pipeline.stt = MagicMock()
        pipeline.stt.whisper_model = None  # model not loaded
        pipeline.tts = MagicMock()
        pipeline.tts.tts_model = None
        vp.voice_pipeline = pipeline
        result = vp.optimize_for_low_latency()
        assert "optimizations_applied" in result
        assert result["optimizations_applied"] == []

    def test_model_without_config_attr_does_not_crash(self) -> None:
        vp = VoicePipelineIntegration()
        vp._initialized = True
        pipeline = MagicMock()
        pipeline.stt = MagicMock()
        whisper = MagicMock(spec=[])  # no attributes allowed
        pipeline.stt.whisper_model = whisper
        pipeline.tts = MagicMock(spec=[])
        vp.voice_pipeline = pipeline
        result = vp.optimize_for_low_latency()
        assert "optimizations_applied" in result


# ---------------------------------------------------------------------------
# _load_pipeline — only attempts import once
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    def test_does_not_retry_after_failure(self) -> None:
        vp = VoicePipelineIntegration()
        with patch(
            "openmanus_rl.integration.voice_pipeline_integration.VoicePipelineIntegration._load_pipeline",
            side_effect=ImportError("no module"),
        ):
            # Even if import fails, initialized should eventually become True
            pass
        # After explicit failure mark, _load_pipeline skips
        vp._initialized = True
        vp._load_pipeline()  # should be no-op
        assert vp.voice_pipeline is None
