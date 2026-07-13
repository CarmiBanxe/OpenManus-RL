"""
Voice Pipeline Integration — Sprint 2
Wraps LegionVoicePipeline for EnhancedDecisionAgent
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VoicePipelineIntegration:
    """Интеграция голосового пайплайна Sprint 2"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.voice_pipeline: Any = None
        self._initialized = False
        logger.info("VoicePipelineIntegration initialized (pipeline not yet loaded)")

    def _load_pipeline(self) -> None:
        if self._initialized:
            return
        try:
            from openmanus_rl.integration.legion_voice_integration import (
                LegionVoicePipeline,
            )

            self.voice_pipeline = LegionVoicePipeline(self.config)
            self._initialized = True
            logger.info("LegionVoicePipeline loaded successfully")
        except Exception as exc:
            logger.warning(f"Could not load LegionVoicePipeline: {exc}")
            self._initialized = True  # mark so we don't retry on every call

    async def transcribe(
        self, audio_data: bytes, language: str = "ru"
    ) -> Dict[str, Any]:
        """Транскрибирование аудио через Legion STT"""
        self._load_pipeline()
        if self.voice_pipeline is None or self.voice_pipeline.stt is None:
            return {"text": "", "error": "STT not available", "success": False}

        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self.voice_pipeline.stt.transcribe_audio(audio_data),
            )
            return {
                "text": text or "",
                "language": language,
                "confidence": 0.85,
                "success": True,
            }
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            return {"text": "", "error": str(exc), "success": False}

    async def synthesize(self, text: str) -> Dict[str, Any]:
        """Синтез речи через Legion TTS"""
        self._load_pipeline()
        if self.voice_pipeline is None or self.voice_pipeline.tts is None:
            return {"audio": None, "error": "TTS not available", "success": False}

        try:
            loop = asyncio.get_event_loop()
            audio_bytes: Optional[bytes] = await loop.run_in_executor(
                None,
                lambda: self.voice_pipeline.tts.text_to_speech(text),
            )
            return {
                "audio": audio_bytes,
                "text": text,
                "success": audio_bytes is not None,
            }
        except Exception as exc:
            logger.error(f"Synthesis error: {exc}")
            return {"audio": None, "error": str(exc), "success": False}

    async def process_voice_command(
        self,
        audio_data: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Полная цепочка: аудио → текст → обогащение контекста"""
        transcript_result = await self.transcribe(audio_data)

        if not transcript_result.get("success"):
            return {
                "transcription": "",
                "context_enrichment": {},
                "voice_features": {},
                "success": False,
                "error": transcript_result.get("error", "unknown"),
            }

        text = transcript_result["text"]
        enriched = self._enrich_context(text, context)

        return {
            "transcription": text,
            "context_enrichment": enriched,
            "voice_features": transcript_result,
            "success": True,
        }

    def optimize_for_low_latency(self) -> Dict[str, Any]:
        """Настройки для минимальной задержки"""
        self._load_pipeline()
        if self.voice_pipeline is None:
            return {"optimizations_applied": [], "error": "pipeline not loaded"}

        optimizations: List[str] = []
        try:
            stt = getattr(self.voice_pipeline, "stt", None)
            whisper_model = getattr(stt, "whisper_model", None) if stt else None
            if whisper_model is not None:
                cfg = getattr(whisper_model, "config", None)
                if cfg is not None:
                    cfg["beam_size"] = 1
                    cfg["best_of"] = 1
                optimizations.append("stt_beam_size_reduced")
        except Exception as exc:
            logger.debug(f"STT latency opt skipped: {exc}")

        try:
            tts = getattr(self.voice_pipeline, "tts", None)
            tts_model = getattr(tts, "tts_model", None) if tts else None
            if tts_model is not None:
                cfg = getattr(tts_model, "config", None)
                if cfg is not None:
                    cfg["speed"] = 1.2
                optimizations.append("tts_speed_increased")
        except Exception as exc:
            logger.debug(f"TTS latency opt skipped: {exc}")

        return {"optimizations_applied": optimizations}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enrich_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        enriched: Dict[str, Any] = dict(context) if context else {}
        enriched.update(
            {
                "voice_input": text,
                "has_voice_input": True,
                "voice_word_count": len(text.split()),
                "voice_length": len(text),
            }
        )
        return enriched
