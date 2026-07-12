"""
Голосовая интеграция для системы Legion.
STT: Faster-Whisper (AMD ROCm) | TTS: Kokoro-82M (ONNX)
"""
import io
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class LegionSTTIntegration:
    """STT через Faster-Whisper с поддержкой AMD ROCm."""

    def __init__(
        self,
        model_name: str = "faster-whisper-large-v3-turbo",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.whisper_model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import faster_whisper  # type: ignore[import]

            self.whisper_model = faster_whisper.WhisperModel(
                self.model_name, device=self.device, compute_type="float16"
            )
            logger.info(f"Loaded {self.model_name} on {self.device}")
        except ImportError:
            logger.warning("faster-whisper not installed — STT unavailable")

    def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        if self.whisper_model is None:
            return ""
        try:
            if isinstance(audio_data, bytes):
                import soundfile as sf  # type: ignore[import]

                array, _ = sf.read(io.BytesIO(audio_data))
                audio_data = array
            segments, _ = self.whisper_model.transcribe(audio_data, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            return ""

    def enhance_decision_context(
        self,
        audio_input: Union[bytes, np.ndarray],
        decision_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            **decision_context,
            'voice_input': self.transcribe_audio(audio_input),
            'input_modality': 'voice',
        }


class LegionTTSIntegration:
    """TTS через Kokoro-82M (до 210× real-time, AMD ROCm via ONNX)."""

    def __init__(
        self,
        model_name: str = "kokoro-82m",
        voice: str = "af_sky",
    ) -> None:
        self.model_name = model_name
        self.voice = voice
        self.tts_model: Any = None
        self.voice_cache: Dict[str, bytes] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            import kokoro  # type: ignore[import]

            self.tts_model = kokoro.Kokoro(self.model_name)
            logger.info(f"Loaded {self.model_name} TTS")
        except ImportError:
            logger.warning("kokoro not installed — TTS unavailable")

    def text_to_speech(self, text: str) -> Optional[bytes]:
        if not text or self.tts_model is None:
            return None
        if text in self.voice_cache:
            return self.voice_cache[text]
        try:
            audio: bytes = self.tts_model.generate(text, voice=self.voice)
            self.voice_cache[text] = audio
            return audio
        except Exception as exc:
            logger.error(f"TTS error: {exc}")
            return None


class LegionVoicePipeline:
    """Полный голосовой pipeline: STT → decision stub (Sprint 2 wires engine) → TTS."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.stt = LegionSTTIntegration(
            model_name=cfg.get('stt_model', 'faster-whisper-large-v3-turbo'),
            device=cfg.get('device', 'cuda'),
        )
        self.tts = LegionTTSIntegration(
            model_name=cfg.get('tts_model', 'kokoro-82m'),
            voice=cfg.get('voice', 'af_sky'),
        )
        logger.info("LegionVoicePipeline initialized")

    def process_voice_interaction(
        self, audio_input: Union[bytes, np.ndarray]
    ) -> Optional[bytes]:
        text = self.stt.transcribe_audio(audio_input)
        if not text:
            return None
        # Sprint 2: заменить заглушку на реальный decision engine
        reply_text = f"Processed: {text}"
        return self.tts.text_to_speech(reply_text)
