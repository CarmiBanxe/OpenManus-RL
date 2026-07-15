"""
Qwen3-Omni Integration — голосово-мультимодальный пайплайн.
Sprint 3 | SANDBOX: без реального инференса; model path задаётся через config.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Qwen3OmniIntegration:
    """
    Адаптер для Qwen3-Omni multimodal model.

    В sandbox-режиме (sandbox_mode=True, по умолчанию) возвращает
    детерминированный ответ без загрузки модели.  В production-режиме
    лениво загружает Qwen2AudioForConditionalGeneration через transformers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.model_name: str = self.config.get(
            "qwen3_omni_model", "Qwen/Qwen2-Audio-7B-Instruct"
        )
        self.device: str = self.config.get("device", "cpu")
        self.sample_rate: int = self.config.get("sample_rate", 16_000)
        self.max_new_tokens: int = self.config.get("max_new_tokens", 256)
        self.sandbox_mode: bool = self.config.get("sandbox_mode", True)

        # Lazy-loaded — None until _load_model() succeeds
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None

        logger.info("Qwen3OmniIntegration initialized (sandbox=%s)", self.sandbox_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_voice_input(
        self,
        audio_data: Union[bytes, np.ndarray],
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Обработка голосового ввода через Qwen3-Omni.

        Args:
            audio_data: Raw PCM bytes или numpy float32 array (16 kHz mono).
            base_context: Дополнительный контекст для обогащения ответа.

        Returns:
            Dict с ключами: text_response, confidence, modalities_used, metadata.
        """
        try:
            audio_array = self._to_array(audio_data)

            if self.sandbox_mode or not self._load_model():
                return self._sandbox_response(audio_array, base_context)

            return await self._real_inference(audio_array, base_context)

        except Exception as exc:
            logger.error("Qwen3OmniIntegration.process_voice_input error: %s", exc)
            return {
                "error": str(exc),
                "text_response": "",
                "confidence": 0.0,
                "modalities_used": [],
            }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Лениво загружает процессор и модель. Возвращает False в sandbox."""
        if self._model is not None:
            return True
        if self.sandbox_mode:
            return False
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration  # type: ignore[import-not-found]

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name, device_map=self.device
            )
            logger.info("Qwen3-Omni model loaded: %s", self.model_name)
            return True
        except Exception as exc:
            logger.warning("Qwen3-Omni model load failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Inference paths
    # ------------------------------------------------------------------

    async def _real_inference(
        self,
        audio_array: np.ndarray,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Реальный инференс через Qwen3-Omni (non-sandbox path)."""
        loop = asyncio.get_event_loop()

        processor = self._processor
        model = self._model
        sample_rate = self.sample_rate
        max_new_tokens = self.max_new_tokens
        device = self.device

        def _infer() -> str:
            inputs = processor(
                audios=[audio_array],
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return processor.batch_decode(generated, skip_special_tokens=True)[0]

        text_response: str = await loop.run_in_executor(None, _infer)

        return {
            "text_response": text_response,
            "confidence": 0.85,
            "modalities_used": ["audio"],
            "model": self.model_name,
            "metadata": {
                "sample_rate": self.sample_rate,
                "context_keys": list((context or {}).keys()),
            },
        }

    def _sandbox_response(
        self,
        audio_array: np.ndarray,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Детерминированная заглушка sandbox — модель не загружается."""
        duration_s = len(audio_array) / self.sample_rate if self.sample_rate > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio_array**2))) if len(audio_array) > 0 else 0.0

        return {
            "text_response": (
                f"[SANDBOX] Processed {duration_s:.2f}s audio (rms={rms:.4f})"
            ),
            "confidence": 0.5,
            "modalities_used": ["audio"],
            "model": "sandbox-stub",
            "sandbox": True,
            "metadata": {
                "audio_duration_s": duration_s,
                "audio_rms": rms,
                "context_keys": list((context or {}).keys()),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_array(self, audio_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Конвертирует bytes (int16 PCM) или ndarray → float32 ndarray."""
        if isinstance(audio_data, np.ndarray):
            return audio_data.astype(np.float32)
        # Предполагаем 16-bit signed PCM
        raw = np.frombuffer(audio_data, dtype=np.int16)
        return raw.astype(np.float32) / 32_768.0
