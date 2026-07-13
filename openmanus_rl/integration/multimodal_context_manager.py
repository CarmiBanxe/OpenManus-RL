"""
Мультимодальный менеджер контекста для принятия решений
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class InputModality(Enum):
    """Типы модальностей ввода"""

    VOICE = "voice"
    TEXT = "text"
    OSINT = "osint"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class MultimodalContextManager:
    """Менеджер мультимодального контекста"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.modality_priorities: Dict[InputModality, float] = {
            InputModality.VOICE: 1.0,
            InputModality.TEXT: 0.8,
            InputModality.OSINT: 0.6,
            InputModality.IMAGE: 0.7,
        }
        self.max_context_length: int = self.config.get("max_context_length", 10000)
        logger.info("MultimodalContextManager initialized")

    def create_unified_context(
        self,
        inputs: List[Tuple[InputModality, Any]],
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Создание унифицированного контекста из различных модальностей"""
        try:
            unified_context: Dict[str, Any] = dict(base_context) if base_context else {}

            processed_inputs = [
                (modality, self._process_modality(modality, data))
                for modality, data in inputs
            ]

            prioritized_inputs = self._prioritize_modalities(processed_inputs)

            unified_context.update(
                {
                    "inputs": prioritized_inputs,
                    "primary_modality": (
                        prioritized_inputs[0][0]
                        if prioritized_inputs
                        else InputModality.TEXT
                    ),
                    "modality_weights": self._calculate_modality_weights(
                        prioritized_inputs
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "context_id": self._generate_context_id(prioritized_inputs),
                }
            )

            self._cache_context(unified_context)
            return unified_context

        except Exception as exc:
            logger.error(f"Context creation error: {exc}")
            return dict(base_context) if base_context else {}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _process_modality(self, modality: InputModality, data: Any) -> Dict[str, Any]:
        dispatch = {
            InputModality.VOICE: self._process_voice_input,
            InputModality.TEXT: self._process_text_input,
            InputModality.OSINT: self._process_osint_data,
            InputModality.IMAGE: self._process_image_input,
        }
        handler = dispatch.get(modality)
        if handler:
            return handler(data)
        return {"raw_data": data, "processed": False}

    # ------------------------------------------------------------------
    # Per-modality processors
    # ------------------------------------------------------------------

    def _process_voice_input(
        self, audio_data: Union[bytes, np.ndarray]
    ) -> Dict[str, Any]:
        try:
            from openmanus_rl.integration.legion_voice_integration import (
                LegionSTTIntegration,
            )

            stt = LegionSTTIntegration()
            transcription = stt.transcribe_audio(audio_data)
            return {
                "transcription": transcription,
                "features": self._extract_voice_features(audio_data),
                "confidence": 0.8,
                "modality": InputModality.VOICE,
                "processed": True,
            }
        except Exception as exc:
            logger.error(f"Voice processing error: {exc}")
            return {
                "raw_data": audio_data,
                "error": str(exc),
                "modality": InputModality.VOICE,
                "processed": False,
            }

    def _process_text_input(self, text: str) -> Dict[str, Any]:
        try:
            return {
                "text": text,
                "features": self._extract_text_features(text),
                "modality": InputModality.TEXT,
                "processed": True,
            }
        except Exception as exc:
            logger.error(f"Text processing error: {exc}")
            return {
                "raw_data": text,
                "error": str(exc),
                "modality": InputModality.TEXT,
                "processed": False,
            }

    def _process_osint_data(self, osint_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "data": self._extract_relevant_osint(osint_data),
                "risk_factors": self._extract_risk_factors(osint_data),
                "confidence": self._calculate_osint_confidence(osint_data),
                "modality": InputModality.OSINT,
                "processed": True,
            }
        except Exception as exc:
            logger.error(f"OSINT processing error: {exc}")
            return {
                "raw_data": osint_data,
                "error": str(exc),
                "modality": InputModality.OSINT,
                "processed": False,
            }

    def _process_image_input(
        self, image_data: Union[bytes, np.ndarray]
    ) -> Dict[str, Any]:
        try:
            return {
                "features": self._extract_image_features(image_data),
                "modality": InputModality.IMAGE,
                "processed": True,
            }
        except Exception as exc:
            logger.error(f"Image processing error: {exc}")
            return {
                "raw_data": image_data,
                "error": str(exc),
                "modality": InputModality.IMAGE,
                "processed": False,
            }

    # ------------------------------------------------------------------
    # Prioritisation & weights
    # ------------------------------------------------------------------

    def _prioritize_modalities(
        self,
        inputs: List[Tuple[InputModality, Dict[str, Any]]],
    ) -> List[Tuple[InputModality, Dict[str, Any]]]:
        return sorted(
            inputs,
            key=lambda x: self.modality_priorities.get(x[0], 0.5),
            reverse=True,
        )

    def _calculate_modality_weights(
        self,
        inputs: List[Tuple[InputModality, Dict[str, Any]]],
    ) -> Dict[str, float]:
        total = sum(
            self.modality_priorities.get(modality, 0.5) for modality, _ in inputs
        )
        return {
            modality.value: (
                self.modality_priorities.get(modality, 0.5) / total
                if total > 0
                else 0.0
            )
            for modality, _ in inputs
        }

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _generate_context_id(
        self,
        inputs: List[Tuple[InputModality, Dict[str, Any]]],
    ) -> str:
        context_str = json.dumps(
            [
                (
                    modality.value,
                    data.get("text", data.get("transcription", str(data))),
                )
                for modality, data in inputs
            ],
            sort_keys=True,
        )
        return hashlib.md5(context_str.encode()).hexdigest()

    def _cache_context(self, context: Dict[str, Any]) -> None:
        context_id: Optional[str] = context.get("context_id")
        if not context_id:
            return
        self.context_cache[context_id] = context
        if len(self.context_cache) > 1000:
            for key in sorted(self.context_cache.keys())[:100]:
                del self.context_cache[key]

    # ------------------------------------------------------------------
    # Feature extractors (lightweight placeholders)
    # ------------------------------------------------------------------

    def _extract_voice_features(
        self, audio_data: Union[bytes, np.ndarray]
    ) -> Dict[str, Any]:
        return {
            "duration": len(audio_data) if isinstance(audio_data, (bytes, np.ndarray)) else 0,
            "energy": 0.5,
            "pitch": 0.3,
            "tempo": 0.7,
        }

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "complexity": min(len(text.split()) / 10.0, 1.0),
            "sentiment": 0.5,
        }

    def _extract_relevant_osint(
        self, osint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "entities": osint_data.get("entities", []),
            "relationships": osint_data.get("relationships", []),
            "risk_indicators": osint_data.get("risk_factors", []),
        }

    def _extract_risk_factors(self, osint_data: Dict[str, Any]) -> List[str]:
        return osint_data.get("risk_factors", [])

    def _calculate_osint_confidence(self, osint_data: Dict[str, Any]) -> float:
        return osint_data.get("confidence", 0.5)

    def _extract_image_features(
        self, image_data: Union[bytes, np.ndarray]
    ) -> Dict[str, Any]:
        return {
            "size": len(image_data) if isinstance(image_data, bytes) else 0,
            "format": "unknown",
            "complexity": 0.5,
        }
