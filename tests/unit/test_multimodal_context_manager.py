"""
Unit tests — MultimodalContextManager (Sprint 2)
No external dependencies required.
"""
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openmanus_rl.integration.multimodal_context_manager import (
    InputModality,
    MultimodalContextManager,
)


# ---------------------------------------------------------------------------
# InputModality
# ---------------------------------------------------------------------------


class TestInputModality:
    def test_all_modalities_exist(self) -> None:
        assert InputModality.VOICE.value == "voice"
        assert InputModality.TEXT.value == "text"
        assert InputModality.OSINT.value == "osint"
        assert InputModality.IMAGE.value == "image"
        assert InputModality.MULTIMODAL.value == "multimodal"

    def test_modalities_are_distinct(self) -> None:
        values = [m.value for m in InputModality]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# MultimodalContextManager — construction
# ---------------------------------------------------------------------------


class TestMultimodalContextManagerInit:
    def test_default_config(self) -> None:
        mgr = MultimodalContextManager()
        assert mgr.max_context_length == 10000
        assert isinstance(mgr.context_cache, dict)

    def test_custom_config(self) -> None:
        mgr = MultimodalContextManager({"max_context_length": 5000})
        assert mgr.max_context_length == 5000

    def test_priority_map_covers_all_input_modalities(self) -> None:
        mgr = MultimodalContextManager()
        for modality in [
            InputModality.VOICE,
            InputModality.TEXT,
            InputModality.OSINT,
            InputModality.IMAGE,
        ]:
            assert modality in mgr.modality_priorities
            assert 0.0 <= mgr.modality_priorities[modality] <= 1.0


# ---------------------------------------------------------------------------
# Text modality processing
# ---------------------------------------------------------------------------


class TestTextProcessing:
    def setup_method(self) -> None:
        self.mgr = MultimodalContextManager()

    def test_text_input_processed_successfully(self) -> None:
        result = self.mgr._process_text_input("Hello world test")
        assert result["processed"] is True
        assert result["modality"] == InputModality.TEXT
        assert "features" in result
        assert result["text"] == "Hello world test"

    def test_text_features_word_count(self) -> None:
        features = self.mgr._extract_text_features("one two three four five")
        assert features["word_count"] == 5
        assert features["length"] == len("one two three four five")

    def test_empty_text_handled(self) -> None:
        result = self.mgr._process_text_input("")
        assert result["processed"] is True
        assert result["features"]["word_count"] == 0

    def test_complexity_capped_at_one(self) -> None:
        long_text = " ".join([f"word{i}" for i in range(200)])
        features = self.mgr._extract_text_features(long_text)
        assert features["complexity"] <= 1.0


# ---------------------------------------------------------------------------
# OSINT modality processing
# ---------------------------------------------------------------------------


class TestOSINTProcessing:
    def setup_method(self) -> None:
        self.mgr = MultimodalContextManager()

    def test_osint_processed_successfully(self) -> None:
        data: Dict[str, Any] = {
            "entities": ["Acme Corp"],
            "risk_factors": ["sanction"],
            "confidence": 0.9,
        }
        result = self.mgr._process_osint_data(data)
        assert result["processed"] is True
        assert result["confidence"] == 0.9
        assert "sanction" in result["risk_factors"]

    def test_empty_osint_handled(self) -> None:
        result = self.mgr._process_osint_data({})
        assert result["processed"] is True
        assert result["risk_factors"] == []
        assert result["confidence"] == 0.5

    def test_relevant_osint_extraction(self) -> None:
        data = {
            "entities": ["A", "B"],
            "relationships": [{"from": "A", "to": "B"}],
            "risk_factors": ["risk1"],
        }
        extracted = self.mgr._extract_relevant_osint(data)
        assert extracted["entities"] == ["A", "B"]
        assert len(extracted["relationships"]) == 1


# ---------------------------------------------------------------------------
# Image modality processing
# ---------------------------------------------------------------------------


class TestImageProcessing:
    def setup_method(self) -> None:
        self.mgr = MultimodalContextManager()

    def test_image_bytes_processed(self) -> None:
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100
        result = self.mgr._process_image_input(fake_image)
        assert result["processed"] is True
        assert result["features"]["size"] > 0

    def test_image_numpy_array(self) -> None:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = self.mgr._process_image_input(arr)
        assert result["processed"] is True


# ---------------------------------------------------------------------------
# Prioritisation
# ---------------------------------------------------------------------------


class TestModalitiPrioritization:
    def setup_method(self) -> None:
        self.mgr = MultimodalContextManager()

    def test_voice_ranked_first_by_default(self) -> None:
        inputs = [
            (InputModality.TEXT, {"text": "t", "processed": True}),
            (InputModality.VOICE, {"transcription": "v", "processed": True}),
            (InputModality.OSINT, {"data": {}, "processed": True}),
        ]
        prioritized = self.mgr._prioritize_modalities(inputs)
        assert prioritized[0][0] == InputModality.VOICE

    def test_weights_sum_to_one(self) -> None:
        inputs = [
            (InputModality.TEXT, {}),
            (InputModality.OSINT, {}),
        ]
        weights = self.mgr._calculate_modality_weights(inputs)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_inputs_weights(self) -> None:
        weights = self.mgr._calculate_modality_weights([])
        assert weights == {}


# ---------------------------------------------------------------------------
# Unified context creation
# ---------------------------------------------------------------------------


class TestCreateUnifiedContext:
    def setup_method(self) -> None:
        self.mgr = MultimodalContextManager()

    def test_creates_context_with_required_keys(self) -> None:
        ctx = self.mgr.create_unified_context(
            [(InputModality.TEXT, "hello world")]
        )
        assert "inputs" in ctx
        assert "primary_modality" in ctx
        assert "modality_weights" in ctx
        assert "timestamp" in ctx
        assert "context_id" in ctx

    def test_base_context_merged(self) -> None:
        base = {"session_id": "abc-123", "user": "test"}
        ctx = self.mgr.create_unified_context(
            [(InputModality.TEXT, "test")],
            base_context=base,
        )
        assert ctx["session_id"] == "abc-123"
        assert ctx["user"] == "test"

    def test_context_cached(self) -> None:
        ctx = self.mgr.create_unified_context(
            [(InputModality.TEXT, "cached test")]
        )
        cid = ctx["context_id"]
        assert cid in self.mgr.context_cache

    def test_empty_inputs_returns_base_context(self) -> None:
        base = {"key": "value"}
        ctx = self.mgr.create_unified_context([], base_context=base)
        assert ctx.get("key") == "value"

    def test_voice_modality_falls_back_gracefully(self) -> None:
        # No LegionSTTIntegration available in test env — should not crash
        with patch(
            "openmanus_rl.integration.multimodal_context_manager.InputModality",
            InputModality,
        ):
            ctx = self.mgr.create_unified_context(
                [(InputModality.VOICE, b"\x00" * 10)]
            )
        assert "inputs" in ctx or isinstance(ctx, dict)

    def test_context_id_deterministic(self) -> None:
        ctx1 = self.mgr.create_unified_context([(InputModality.TEXT, "same")])
        ctx2 = self.mgr.create_unified_context([(InputModality.TEXT, "same")])
        assert ctx1["context_id"] == ctx2["context_id"]

    def test_context_id_changes_with_different_input(self) -> None:
        ctx1 = self.mgr.create_unified_context([(InputModality.TEXT, "aaa")])
        ctx2 = self.mgr.create_unified_context([(InputModality.TEXT, "bbb")])
        assert ctx1["context_id"] != ctx2["context_id"]


# ---------------------------------------------------------------------------
# Cache eviction
# ---------------------------------------------------------------------------


class TestCacheEviction:
    def test_cache_does_not_grow_unbounded(self) -> None:
        mgr = MultimodalContextManager()
        for i in range(1200):
            mgr.create_unified_context([(InputModality.TEXT, f"unique text {i}")])
        assert len(mgr.context_cache) < 1200
