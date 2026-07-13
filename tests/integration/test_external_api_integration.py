"""
Тесты интеграции с внешними API через LiteLLM (реальный API adapter).

Моки — на HTTP-слое (requests.Session), реальные внешние ключи не нужны.
Реальный тест LiteLLM (TestRealLiteLLMIntegration) — под skip без env.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.enhanced_factory import (  # noqa: E402
    EnhancedEngineFactory,
    LiteLLMAdapter,
    create_engine,
    get_available_engines,
    get_engine_info,
)


class TestLiteLLMAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "base_url": "http://localhost:4000",
            "model": "gpt-3.5-turbo",
            "timeout": 60,
            "max_retries": 3,
            "fallback_models": ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
        }

    def test_init_success(self) -> None:
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            self.assertEqual(adapter.base_url, "http://localhost:4000")
            self.assertEqual(adapter.model, "gpt-3.5-turbo")
            self.assertEqual(adapter.timeout, 60)
            self.assertEqual(adapter.max_retries, 3)
            self.assertEqual(
                adapter.fallback_models,
                ["gpt-3.5-turbo", "togethercomputer/llama-2-7b-chat"],
            )

    @patch("requests.Session.get")
    def test_init_with_auth_401(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=401)
        adapter = LiteLLMAdapter(self.config)
        # 401 => сервис поднят, но требует auth => считаем доступным.
        self.assertTrue(adapter.is_available())

    @patch("requests.Session.get")
    def test_init_failure(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Connection error")
        adapter = LiteLLMAdapter(self.config)
        self.assertFalse(adapter.is_available())

    @patch("requests.Session.post")
    def test_generate_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            **{
                "json.return_value": {
                    "choices": [{"text": "Hello! How can I help you today?"}],
                    "usage": {"total_tokens": 20},
                }
            },
        )
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            result = adapter.generate("Hello, how are you?")

            self.assertIn("choices", result)
            self.assertEqual(result["choices"][0]["text"], "Hello! How can I help you today?")

            metrics = adapter.get_metrics()
            self.assertEqual(metrics["total_requests"], 1)
            self.assertEqual(metrics["successful_requests"], 1)
            self.assertEqual(metrics["failed_requests"], 0)
            self.assertGreater(metrics["total_time"], 0)
            self.assertGreater(metrics["avg_response_time"], 0)
            self.assertGreater(metrics["tokens_per_second"], 0)

    @patch("requests.Session.post")
    def test_generate_with_fallback(self, mock_post: MagicMock) -> None:
        error = MagicMock(status_code=500, text="Internal Server Error")
        success = MagicMock(
            status_code=200,
            **{
                "json.return_value": {
                    "choices": [{"text": "Hello! How can I help you today?"}],
                    "usage": {"total_tokens": 20},
                }
            },
        )
        mock_post.side_effect = [error, success]

        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            result = adapter.generate("Hello, how are you?")

            self.assertEqual(result["choices"][0]["text"], "Hello! How can I help you today?")
            metrics = adapter.get_metrics()
            self.assertEqual(metrics["total_requests"], 1)
            self.assertEqual(metrics["successful_requests"], 1)
            self.assertEqual(metrics["failed_requests"], 0)
            self.assertEqual(metrics["fallback_used"], 1)

    @patch("requests.Session.post")
    def test_chat_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            **{
                "json.return_value": {
                    "choices": [{"message": {"content": "Hello! How can I help you today?"}}],
                    "usage": {"total_tokens": 20},
                }
            },
        )
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            result = adapter.chat([{"role": "user", "content": "Hello, how are you?"}])

            self.assertEqual(
                result["choices"][0]["message"]["content"], "Hello! How can I help you today?"
            )
            metrics = adapter.get_metrics()
            self.assertEqual(metrics["total_requests"], 1)
            self.assertEqual(metrics["successful_requests"], 1)
            self.assertEqual(metrics["failed_requests"], 0)

    def test_list_models(self) -> None:
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            with patch.object(adapter, "session") as mock_session:
                mock_session.get.return_value = MagicMock(
                    status_code=200,
                    **{
                        "json.return_value": {
                            "data": [
                                {"id": "gpt-3.5-turbo"},
                                {"id": "gpt-4"},
                                {"id": "togethercomputer/llama-2-7b-chat"},
                            ]
                        }
                    },
                )
                models = adapter.list_models()
                self.assertEqual(len(models), 3)
                self.assertEqual(models[0]["id"], "gpt-3.5-turbo")
                self.assertEqual(models[1]["id"], "gpt-4")
                self.assertEqual(models[2]["id"], "togethercomputer/llama-2-7b-chat")

    def test_model_info(self) -> None:
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(self.config)
            with patch.object(adapter, "list_models") as mock_list:
                mock_list.return_value = [
                    {"id": "gpt-3.5-turbo"},
                    {"id": "gpt-4"},
                    {"id": "togethercomputer/llama-2-7b-chat"},
                ]
                self.assertEqual(adapter.model_info("gpt-3.5-turbo")["id"], "gpt-3.5-turbo")
                self.assertEqual(adapter.model_info("unknown-model"), {})


class TestEnhancedEngineFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "engines.toml"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_default_config(self) -> None:
        factory = EnhancedEngineFactory()
        self.assertEqual(factory.config["default_engine"], "litellm")
        self.assertEqual(factory.config["litellm"]["base_url"], "http://localhost:4000")
        self.assertEqual(factory.config["litellm"]["model"], "gpt-3.5-turbo")

    def test_init_with_custom_config(self) -> None:
        self.config_path.write_text(
            '[default]\n'
            'default_engine = "ollama"\n\n'
            '[ollama]\n'
            'host = "localhost"\n'
            'port = 11434\n'
            'model = "qwen2.5:7b-instruct"\n'
            'timeout = 60\n',
            encoding="utf-8",
        )
        factory = EnhancedEngineFactory(str(self.config_path))
        self.assertEqual(factory.config["default_engine"], "ollama")
        self.assertEqual(factory.config["ollama"]["host"], "localhost")
        self.assertEqual(factory.config["ollama"]["port"], 11434)
        self.assertEqual(factory.config["ollama"]["model"], "qwen2.5:7b-instruct")

    def test_register_engine(self) -> None:
        factory = EnhancedEngineFactory()

        class MockEngine:
            def __init__(self, config):
                self.config = config

        factory.register_engine("mock", MockEngine)
        self.assertIn("mock", factory.get_available_engines())

    def test_get_available_engines(self) -> None:
        engines = EnhancedEngineFactory().get_available_engines()
        for name in ("ollama", "openai", "optimized_ollama", "litellm"):
            self.assertIn(name, engines)

    def test_get_engine_info(self) -> None:
        factory = EnhancedEngineFactory()
        info = factory.get_engine_info("litellm")
        self.assertEqual(info["name"], "litellm")
        self.assertIn("config", info)
        self.assertEqual(factory.get_engine_info("unknown"), {})


class TestIntegrationFunctions(unittest.TestCase):
    def test_get_available_engines(self) -> None:
        engines = get_available_engines()
        for name in ("ollama", "openai", "optimized_ollama", "litellm"):
            self.assertIn(name, engines)

    def test_get_engine_info(self) -> None:
        self.assertEqual(get_engine_info("litellm")["name"], "litellm")
        self.assertEqual(get_engine_info("unknown"), {})


class TestRealLiteLLMIntegration(unittest.TestCase):
    @pytest.mark.skipif(
        not os.environ.get("LITELLM_MASTER_KEY"), reason="LITELLM_MASTER_KEY не установлен"
    )
    @pytest.mark.skipif(
        not os.environ.get("RUN_LITELLM_TESTS"), reason="RUN_LITELLM_TESTS не установлен"
    )
    def test_real_litellm_connection(self) -> None:
        adapter = LiteLLMAdapter(
            {
                "base_url": "http://localhost:4000",
                "model": "gpt-3.5-turbo",
                "timeout": 60,
                "max_retries": 3,
                "fallback_models": ["gpt-3.5-turbo"],
                "master_key": os.environ.get("LITELLM_MASTER_KEY"),
            }
        )
        self.assertTrue(adapter.is_available(), "LiteLLM должен быть доступен")
        self.assertGreater(len(adapter.list_models()), 0)
        response = adapter.generate("Hello, how are you?", max_tokens=10)
        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)


if __name__ == "__main__":
    unittest.main()
