"""
Pytest-гейт финальной проверки Спринта 10 (замена os.path.exists-финалчека).

Гейт = реальная функциональность + прогон интеграционных тестов + живой ping
LiteLLM с auth (skip, если недоступен). Проверка интеграции с валидатором —
под РЕАЛЬНЫЙ дизайн validate_sprint.SUITES (Dict[str, List[str]]).
"""
import os
import subprocess
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.engines.enhanced_factory import (  # noqa: E402
    EnhancedEngineFactory,
    LiteLLMAdapter,
    get_available_engines,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSprint10Gate:
    def test_external_api_integration_files(self) -> None:
        for rel in (
            "openmanus_rl/engines/enhanced_factory.py",
            "config/litellm_config.yaml",
            "config/engines.toml",
            "tests/integration/test_external_api_integration.py",
        ):
            assert os.path.exists(os.path.join(REPO_ROOT, rel)), f"Файл {rel} отсутствует"

    def test_enhanced_factory_functionality(self) -> None:
        factory = EnhancedEngineFactory()
        assert hasattr(factory, "_engines")
        assert hasattr(factory, "config")
        for method in ("register_engine", "create_engine", "get_available_engines", "get_engine_info"):
            assert hasattr(factory, method), f"метод {method} отсутствует"
        engines = get_available_engines()
        for name in ("ollama", "openai", "optimized_ollama", "litellm"):
            assert name in engines, f"движок {name} недоступен"

    def test_litellm_adapter_functionality(self) -> None:
        from unittest.mock import patch

        config = {
            "base_url": "http://localhost:4000",
            "model": "gpt-3.5-turbo",
            "timeout": 60,
            "max_retries": 3,
            "fallback_models": ["gpt-3.5-turbo"],
        }
        with patch.object(LiteLLMAdapter, "_check_litellm_availability"):
            adapter = LiteLLMAdapter(config)
            for attr in ("base_url", "model", "metrics"):
                assert hasattr(adapter, attr)
            for method in ("generate", "chat", "get_metrics", "list_models", "model_info", "is_available"):
                assert hasattr(adapter, method), f"метод {method} отсутствует"

    def test_config_files(self) -> None:
        litellm_yaml = os.path.join(REPO_ROOT, "config/litellm_config.yaml")
        content = open(litellm_yaml, encoding="utf-8").read()
        assert "model_list:" in content
        assert "router_settings:" in content
        # Секрет — только плейсхолдер, не реальный ключ.
        assert "${LITELLM_MASTER_KEY}" in content

        engines_toml = os.path.join(REPO_ROOT, "config/engines.toml")
        content = open(engines_toml, encoding="utf-8").read()
        assert "[default]" in content
        assert "[litellm]" in content

    def test_no_hardcoded_secret_in_config(self) -> None:
        # Явная проверка отсутствия захардкоженного ключа sk-... в конфиге.
        content = open(os.path.join(REPO_ROOT, "config/litellm_config.yaml"), encoding="utf-8").read()
        assert "sk-1234567890abcdef" not in content
        assert 'master_key: "sk-' not in content

    def test_external_api_integration_tests(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/integration/test_external_api_integration.py", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Интеграционные тесты не пройдены:\n{result.stdout}\n{result.stderr}"

    def test_litellm_availability_with_auth(self) -> None:
        try:
            master_key = os.environ.get("LITELLM_MASTER_KEY", "")
            headers = {"Authorization": f"Bearer {master_key}"} if master_key else {}
            response = requests.get("http://localhost:4000/health", headers=headers, timeout=5)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"LiteLLM недоступен: {exc}")
        # 200 или 401 = сервис поднят.
        assert response.status_code in (200, 401), f"LiteLLM недоступен: {response.status_code}"

    def test_validator_integration(self) -> None:
        # Реальный дизайн: группа external-api зарегистрирована в SUITES (Dict[str, List[str]]).
        from scripts.validate_sprint import SUITES

        assert "external-api" in SUITES, "группа external-api отсутствует в SUITES"
        assert "tests/integration/test_external_api_integration.py" in SUITES["external-api"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
