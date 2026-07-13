"""Unit-тесты для openmanus_rl.config.load_config (Sprint 6 P0)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.config import load_config


class TestLoadConfig(unittest.TestCase):
    def test_returns_dict(self) -> None:
        cfg = load_config("production")
        self.assertIsInstance(cfg, dict)
        self.assertEqual(cfg["environment"], "production")

    def test_environment_names(self) -> None:
        for env in ("production", "development", "testing"):
            self.assertEqual(load_config(env)["environment"], env)

    def test_path_form(self) -> None:
        self.assertEqual(load_config("config/testing.py")["environment"], "testing")

    def test_unknown_falls_back_to_production(self) -> None:
        self.assertEqual(load_config("nonsense")["environment"], "production")

    def test_agent_compatible_keys(self) -> None:
        cfg = load_config("testing")
        for key in ("enable_mean_field_games", "enable_performance_optimization",
                    "qwen3_omni", "voice_pipeline", "mean_field_games"):
            self.assertIn(key, cfg)
        self.assertTrue(cfg["qwen3_omni"]["sandbox_mode"])

    def test_security_defaults(self) -> None:
        cfg = load_config("production")
        self.assertEqual(cfg["host"], "127.0.0.1")           # не 0.0.0.0
        self.assertFalse(cfg["gradio_share"])                # без публичного туннеля
        self.assertNotIn("*", cfg["cors_allow_origins"])     # CORS не wildcard

    def test_secret_from_env_only(self) -> None:
        os.environ.pop("OPENMANUS_SECRET_KEY", None)
        self.assertIsNone(load_config("production")["secret_key"])  # нет небезопасного дефолта
        os.environ["OPENMANUS_SECRET_KEY"] = "test-secret"
        try:
            self.assertEqual(load_config("production")["secret_key"], "test-secret")
        finally:
            os.environ.pop("OPENMANUS_SECRET_KEY", None)

    def test_env_override_port_cors(self) -> None:
        os.environ["OPENMANUS_PORT"] = "9999"
        os.environ["OPENMANUS_CORS_ORIGINS"] = "https://a.example, https://b.example"
        try:
            cfg = load_config("production")
            self.assertEqual(cfg["port"], 9999)
            self.assertEqual(cfg["cors_allow_origins"], ["https://a.example", "https://b.example"])
        finally:
            os.environ.pop("OPENMANUS_PORT", None)
            os.environ.pop("OPENMANUS_CORS_ORIGINS", None)


if __name__ == "__main__":
    unittest.main()
