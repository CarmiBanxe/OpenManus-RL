"""
Интеграционный тест реального perf-валидатора — in-process (без сервера, без /query/public).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.performance_validator import PerformanceValidator


class TestPerformanceValidator(unittest.TestCase):
    def test_validate_real_inprocess(self) -> None:
        v = PerformanceValidator()
        ok = v.validate()
        self.assertTrue(ok, f"errors={v.errors}")
        # реальные метрики собраны
        for key in ("agent_build_s", "inference_s", "api_query_s", "disk_percent"):
            self.assertIn(key, v.metrics)
            self.assertGreaterEqual(v.metrics[key], 0.0)

    def test_no_forbidden_public_endpoint(self) -> None:
        # Красная линия: perf-валидатор не бьёт публичный роут /query/public.
        # Проверяем КОДОВУЮ конструкцию (quoted route literal), не прозу докстринга.
        src = open(os.path.join(os.path.dirname(__file__), "..", "..",
                                "scripts", "performance_validator.py"), encoding="utf-8").read()
        self.assertNotIn('"/query/public"', src)   # нет вызова публичного роута
        self.assertIn('"/query"', src)             # использует аутентифицированный /query


if __name__ == "__main__":
    unittest.main()
