"""
Smoke-тест UI OpenManus — реальное ядро (select_action) + security-инварианты UI-файлов.

Ядро (ui.core.run_query) тестируется headless (без запуска streamlit/gradio серверов).
Дополнительно статически проверяем красную линию: gradio share=False, никакого process_input.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui import core as ui_core  # noqa: E402

UI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ui")


class TestUiSmoke(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        ui_core.shutdown()  # освободить monitoring-поток агента

    def test_run_query_real_select_action(self) -> None:
        result = ui_core.run_query("What is the risk of BTC?", ["buy", "sell", "wait"])
        for key in ("action", "confidence", "osint_enhanced", "episode_id", "timestamp"):
            self.assertIn(key, result)
        self.assertIn(result["action"], ["buy", "sell", "wait", "error"])

    def test_gradio_build_returns_blocks(self) -> None:
        import gradio as gr

        from ui.gradio_app import create_ui
        demo = create_ui()
        self.assertIsInstance(demo, gr.Blocks)

    def test_streamlit_module_imports(self) -> None:
        # import не должен запускать UI (main вызывается только под __main__)
        import ui.streamlit_app  # noqa: F401

    def _read(self, name: str) -> str:
        with open(os.path.join(UI_DIR, name), encoding="utf-8") as fh:
            return fh.read()

    def test_security_invariants_static(self) -> None:
        gradio_src = self._read("gradio_app.py")
        self.assertIn("share=False", gradio_src)
        self.assertNotIn("share=True", gradio_src)
        self.assertIn('server_name="127.0.0.1"', gradio_src)
        # красная линия фиктивного API — ни в одном UI-файле
        for name in ("gradio_app.py", "streamlit_app.py", "core.py"):
            self.assertNotIn("process_input", self._read(name))


if __name__ == "__main__":
    unittest.main()
