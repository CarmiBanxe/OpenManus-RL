"""Тесты для scripts.security_validator (реальный repo-позитив + temp-негатив)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.security_validator import SecurityValidator, scan


class TestScanHelper(unittest.TestCase):
    def _write(self, content: str) -> Path:
        d = tempfile.mkdtemp()
        p = Path(d) / "f.py"
        p.write_text(content, encoding="utf-8")
        return p

    def test_forbidden_detected(self) -> None:
        p = self._write("demo.launch(share=True)")
        self.assertTrue(scan(p, forbidden=["share=True"], required=[]))

    def test_required_missing_detected(self) -> None:
        p = self._write("demo.launch()")
        self.assertTrue(scan(p, forbidden=[], required=["share=False"]))

    def test_clean_file_no_violations(self) -> None:
        p = self._write('demo.launch(share=False, server_name="127.0.0.1")')
        self.assertEqual(scan(p, forbidden=["share=True"], required=["share=False"]), [])

    def test_missing_file(self) -> None:
        self.assertTrue(scan(Path("/nonexistent/f.py"), [], []))


class TestSecurityValidatorRealRepo(unittest.TestCase):
    def test_real_repo_passes(self) -> None:
        # доказывает, что реальные файлы соблюдают security-инварианты
        self.assertTrue(SecurityValidator().validate())


class TestSecurityValidatorNegative(unittest.TestCase):
    def test_bad_gradio_share_true_fails(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "ui").mkdir()
        (root / "openmanus_rl" / "api").mkdir(parents=True)
        (root / "openmanus_rl").mkdir(exist_ok=True)
        # плохой gradio: share=True
        (root / "ui" / "gradio_app.py").write_text(
            'demo.launch(share=True, server_name="127.0.0.1")', encoding="utf-8")
        (root / "ui" / "streamlit_app.py").write_text("import streamlit", encoding="utf-8")
        (root / "openmanus_rl" / "api" / "server.py").write_text(
            'os.environ.get("OPENMANUS_SECRET_KEY")\nget_current_user', encoding="utf-8")
        (root / "openmanus_rl" / "config.py").write_text('"127.0.0.1"', encoding="utf-8")
        (root / "docker-compose.yml").write_text(
            "services:\n  a:\n    ports:\n      - '127.0.0.1:8000:8000'\n", encoding="utf-8")
        v = SecurityValidator(project_root=root)
        self.assertFalse(v.validate())
        self.assertTrue(any("share=True" in x for x in v.violations))


if __name__ == "__main__":
    unittest.main()
