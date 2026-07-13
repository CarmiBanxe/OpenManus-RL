"""
Реальная валидация Sphinx-доков: sphinx-build должен успешно собраться,
примеры — под реальный API (select_action, не process_input).
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPHINX_SRC = ROOT / "docs" / "sphinx"


class TestDocsBuild(unittest.TestCase):
    def test_sphinx_build_succeeds(self) -> None:
        out = SPHINX_SRC / "_build" / "html_test"   # под _build/ (gitignored)
        r = subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", str(SPHINX_SRC), str(out)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue((out / "index.html").exists())
        self.assertTrue((out / "api.html").exists())
        self.assertTrue((out / "quickstart.html").exists())

    def test_quickstart_uses_real_api(self) -> None:
        qs = (SPHINX_SRC / "quickstart.rst").read_text(encoding="utf-8")
        self.assertIn("select_action", qs)
        self.assertNotIn("process_input", qs)


if __name__ == "__main__":
    unittest.main()
