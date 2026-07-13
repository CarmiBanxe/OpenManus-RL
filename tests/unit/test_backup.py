"""Тесты реального backup-скрипта (create / verify / restore / tamper)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.backup import create_backup, restore_backup, verify_backup


class TestBackup(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        # имитируем реальные цели
        (self.root / "config").mkdir()
        (self.root / "config" / "config.toml").write_text("[llm]\nmodel='x'\n", encoding="utf-8")
        (self.root / "config" / "__pycache__").mkdir()
        (self.root / "config" / "__pycache__" / "junk.pyc").write_text("junk", encoding="utf-8")
        (self.root / "requirements-legion.txt").write_text("torch>=2.0\n", encoding="utf-8")
        (self.root / "app.db").write_text("SQLITEDATA", encoding="utf-8")  # авто-детект БД
        self.dest = self.root / "backups"
        self.targets = ["config", "requirements-legion.txt"]

    def test_create_and_verify(self) -> None:
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        self.assertTrue(archive.exists())
        self.assertTrue((self.dest / f"{archive.name}.sha256").exists())
        self.assertTrue((self.dest / f"{archive.name}.manifest.json").exists())
        self.assertTrue(verify_backup(archive))

    def test_manifest_lists_targets_and_autodetects_db(self) -> None:
        import json
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        manifest = json.loads((self.dest / f"{archive.name}.manifest.json").read_text(encoding="utf-8"))
        self.assertIn("requirements-legion.txt", manifest["targets"])
        self.assertIn("app.db", manifest["targets"])  # авто-детект *.db

    def test_pycache_excluded(self) -> None:
        import tarfile
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        self.assertTrue(any("config.toml" in n for n in names))
        self.assertFalse(any("__pycache__" in n or n.endswith(".pyc") for n in names))

    def test_verify_detects_tamper(self) -> None:
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        archive.write_bytes(archive.read_bytes() + b"corruption")  # портим архив
        self.assertFalse(verify_backup(archive))

    def test_verify_missing(self) -> None:
        self.assertFalse(verify_backup(self.dest / "nope.tar.gz"))

    def test_restore_roundtrip(self) -> None:
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        out = self.root / "restored"
        self.assertTrue(restore_backup(archive, out))
        self.assertEqual((out / "config" / "config.toml").read_text(encoding="utf-8"),
                         "[llm]\nmodel='x'\n")

    def test_restore_refuses_corrupt(self) -> None:
        archive = create_backup(self.root, self.dest, "test", targets=self.targets)
        archive.write_bytes(b"broken")
        self.assertFalse(restore_backup(archive, self.root / "out2"))


if __name__ == "__main__":
    unittest.main()
