#!/usr/bin/env python3
"""
Реальный backup-скрипт OpenManus — работает с текущим состоянием проекта.

Бэкапит: config/ (включая config.toml), верхнеуровневые артефакты (requirements/pyproject/
setup/Docker), .github/, любые *.db/*.sqlite (если появятся). data/ (47MB parquet) — опционально.
Проверяет целостность через sha256 + чтение tar. Пишет манифест.

SECURITY (приватный Legion-контур): бэкап пишется ЛОКАЛЬНО (dest по умолчанию ./backups или
env OPENMANUS_BACKUP_DIR), не в /var/... (не требует root) и НЕ коммитится в git. Может содержать
config.toml (конфиг uncensored-движка) — держать локально, не публиковать.
"""
import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DEFAULT_TARGETS = [
    "config",
    "requirements-legion.txt",
    "pyproject.toml",
    "setup.py",
    "Dockerfile",
    "docker-compose.yml",
    ".github",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _exclude(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    name = tarinfo.name
    if "__pycache__" in name or name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def _collect(root: Path, targets: List[str], include_data: bool) -> List[Path]:
    found = [root / t for t in targets if (root / t).exists()]
    if include_data and (root / "data").exists():
        found.append(root / "data")
    found += sorted(root.glob("*.db")) + sorted(root.glob("*.sqlite"))
    return found


def create_backup(root: Path, dest_dir: Path, timestamp: str,
                  targets: List[str] = None, include_data: bool = False) -> Path:
    targets = DEFAULT_TARGETS if targets is None else targets
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"openmanus_backup_{timestamp}.tar.gz"
    members: List[str] = []
    with tarfile.open(archive, "w:gz") as tar:
        for path in _collect(root, targets, include_data):
            arcname = path.relative_to(root).as_posix()
            tar.add(path, arcname=arcname, filter=_exclude)
            members.append(arcname)
    sha = _sha256(archive)
    (dest_dir / f"{archive.name}.sha256").write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    manifest = {"archive": archive.name, "sha256": sha, "timestamp": timestamp, "targets": members}
    (dest_dir / f"{archive.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return archive


def verify_backup(archive: Path) -> bool:
    archive = Path(archive)
    sha_file = archive.with_name(f"{archive.name}.sha256")
    if not archive.exists() or not sha_file.exists():
        return False
    expected = sha_file.read_text(encoding="utf-8").split()[0]
    if _sha256(archive) != expected:
        return False
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.getmembers()  # бросит при повреждении
    except (tarfile.TarError, OSError):
        return False
    return True


def restore_backup(archive: Path, dest: Path) -> bool:
    archive, dest = Path(archive), Path(dest)
    if not verify_backup(archive):
        return False
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")  # py3.12 secure extraction
    return True


def _default_dest() -> Path:
    import os
    return Path(os.environ.get("OPENMANUS_BACKUP_DIR", "backups"))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="OpenManus backup")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="создать бэкап")
    p_create.add_argument("--dest", default=None)
    p_create.add_argument("--include-data", action="store_true", help="включить data/ (47MB parquet)")

    p_verify = sub.add_parser("verify", help="проверить целостность")
    p_verify.add_argument("archive")

    p_restore = sub.add_parser("restore", help="восстановить")
    p_restore.add_argument("archive")
    p_restore.add_argument("dest")

    args = parser.parse_args()

    if args.cmd == "create":
        dest = Path(args.dest) if args.dest else _default_dest()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = create_backup(root, dest, ts, include_data=args.include_data)
        ok = verify_backup(archive)
        print(f"{'✅' if ok else '❌'} backup: {archive} (verified={ok})")
        return 0 if ok else 1
    if args.cmd == "verify":
        ok = verify_backup(Path(args.archive))
        print(f"{'✅ OK' if ok else '❌ CORRUPT/MISSING'}: {args.archive}")
        return 0 if ok else 1
    if args.cmd == "restore":
        ok = restore_backup(Path(args.archive), Path(args.dest))
        print(f"{'✅ restored' if ok else '❌ restore failed'}: {args.archive} -> {args.dest}")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
