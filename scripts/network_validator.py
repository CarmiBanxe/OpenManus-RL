#!/usr/bin/env python3
"""
Network-валидатор OpenManus (приватный Legion-контур, S-18 §1.2 — защитный).

Проверяет СЕТЕВУЮ ИЗОЛЯЦИЮ и безопасность (основное, defensive):
  - публичные слушатели (0.0.0.0/::) среди наших портов -> нарушение (должно быть 127.0.0.1);
  - egress-конфиг указывает только на localhost/allowlist (нет утечки к banking/внешним хостам);
  - доступность Tor SOCKS (:9050) как ОПЦИЯ приватности (без abuse exit-нод; graceful-skip).

Дизайн: чистые функции (classify_listeners/scan_config_egress/tor_available) — тестируемы
детерминированно; NetworkValidator.validate() запускает live-проверки и репортит.
"""
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

PUBLIC_BINDS = {"0.0.0.0", "::", "*", "[::]"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
WATCHED_PORTS: Set[int] = {8000, 7860, 4000, 8080, 8081, 11434, 9090, 3000}


def classify_listeners(ss_output: str, watched: Set[int] = WATCHED_PORTS) -> List[Dict]:
    """Из `ss -tlnH` вернуть публично-открытые слушатели среди watched-портов."""
    findings: List[Dict] = []
    for line in ss_output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]
        m = re.search(r":(\d+)$", local)
        if not m:
            continue
        port = int(m.group(1))
        if watched and port not in watched:
            continue
        host = local.rsplit(":", 1)[0]
        if host in PUBLIC_BINDS:
            findings.append({"port": port, "bind": host})
    return findings


def scan_config_egress(config_text: str) -> List[str]:
    """Вернуть НЕ-localhost хосты из base_url/host (потенциальная утечка изоляции)."""
    hosts: List[str] = []
    for m in re.finditer(r'(?:base_url|host)\s*=\s*["\']?\s*https?://([^/"\'\s]+)', config_text):
        host = m.group(1).rsplit(":", 1)[0] if ":" in m.group(1) else m.group(1)
        if host not in LOCAL_HOSTS:
            hosts.append(host)
    return hosts


def tor_available(host: str = "127.0.0.1", port: int = 9050, timeout: float = 1.0) -> bool:
    """Доступен ли Tor SOCKS (только проверка соединения, без запросов через exit)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class NetworkValidator:
    def __init__(self, root: Path = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent
        self.violations: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def check_listeners(self) -> None:
        try:
            out = subprocess.run(["ss", "-tlnH"], capture_output=True, text=True, timeout=5).stdout
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"ss unavailable: {exc}")
            return
        for f in classify_listeners(out):
            self.violations.append(f"public listener {f['bind']}:{f['port']} — should bind 127.0.0.1")

    def check_config_egress(self) -> None:
        cfg = self.root / "config" / "config.toml"
        if not cfg.exists():
            return
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        for host in scan_config_egress(text):
            self.warnings.append(f"egress to non-local host in config: {host}")
        if "0.0.0.0" in text:
            self.warnings.append("config mentions 0.0.0.0 bind (verify it's localhost-only in prod)")

    def check_tor(self) -> None:
        self.info.append(f"tor_socks_9050 = {'up' if tor_available() else 'down (privacy proxy optional)'}")

    def validate(self) -> bool:
        self.check_listeners()
        self.check_config_egress()
        self.check_tor()
        print("\n# Network posture (private Legion contour)")
        for i in self.info:
            print(f"  ℹ {i}")
        for w in self.warnings:
            print(f"  ⚠ {w}")
        for v in self.violations:
            print(f"  ❌ {v}")
        ok = not self.violations
        print("✅ NETWORK ISOLATION OK" if ok else "❌ NETWORK ISOLATION VIOLATIONS (public listeners)")
        return ok


if __name__ == "__main__":
    sys.exit(0 if NetworkValidator().validate() else 1)
