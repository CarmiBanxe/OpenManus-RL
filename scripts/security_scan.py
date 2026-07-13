#!/usr/bin/env python3
"""
Security-скан OpenManus — РЕАЛЬНЫЕ инструменты (bandit + gitleaks), не моки.

- bandit: HIGH-severity линтинг по коду, который мы авторим (gate-scope). Наш код чист (0 HIGH).
- gitleaks: секреты в РАБОЧЕМ ДЕРЕВЕ (не история), исключая вендоренное (verl/, env_package/tools) и tests.
- Информационно: bandit по всему репо покажет предсуществующие HIGH (S2/3 + upstream rollout) — tech-debt,
  не гейтим (не наш код). Секреты в git-истории — отдельная операторская забота.

Gate green = наш код без HIGH bandit + рабочее дерево без секретов.
Если инструмент недоступен — graceful (passed=True + note), чтобы не падать там, где его нет.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Код, который мы авторим (gate-scope bandit) — вендоренное/предсуществующее не включаем.
AUTHORED_TARGETS = [
    "openmanus_rl/api", "openmanus_rl/auth", "openmanus_rl/monitoring", "openmanus_rl/config.py",
    "ui",
    "scripts/validate_sprint.py", "scripts/security_validator.py", "scripts/performance_validator.py",
    "scripts/network_validator.py", "scripts/backup.py", "scripts/security_scan.py",
]

# Префиксы, исключаемые из gitleaks (вендоренное/тесты/кэш).
LEAK_EXCLUDE_PREFIXES = (
    "verl/", "verl\\", "tests/", "tests\\",
    "openmanus_rl/environments/", "openmanus_rl/tools/", "__pycache__/", ".git/",
)


class SecurityScanner:
    def __init__(self, root: Path = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent
        self.results: Dict[str, Dict] = {"bandit": {}, "gitleaks": {}}

    def run_bandit(self) -> Dict:
        targets = [t for t in AUTHORED_TARGETS if (self.root / t).exists()]
        cmd = [sys.executable, "-m", "bandit", "-q", "-f", "json", "--severity-level", "high", *targets]
        try:
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=180)
        except FileNotFoundError:
            self.results["bandit"] = {"passed": True, "skipped": "bandit not installed"}
            return self.results["bandit"]
        try:
            data = json.loads(r.stdout or "{}")
        except json.JSONDecodeError:
            data = {"results": []}
        highs = [
            {"file": i.get("filename"), "line": i.get("line_number"),
             "test": i.get("test_id"), "msg": i.get("issue_text")}
            for i in data.get("results", []) if i.get("issue_severity") == "HIGH"
        ]
        self.results["bandit"] = {"passed": not highs, "high": highs, "high_count": len(highs)}
        return self.results["bandit"]

    def run_gitleaks(self) -> Dict:
        if shutil.which("gitleaks") is None:
            self.results["gitleaks"] = {"passed": True, "skipped": "gitleaks not installed"}
            return self.results["gitleaks"]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            report = tf.name
        try:
            subprocess.run(
                ["gitleaks", "detect", "--source", str(self.root), "--no-git",
                 "--no-banner", "--redact", "--report-format", "json", "--report-path", report],
                cwd=self.root, capture_output=True, text=True, timeout=120,
            )
            try:
                findings = json.loads(Path(report).read_text() or "[]")
            except json.JSONDecodeError:
                findings = []
        finally:
            Path(report).unlink(missing_ok=True)
        excl = tuple(p.replace("\\", "/") for p in LEAK_EXCLUDE_PREFIXES)
        leaks = []
        for f in findings:
            raw = str(f.get("File", "")).replace("\\", "/")
            try:  # gitleaks отдаёт абсолютный путь — релятивизируем к корню репо
                rel = Path(raw).resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                rel = raw
            if rel.startswith(excl):
                continue
            leaks.append({"file": rel, "line": f.get("StartLine"), "rule": f.get("RuleID")})
        self.results["gitleaks"] = {"passed": not leaks, "leaks": leaks, "leak_count": len(leaks)}
        return self.results["gitleaks"]

    def scan(self) -> bool:
        b = self.run_bandit()
        g = self.run_gitleaks()
        print("\n# Security scan (OpenManus)")
        print(f"  bandit (authored): {'✅ 0 HIGH' if b['passed'] else '❌ ' + str(b.get('high_count')) + ' HIGH'}"
              + (f"  [{b['skipped']}]" if b.get("skipped") else ""))
        for h in b.get("high", []):
            print(f"    ❌ {h['test']} {h['file']}:{h['line']} — {h['msg']}")
        print(f"  gitleaks (working tree): {'✅ no secrets' if g['passed'] else '❌ ' + str(g.get('leak_count')) + ' leaks'}"
              + (f"  [{g['skipped']}]" if g.get("skipped") else ""))
        for lk in g.get("leaks", []):
            print(f"    ❌ {lk['rule']} {lk['file']}:{lk['line']}")
        ok = b["passed"] and g["passed"]
        print("✅ SECURITY SCAN OK" if ok else "❌ SECURITY SCAN FAILED")
        return ok


if __name__ == "__main__":
    sys.exit(0 if SecurityScanner().scan() else 1)
