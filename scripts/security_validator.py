#!/usr/bin/env python3
"""
Валидатор security-инвариантов OpenManus (S-18 §1.2, приватный Legion-контур).

Проверяет РЕАЛЬНЫЕ файлы репозитория на:
  - отсутствие неаутентифицированного /query/public в API;
  - localhost-биндинг (config host=127.0.0.1, docker-порты 127.0.0.1);
  - gradio share=False (никогда share=True);
  - отсутствие фиктивного process_input в UI;
  - секрет из env (не хардкод).
Падает (exit!=0) при нарушении.
"""
import sys
from pathlib import Path
from typing import Dict, List


def scan(path: Path, forbidden: List[str], required: List[str]) -> List[str]:
    """Вернуть список нарушений: запрещённое присутствует / обязательное отсутствует."""
    if not path.exists():
        return [f"missing file: {path.name}"]
    text = path.read_text(encoding="utf-8")
    violations = [f"{path.name}: forbidden '{tok}'" for tok in forbidden if tok in text]
    violations += [f"{path.name}: missing '{tok}'" for tok in required if tok not in text]
    return violations


class SecurityValidator:
    def __init__(self, project_root: Path = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.violations: List[str] = []

    def _check(self, rel: str, forbidden: List[str], required: List[str]) -> None:
        self.violations += scan(self.project_root / rel, forbidden, required)

    def check_api(self) -> None:
        # Проверяем КОДОВЫЕ конструкции, не прозу докстрингов:
        #   .post("/query/public"  — реальная регистрация публичного роута;
        #   == "admin"             — хардкод админ-кред (antipattern server_with_auth).
        self._check(
            "openmanus_rl/api/server.py",
            forbidden=['.post("/query/public"', '== "admin"'],
            required=['os.environ.get("OPENMANUS_SECRET_KEY")', "get_current_user"],
        )

    def check_ui(self) -> None:
        self._check("ui/gradio_app.py",
                    forbidden=["share=True", "process_input("],
                    required=["share=False", 'server_name="127.0.0.1"'])
        self._check("ui/streamlit_app.py", forbidden=["process_input("], required=[])

    def check_docker(self) -> None:
        # все проброшенные порты должны быть на 127.0.0.1
        compose = self.project_root / "docker-compose.yml"
        if not compose.exists():
            self.violations.append("missing docker-compose.yml")
        else:
            import yaml
            data = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
            for name, svc in (data.get("services") or {}).items():
                for port in (svc or {}).get("ports", []):
                    if not str(port).startswith("127.0.0.1:"):
                        self.violations.append(f"docker-compose: {name} port not localhost: {port}")

    def check_config(self) -> None:
        # реальный ключ security-дефолта: host 127.0.0.1 в openmanus_rl/config.py
        self._check("openmanus_rl/config.py",
                    forbidden=['"host": "0.0.0.0"', "'host': '0.0.0.0'"],
                    required=['"127.0.0.1"'])

    def validate(self) -> bool:
        for check in (self.check_api, self.check_ui, self.check_docker, self.check_config):
            check()
        if self.violations:
            print("❌ SECURITY VIOLATIONS:")
            for v in self.violations:
                print(f"  - {v}")
            return False
        print("✅ SECURITY INVARIANTS OK (no /query/public, localhost bind, share=False, env-secret)")
        return True


if __name__ == "__main__":
    sys.exit(0 if SecurityValidator().validate() else 1)
