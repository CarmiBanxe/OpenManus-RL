#!/usr/bin/env python3
"""
Реальный perf-валидатор OpenManus — IN-PROCESS, без живого сервера и без /query/public.

Меряет:
  - время построения агента и реального select_action (in-process);
  - латентность аутентифицированного /query через FastAPI TestClient (без сети/сервера);
  - память процесса (psutil) и диск (shutil).
Падает (exit!=0) только при реальной ОШИБКЕ (не при «медленно» — это warning).
"""
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.agents.enhanced_decision_agent import EnhancedDecisionAgent
from openmanus_rl.config import load_config

THRESHOLDS = {
    "agent_build_s": 30.0, "inference_s": 15.0, "api_query_s": 20.0,
    "memory_mb": 8192.0, "disk_percent": 95.0,
}


class PerformanceValidator:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.metrics: Dict[str, float] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def _warn_if_slow(self, key: str) -> None:
        if key in self.metrics and self.metrics[key] > THRESHOLDS[key]:
            self.warnings.append(f"{key} slow: {self.metrics[key]:.2f} > {THRESHOLDS[key]}")

    def check_agent_and_inference(self) -> bool:
        try:
            t0 = time.time()
            agent = EnhancedDecisionAgent(config=load_config("testing"))
            self.metrics["agent_build_s"] = time.time() - t0
            try:
                t1 = time.time()
                result = asyncio.run(agent.select_action({"text": "perf probe"}, ["proceed", "wait"]))
                self.metrics["inference_s"] = time.time() - t1
                if "action" not in result:
                    self.errors.append("select_action returned no 'action'")
                    return False
            finally:
                asyncio.run(agent.cleanup())
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"agent/inference error: {exc}")
            return False
        self._warn_if_slow("agent_build_s")
        self._warn_if_slow("inference_s")
        return True

    def check_api_latency(self) -> bool:
        os.environ["OPENMANUS_CONFIG_FILE"] = "testing"
        os.environ["OPENMANUS_SECRET_KEY"] = "perf-secret-key-at-least-32-bytes-long!!"
        os.environ["OPENMANUS_ADMIN_USER"] = "perfadmin"
        os.environ["OPENMANUS_ADMIN_PASSWORD"] = "perfpass"
        try:
            from fastapi.testclient import TestClient

            from openmanus_rl.api.server import app
            with TestClient(app) as client:
                login = client.post("/auth/login", json={"username": "perfadmin", "password": "perfpass"})
                if login.status_code != 200:
                    self.errors.append(f"/auth/login status {login.status_code}")
                    return False
                token = login.json()["access_token"]
                t0 = time.time()
                r = client.post(
                    "/query",
                    json={"text": "perf", "available_actions": ["buy", "sell", "wait"]},
                    headers={"Authorization": f"Bearer {token}"},
                )
                self.metrics["api_query_s"] = time.time() - t0
                if r.status_code != 200:
                    self.errors.append(f"/query status {r.status_code}")
                    return False
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"api latency error: {exc}")
            return False
        self._warn_if_slow("api_query_s")
        return True

    def check_system(self) -> bool:
        try:
            import psutil
            self.metrics["memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
            self._warn_if_slow("memory_mb")
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"psutil unavailable: {exc}")
        total, used, _ = shutil.disk_usage("/")
        self.metrics["disk_percent"] = used / total * 100
        self._warn_if_slow("disk_percent")
        return True

    def validate(self) -> bool:
        ok = self.check_system()
        ok = self.check_agent_and_inference() and ok
        ok = self.check_api_latency() and ok
        ok = ok and not self.errors

        print("\n# Performance report")
        for k, v in self.metrics.items():
            unit = "s" if k.endswith("_s") else ("MB" if k.endswith("_mb") else "%")
            print(f"- {k}: {v:.3f}{unit}")
        for w in self.warnings:
            print(f"⚠️ {w}")
        for e in self.errors:
            print(f"❌ {e}")
        print("✅ PERFORMANCE OK" if ok else "❌ PERFORMANCE FAILED")
        return ok


if __name__ == "__main__":
    sys.exit(0 if PerformanceValidator().validate() else 1)
