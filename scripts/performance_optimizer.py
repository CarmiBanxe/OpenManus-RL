#!/usr/bin/env python3
"""
Оптимизатор производительности OpenManus для архитектуры Legion (RTX 4070 8GB, 64GB RAM).

Детектит реальные ресурсы (nvidia-smi + psutil), считает рекомендации по распределению
моделей/Ollama-слоёв/rollout-воркеров/мониторинга. Пишет config/performance.toml.
pynvml опционален (guarded); детект GPU и без него работает через nvidia-smi.
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

import psutil


@dataclass
class SystemResources:
    cpu_count: int
    cpu_freq: float
    memory_total: int
    memory_available: int
    gpu_available: bool
    gpu_name: str
    gpu_memory_total: int  # bytes
    gpu_memory_free: int   # bytes
    gpu_utilization: float


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "model_distribution": {"large_models_gpu": True, "medium_models_threshold": 4_000_000_000,
                           "small_models_cpu": True, "ollama_gpu_layers": 35},
    "ollama": {"host": "127.0.0.1", "port": 11434, "model": "qwen2.5:7b-instruct",
               "timeout": 60, "max_concurrent_requests": 2, "gpu_layers": 0},
    "rollout": {"max_workers": 4, "chunk_size": 10, "preload_environments": True,
                "cache_environment_state": True},
    "monitoring": {"enabled": True, "interval": 5, "history_size": 1000},
}


def detect_system_resources() -> SystemResources:
    cpu_count = mp.cpu_count()
    freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    gpu_available, gpu_name, gpu_total, gpu_free, gpu_util = False, "none", 0, 0, 0.0
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            name, total, free, util = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            gpu_available, gpu_name = True, name
            gpu_total, gpu_free = int(total) * 1024 * 1024, int(free) * 1024 * 1024
            gpu_util = float(util)
    except Exception:  # noqa: BLE001
        pass
    return SystemResources(cpu_count, freq.current if freq else 0.0, mem.total, mem.available,
                           gpu_available, gpu_name, gpu_total, gpu_free, gpu_util)


class PerformanceOptimizer:
    def __init__(self, config_path: str = None) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = Path(config_path) if config_path else self.project_root / "config" / "performance.toml"
        self.system_resources = detect_system_resources()
        self.config = self._load_config()
        self.optimization_strategies: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
        if self.config_path.exists() and tomllib is not None:
            try:
                loaded = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
                for section, values in loaded.items():
                    cfg.setdefault(section, {}).update(values)
            except Exception as exc:  # noqa: BLE001
                print(f"config load error ({exc}); using defaults")
        return cfg

    def _save_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# OpenManus performance config (generated)\n"]
        for section, values in self.config.items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        self.config_path.write_text("\n".join(lines), encoding="utf-8")

    def optimize_model_distribution(self) -> Dict[str, Any]:
        r, thr = self.system_resources, self.config["model_distribution"]["medium_models_threshold"]
        recs, layers = [], 0
        if r.gpu_available:
            free_mb = r.gpu_memory_free // (1024 * 1024)
            layers = min(self.config["model_distribution"]["ollama_gpu_layers"], free_mb // 250)
            recs.append(f"GPU {r.gpu_name}: {free_mb}MB free -> {layers} GPU layers for 7B (~{layers*250}MB)")
        else:
            recs.append("No GPU -> CPU only")
        return {"large_models_gpu": r.gpu_available and r.gpu_memory_free > thr,
                "ollama_gpu_layers": layers, "recommendations": recs}

    def optimize_ollama_config(self) -> Dict[str, Any]:
        r = self.system_resources
        layers = min(35, (r.gpu_memory_free // (1024 * 1024)) // 250) if r.gpu_available else 0
        concurrent = min(4, max(1, r.cpu_count // 2)) if r.gpu_available else min(2, max(1, r.cpu_count // 4))
        return {"gpu_layers": layers, "max_concurrent_requests": concurrent,
                "recommendations": [f"gpu_layers={layers}", f"max_concurrent_requests={concurrent}"]}

    def optimize_rollout_config(self) -> Dict[str, Any]:
        r = self.system_resources
        workers = min(8, r.cpu_count) if r.gpu_available else min(4, max(2, r.cpu_count // 2))
        gb = r.memory_available // (1024 ** 3)
        chunk = 20 if gb > 32 else (10 if gb > 16 else 5)
        return {"max_workers": workers, "chunk_size": chunk,
                "recommendations": [f"max_workers={workers}", f"chunk_size={chunk} ({gb}GB free)"]}

    def optimize_monitoring_config(self) -> Dict[str, Any]:
        r = self.system_resources
        gb = r.memory_available // (1024 ** 3)
        return {"interval": 5 if r.gpu_available else 10,
                "history_size": 2000 if gb > 32 else (1000 if gb > 16 else 500),
                "recommendations": ["monitoring tuned to resources"]}

    def optimize(self) -> Dict[str, Any]:
        self.optimization_strategies = {
            "model_distribution": self.optimize_model_distribution(),
            "ollama": self.optimize_ollama_config(),
            "rollout": self.optimize_rollout_config(),
            "monitoring": self.optimize_monitoring_config(),
        }
        self.config["model_distribution"]["ollama_gpu_layers"] = self.optimization_strategies["model_distribution"]["ollama_gpu_layers"]
        self.config["ollama"].update({k: self.optimization_strategies["ollama"][k] for k in ("gpu_layers", "max_concurrent_requests")})
        self.config["rollout"].update({k: self.optimization_strategies["rollout"][k] for k in ("max_workers", "chunk_size")})
        self.config["monitoring"].update({k: self.optimization_strategies["monitoring"][k] for k in ("interval", "history_size")})
        self._save_config()
        return self.optimization_strategies


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenManus performance optimizer")
    ap.add_argument("--config", default=None)
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()
    opt = PerformanceOptimizer(args.config)
    r = opt.system_resources
    print(f"CPU: {r.cpu_count} cores @ {r.cpu_freq:.0f}MHz | RAM: {r.memory_total//(1024**3)}GB "
          f"({r.memory_available//(1024**3)}GB free)")
    print(f"GPU: {r.gpu_name} {r.gpu_memory_total//(1024*1024)}MB ({r.gpu_memory_free//(1024*1024)}MB free)"
          if r.gpu_available else "GPU: none")
    if not args.detect_only:
        for comp, strat in opt.optimize().items():
            print(f"[{comp}] " + "; ".join(strat["recommendations"]))
        print(f"saved -> {opt.config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
