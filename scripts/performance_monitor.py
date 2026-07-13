#!/usr/bin/env python3
"""
Монитор производительности OpenManus — реальные метрики (psutil + опц. pynvml GPU).

Исправлено vs черновик: pynvml опционален (guarded); __init__ НЕ стартует поток и НЕ
пишет на диск (нужен явный start()); снапшоты на диск — только по флагу save_snapshots.
"""
import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # noqa: BLE001  # pragma: no cover
    _NVML = False


@dataclass
class PerformanceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    gpu_memory_percent: float
    gpu_memory_used: int
    gpu_utilization: float
    disk_usage: float
    network_sent: int
    network_recv: int
    processes: List[Dict[str, Any]] = field(default_factory=list)


class PerformanceMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.interval = int(self.config.get("interval", 5))
        self.history: deque = deque(maxlen=int(self.config.get("history_size", 1000)))
        self.save_snapshots = bool(self.config.get("save_snapshots", False))
        self.output_dir = Path(__file__).resolve().parent.parent / "performance_monitoring"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._gpu = None
        if _NVML:
            try:
                self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:  # noqa: BLE001
                self._gpu = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval * 2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self.take_snapshot()
                self.history.append(snap)
                if self.save_snapshots:
                    self.output_dir.mkdir(exist_ok=True)
                    (self.output_dir / f"snapshot_{int(snap.timestamp)}.json").write_text(
                        json.dumps(asdict(snap)), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def take_snapshot(self) -> PerformanceSnapshot:
        mem = psutil.virtual_memory()
        gpu_pct, gpu_used, gpu_util = 0.0, 0, 0.0
        if self._gpu is not None:
            try:
                mi = pynvml.nvmlDeviceGetMemoryInfo(self._gpu)
                gpu_used, gpu_pct = mi.used, (mi.used / mi.total * 100) if mi.total else 0.0
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
            except Exception:  # noqa: BLE001
                pass
        net = psutil.net_io_counters()
        return PerformanceSnapshot(
            timestamp=time.time(), cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent, memory_used=mem.used,
            gpu_memory_percent=gpu_pct, gpu_memory_used=gpu_used, gpu_utilization=gpu_util,
            disk_usage=psutil.disk_usage("/").percent,
            network_sent=net.bytes_sent, network_recv=net.bytes_recv, processes=[])

    def get_history(self, count: int = None) -> List[PerformanceSnapshot]:
        return list(self.history)[-(count or len(self.history)):]

    def get_average_metrics(self, duration: int = 60) -> Dict[str, Any]:
        cutoff = time.time() - duration
        snaps = [s for s in self.history if s.timestamp >= cutoff]
        if not snaps:
            return {}
        def avg(attr: str) -> float:
            return sum(getattr(s, attr) for s in snaps) / len(snaps)
        return {"duration": duration, "snapshot_count": len(snaps),
                "avg_cpu_percent": avg("cpu_percent"), "avg_memory_percent": avg("memory_percent"),
                "avg_gpu_memory_percent": avg("gpu_memory_percent"),
                "avg_gpu_utilization": avg("gpu_utilization"), "avg_disk_usage": avg("disk_usage")}

    def generate_report(self, duration: int = 3600) -> str:
        m = self.get_average_metrics(duration)
        if not m:
            return "# OpenManus performance\nНедостаточно данных"
        return ("# OpenManus performance\n"
                f"CPU {m['avg_cpu_percent']:.1f}% | RAM {m['avg_memory_percent']:.1f}% | "
                f"GPU-mem {m['avg_gpu_memory_percent']:.1f}% | GPU-util {m['avg_gpu_utilization']:.1f}% | "
                f"disk {m['avg_disk_usage']:.1f}% ({m['snapshot_count']} snapshots)")


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenManus performance monitor")
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()
    mon = PerformanceMonitor({"interval": 5})
    mon.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    mon.stop()
    print(mon.generate_report(args.duration))
    return 0


if __name__ == "__main__":
    sys.exit(main())
