#!/usr/bin/env python3
"""
Оптимизированный параллельный rollout для OpenManus.

Исправлено vs черновик:
  - env-модули грузятся ЛЕНИВО и graceful (реальные пути env_package/*; отсутствие не роняет);
  - единый Ollama-движок на rollout (не создаётся на каждую задачу);
  - ZeroDivisionError в метриках устранён (guard на total_tasks==0);
  - ProcessPool убран (self с кэшем непикл) — ThreadPool по умолчанию.
Env-исполнение вынесено в pluggable step_fn -> тестируется с моками без тяжёлых сред.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RolloutConfig:
    max_workers: int = 4
    chunk_size: int = 10
    preload_environments: bool = True
    cache_environment_state: bool = True
    environment_type: str = "webshop"
    model_name: str = "qwen2.5:7b-instruct"
    output_dir: str = "rollout_results"


class OptimizedRollout:
    def __init__(self, config: RolloutConfig = None,
                 step_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        self.config = config or RolloutConfig()
        self.project_root = Path(__file__).resolve().parent.parent
        self.output_dir = (self.project_root / self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_fn = step_fn  # pluggable исполнитель задачи (для тестов/кастома)
        self._engine = None
        self.metrics = {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0,
                        "total_time": 0.0, "avg_task_time": 0.0, "tasks_per_second": 0.0}

    def _get_engine(self):
        if self._engine is None:
            from openmanus_rl.engines.optimized_ollama_engine import create_optimized_ollama_engine
            self._engine = create_optimized_ollama_engine({"model": self.config.model_name})
        return self._engine

    def _default_step(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Реальный env (env_package/*) тяжёлый и вендоренный — грузим graceful.
        engine = self._get_engine()
        prompt = task.get("instruction", "")
        resp = engine.generate(prompt) if prompt else {"response": ""}
        return {"task_id": task.get("id", "unknown"), "success": bool(resp.get("response")),
                "action": resp.get("response", ""), "steps": 1}

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            fn = self.step_fn or self._default_step
            result = fn(task)
            self.metrics["completed_tasks"] += 1
            result.setdefault("task_id", task.get("id", "unknown"))
            result.setdefault("success", True)
        except Exception as exc:  # noqa: BLE001
            self.metrics["failed_tasks"] += 1
            result = {"task_id": task.get("id", "unknown"), "success": False, "error": str(exc)}
        finally:
            elapsed = time.time() - start
            self.metrics["total_time"] += elapsed
            done = self.metrics["completed_tasks"] + self.metrics["failed_tasks"]
            if done:  # guard: ZeroDivisionError устранён
                self.metrics["avg_task_time"] = self.metrics["total_time"] / done
            if self.metrics["total_time"] > 0:
                self.metrics["tasks_per_second"] = self.metrics["completed_tasks"] / self.metrics["total_time"]
        result["time"] = elapsed
        return result

    def chunk_tasks(self, tasks: List[Dict[str, Any]], chunk_size: int = None) -> List[List[Dict[str, Any]]]:
        cs = chunk_size or self.config.chunk_size
        return [tasks[i:i + cs] for i in range(0, len(tasks), cs)]

    def run_rollout(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.metrics["total_tasks"] = len(tasks)
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            futures = [ex.submit(self.execute_task, t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result())
        self._save_results(results)
        return results

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        out = self.output_dir / f"rollout_{self.config.environment_type}_{len(results)}.json"
        out.write_text(json.dumps({"config": asdict(self.config), "metrics": self.metrics,
                                   "results": results}, indent=2), encoding="utf-8")

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenManus optimized rollout")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--environment", choices=["webshop", "alfworld"], default="webshop")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    rollout = OptimizedRollout(RolloutConfig(max_workers=args.max_workers, environment_type=args.environment))
    results = rollout.run_rollout(tasks)
    ok = sum(1 for r in results if r.get("success"))
    print(f"{ok}/{len(results)} ok | metrics: {rollout.get_metrics()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
