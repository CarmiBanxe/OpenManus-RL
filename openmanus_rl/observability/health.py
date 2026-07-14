"""Модуль health-check для OpenManus."""
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    status: HealthStatus
    message: str
    last_check: float
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "details": self.details or {},
        }


class HealthChecker:
    """Проверщик здоровья компонентов OpenManus."""

    def __init__(self) -> None:
        self._components: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_global_check = 0.0
        self._global_status = HealthStatus.UNKNOWN
        self._global_message = "Not checked yet"

    def register_component(self, name: str, check_func: Callable[[], Dict[str, Any]],
                           interval: float = 30.0) -> None:
        with self._lock:
            self._components[name] = {
                "check_func": check_func,
                "interval": interval,
                "last_check": 0.0,
                "health": ComponentHealth(HealthStatus.UNKNOWN, "Not checked yet", 0.0),
            }

    def unregister_component(self, name: str) -> None:
        with self._lock:
            self._components.pop(name, None)

    def check_component(self, name: str) -> ComponentHealth:
        if name not in self._components:
            return ComponentHealth(HealthStatus.UNKNOWN,
                                   f"Component {name} not registered", time.time())
        component = self._components[name]
        now = time.time()
        if now - component["last_check"] >= component["interval"]:
            try:
                result = component["check_func"]()
                status, message, details = HealthStatus.HEALTHY, "OK", {}
                if isinstance(result, dict):
                    status_str = result.get("status", "healthy")
                    message = result.get("message", "OK")
                    details = dict(result.get("details", {}))
                    details.update({k: v for k, v in result.items()
                                    if k not in ("status", "message", "details")})
                    status = {
                        "healthy": HealthStatus.HEALTHY,
                        "degraded": HealthStatus.DEGRADED,
                        "unhealthy": HealthStatus.UNHEALTHY,
                    }.get(status_str, HealthStatus.UNKNOWN)
                component["health"] = ComponentHealth(status, message, now, details)
                component["last_check"] = now
            except Exception as exc:  # noqa: BLE001
                component["health"] = ComponentHealth(
                    HealthStatus.UNHEALTHY, f"Check failed: {exc}", now, {"error": str(exc)})
                component["last_check"] = now
        return component["health"]

    def check_all(self) -> Dict[str, Any]:
        with self._lock:
            results, overall_healthy, overall_degraded = {}, True, False
            for name in list(self._components):
                # прямой вызов без повторного захвата lock (check_component не лочит)
                health = self._check_component_unlocked(name)
                results[name] = health.to_dict()
                if health.status == HealthStatus.UNHEALTHY:
                    overall_healthy = False
                elif health.status == HealthStatus.DEGRADED:
                    overall_degraded = True
            if overall_healthy and not overall_degraded:
                gs, gm = HealthStatus.HEALTHY, "All components healthy"
            elif overall_healthy and overall_degraded:
                gs, gm = HealthStatus.DEGRADED, "Some components degraded"
            else:
                gs, gm = HealthStatus.UNHEALTHY, "Some components unhealthy"
            self._last_global_check = time.time()
            self._global_status, self._global_message = gs, gm
            return {"status": gs.value, "message": gm,
                    "last_check": self._last_global_check, "components": results}

    def _check_component_unlocked(self, name: str) -> ComponentHealth:
        component = self._components[name]
        now = time.time()
        if now - component["last_check"] >= component["interval"]:
            try:
                result = component["check_func"]()
                status, message, details = HealthStatus.HEALTHY, "OK", {}
                if isinstance(result, dict):
                    status_str = result.get("status", "healthy")
                    message = result.get("message", "OK")
                    details = dict(result.get("details", {}))
                    details.update({k: v for k, v in result.items()
                                    if k not in ("status", "message", "details")})
                    status = {
                        "healthy": HealthStatus.HEALTHY,
                        "degraded": HealthStatus.DEGRADED,
                        "unhealthy": HealthStatus.UNHEALTHY,
                    }.get(status_str, HealthStatus.UNKNOWN)
                component["health"] = ComponentHealth(status, message, now, details)
                component["last_check"] = now
            except Exception as exc:  # noqa: BLE001
                component["health"] = ComponentHealth(
                    HealthStatus.UNHEALTHY, f"Check failed: {exc}", now, {"error": str(exc)})
                component["last_check"] = now
        return component["health"]

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        with self._lock:
            if name in self._components:
                return self._components[name]["health"]
            return None

    def get_global_health(self) -> Dict[str, Any]:
        with self._lock:
            return {"status": self._global_status.value,
                    "message": self._global_message, "last_check": self._last_global_check}


_health_checker = None


def get_health_checker() -> HealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
