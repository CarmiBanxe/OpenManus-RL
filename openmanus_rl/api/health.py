"""Health-check / metrics эндпоинт для OpenManus на FastAPI (127.0.0.1)."""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from openmanus_rl.observability import get_health_checker, get_metrics_collector
    from openmanus_rl.engines.enhanced_factory_with_observability import (
        create_engine,
        get_available_engines,
        get_observability_summary,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OBSERVABILITY_AVAILABLE = False


def create_health_app() -> FastAPI:
    app = FastAPI(title="OpenManus Health Check")

    if OBSERVABILITY_AVAILABLE:
        health_checker = get_health_checker()
        metrics_collector = get_metrics_collector()

    @app.get("/health")
    def health_check():
        if not OBSERVABILITY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Observability not available")
        try:
            metrics_collector.update_system_metrics()
            return health_checker.check_all()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Health check failed: {exc}")

    @app.get("/health/{component}")
    def component_health_check(component: str):
        if not OBSERVABILITY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Observability not available")
        try:
            return health_checker.check_component(component).to_dict()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Component health check failed: {exc}")

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        if not OBSERVABILITY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Observability not available")
        try:
            metrics_collector.update_system_metrics()
            from prometheus_client import generate_latest
            return generate_latest(metrics_collector._registry)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {exc}")

    @app.get("/engines")
    def engines():
        if not OBSERVABILITY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Observability not available")
        try:
            status = {}
            for name in get_available_engines():
                try:
                    engine = create_engine(name)
                    status[name] = {
                        "available": bool(hasattr(engine, "is_available") and engine.is_available()),
                        "metrics": engine.get_metrics() if hasattr(engine, "get_metrics") else {},
                    }
                except Exception as exc:  # noqa: BLE001
                    status[name] = {"available": False, "error": str(exc)}
            return {"engines": status}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get engines status: {exc}")

    @app.get("/observability")
    def observability():
        if not OBSERVABILITY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Observability not available")
        try:
            return get_observability_summary()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get observability summary: {exc}")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_health_app(), host="127.0.0.1", port=8080)
