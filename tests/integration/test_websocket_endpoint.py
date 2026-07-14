"""Интеграционные тесты streaming API (FastAPI TestClient + опц. живой WebSocket)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from fastapi.testclient import TestClient

    from openmanus_rl.api.streaming import create_streaming_app

    STREAMING_APP_AVAILABLE = True
except ImportError:
    STREAMING_APP_AVAILABLE = False


@unittest.skipIf(not STREAMING_APP_AVAILABLE, "FastAPI or streaming app not available")
class TestStreamingApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_streaming_app())

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("OpenManus Streaming Test", response.text)

    def test_connections_endpoint(self):
        response = self.client.get("/connections")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["active_connections"], 0)
        self.assertEqual(data["connection_ids"], [])

    def test_routes_registered(self):
        # HTTP-маршрут SSE — в OpenAPI; WebSocket в OpenAPI НЕ попадает (FastAPI) — через routes.
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/v1/stream/generate", paths)
        route_paths = [getattr(r, "path", None) for r in create_streaming_app().routes]
        self.assertIn("/ws/stream", route_paths)


@unittest.skipIf(not STREAMING_APP_AVAILABLE, "FastAPI or streaming app not available")
class TestWebSocketLive(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_streaming_live(self):
        # Требует поднятого сервера на :8081 — в CI/юнит-прогоне пропускаем.
        try:
            import websockets
            async with websockets.connect("ws://localhost:8081/ws/stream") as ws:
                await ws.send(json.dumps({"request_id": "t", "prompt": "Hello"}))
                await ws.recv()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"WebSocket server не поднят: {exc}")


if __name__ == "__main__":
    unittest.main()
