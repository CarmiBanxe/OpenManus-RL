"""Smoke tests for openmanus_rl.middleware.rate_limiter."""

import pytest
from openmanus_rl.middleware.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
)


class TestRateLimiter:
    def _limiter(self, rpm: int = 5, rph: int = 100, global_rpm: int = 50) -> RateLimiter:
        cfg = RateLimitConfig(
            requests_per_minute=rpm,
            requests_per_hour=rph,
            global_requests_per_minute=global_rpm,
        )
        return RateLimiter(config=cfg)

    def test_allows_requests_under_limit(self):
        limiter = self._limiter(rpm=10)
        for _ in range(5):
            limiter.check_and_record("client-a")  # must not raise

    def test_blocks_at_rpm_limit(self):
        limiter = self._limiter(rpm=3)
        for _ in range(3):
            limiter.check_and_record("client-x")
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_and_record("client-x")
        assert "rate limit" in exc_info.value.reason.lower()

    def test_different_clients_independent(self):
        limiter = self._limiter(rpm=2)
        limiter.check_and_record("client-a")
        limiter.check_and_record("client-a")
        # client-a is now at limit; client-b should still work
        limiter.check_and_record("client-b")

    def test_global_limit_enforced(self):
        limiter = self._limiter(rpm=100, global_rpm=3)
        limiter.check_and_record("client-a")
        limiter.check_and_record("client-b")
        limiter.check_and_record("client-c")
        with pytest.raises(RateLimitExceeded):
            limiter.check_and_record("client-d")

    def test_get_client_id_from_api_key(self):
        from starlette.requests import Request
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [(b"x-api-key", b"secrettoken123456789")],
            "query_string": b"",
        }
        req = Request(scope)
        client_id = self._limiter().get_client_id(req)
        assert client_id.startswith("key:")
        # Must not include full key
        assert "secrettoken123456789" not in client_id

    def test_get_client_id_from_ip(self):
        from starlette.requests import Request
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
            "query_string": b"",
            "client": ("192.168.1.42", 12345),
        }
        req = Request(scope)
        client_id = self._limiter().get_client_id(req)
        assert "192.168.1.42" in client_id

    def test_rate_limit_exceeded_has_reason(self):
        exc = RateLimitExceeded("too many requests")
        assert exc.reason == "too many requests"
        assert "too many requests" in str(exc)


class TestBuildRateLimitMiddleware:
    def test_middleware_class_returned(self):
        from openmanus_rl.middleware.rate_limiter import build_rate_limit_middleware
        cls = build_rate_limit_middleware()
        # Must be a class (callable), not an instance
        assert callable(cls)

    def test_middleware_with_custom_config(self):
        from openmanus_rl.middleware.rate_limiter import build_rate_limit_middleware
        cfg = RateLimitConfig(requests_per_minute=10)
        cls = build_rate_limit_middleware(config=cfg)
        assert callable(cls)
