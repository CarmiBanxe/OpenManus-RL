"""Smoke tests for openmanus_rl.guardrails.policy (Charter §8 enforcement)."""

import pytest
from openmanus_rl.guardrails.policy import (
    GuardrailPolicy,
    PolicyViolation,
    check_request,
)


class TestPolicyViolation:
    def test_dataclass_frozen(self):
        v = PolicyViolation(rule="X", detail="y")
        with pytest.raises((AttributeError, TypeError)):
            v.rule = "Z"  # type: ignore[misc]

    def test_repr(self):
        v = PolicyViolation(rule="CHARTER_8", detail="Tor blocked")
        assert "CHARTER_8" in repr(v)


class TestGuardrailPolicyTorOnion:
    def setup_method(self):
        self.policy = GuardrailPolicy()

    def test_tor_network_blocked(self):
        v = self.policy.check_message("set up tor network proxy")
        assert v is not None
        assert v.rule == "CHARTER_8_TOR_ONION"

    def test_onion_site_blocked(self):
        v = self.policy.check_message("access http://example.onion/api")
        assert v is not None
        assert v.rule == "CHARTER_8_TOR_ONION"

    def test_torsocks_blocked(self):
        v = self.policy.check_message("use torsocks curl http://example.com")
        assert v is not None
        assert v.rule == "CHARTER_8_TOR_ONION"

    def test_onion_routing_blocked(self):
        v = self.policy.check_message("configure onion routing for privacy")
        assert v is not None
        assert v.rule == "CHARTER_8_TOR_ONION"

    def test_normal_message_allowed(self):
        assert self.policy.check_message("search for Python documentation") is None

    def test_search_query_allowed(self):
        assert self.policy.check_message("find articles about encryption") is None

    def test_privacy_without_tor_allowed(self):
        assert self.policy.check_message("use TLS for encrypted transport") is None


class TestGuardrailPolicyBlockedTools:
    def setup_method(self):
        self.policy = GuardrailPolicy()

    def test_blocked_tool_rejected(self):
        v = self.policy.check_tool_call("shell_exec")
        assert v is not None
        assert v.rule == "BLOCKED_TOOL"
        assert "shell_exec" in v.detail

    def test_rm_rf_blocked(self):
        v = self.policy.check_tool_call("rm_rf")
        assert v is not None

    def test_normal_tool_allowed(self):
        assert self.policy.check_tool_call("web_search") is None
        assert self.policy.check_tool_call("file_read") is None


class TestCheckRequestConvenienceFunction:
    def test_tor_blocked_via_convenience(self):
        v = check_request("access the tor network browser")
        assert v is not None
        assert v.rule == "CHARTER_8_TOR_ONION"

    def test_blocked_tool_via_convenience(self):
        v = check_request("normal message", tool_name="sudo_exec")
        assert v is not None
        assert v.rule == "BLOCKED_TOOL"

    def test_clean_request(self):
        assert check_request("summarise this document", tool_name="file_read") is None

    def test_none_tool_name_ok(self):
        assert check_request("hello world", tool_name=None) is None
