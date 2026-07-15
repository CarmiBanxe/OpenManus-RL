"""
Guardrail policy for Legion (OpenManus) engine.

Enforces request-level content policy before any LLM call or tool execution.
Integrated into agent_server.py as a FastAPI middleware.

CHARTER §8 (immutable): Tor/onion/anonymizing transports are FORBIDDEN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Policy violation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyViolation:
    rule: str
    detail: str


# ---------------------------------------------------------------------------
# Banned pattern sets
# ---------------------------------------------------------------------------

# Tor / onion: FORBIDDEN per Charter §8 (Banxe SAFE-PORT, 2026-07-14)
_TOR_ONION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btor\s*(network|socks|proxy|browser|daemon)\b", re.IGNORECASE),
    re.compile(r"\.onion\b", re.IGNORECASE),
    re.compile(r"\bonion[\s_-]?(routing|site|service|hidden|access)\b", re.IGNORECASE),
    re.compile(r"\btorify\b", re.IGNORECASE),
    re.compile(r"\btorsocks\b", re.IGNORECASE),
]

# Tool calls that must never execute autonomously (require HITL clearance)
_BLOCKED_TOOL_NAMES: frozenset[str] = frozenset({
    "shell_exec",
    "bash_unrestricted",
    "sudo_exec",
    "rm_rf",
    "system_command",
})

# Topics that are explicitly out of scope for Legion engine
_BANNED_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    # FCA-regulated financial compliance logic must not run inside Legion
    re.compile(r"\b(CASS|FCA|PSD2|AML|KYC|SAR)\s+compliance\b", re.IGNORECASE),
    # Banking credentials / secrets exfiltration attempts
    re.compile(r"\b(iban|swift|sort\s*code|bic)\b.*\b(send|exfil|extract|dump)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

@dataclass
class GuardrailPolicy:
    """Stateless policy checker. Instantiate once, reuse across requests."""

    extra_banned_patterns: list[re.Pattern[str]] = field(default_factory=list)
    extra_blocked_tools: frozenset[str] = field(default_factory=frozenset)

    def check_message(self, message: str) -> Optional[PolicyViolation]:
        """Return PolicyViolation if message violates policy, else None."""
        for pat in _TOR_ONION_PATTERNS:
            if pat.search(message):
                return PolicyViolation(
                    rule="CHARTER_8_TOR_ONION",
                    detail=f"Request references Tor/onion networks (pattern: {pat.pattern!r}). "
                           "Tor/onion is FORBIDDEN per charter §8.",
                )
        for pat in _BANNED_TOPIC_PATTERNS:
            if pat.search(message):
                return PolicyViolation(
                    rule="SCOPE_VIOLATION",
                    detail=f"Request appears to target a banned topic scope (pattern: {pat.pattern!r}).",
                )
        for pat in self.extra_banned_patterns:
            if pat.search(message):
                return PolicyViolation(
                    rule="CUSTOM_POLICY",
                    detail=f"Request blocked by custom policy pattern: {pat.pattern!r}.",
                )
        return None

    def check_tool_call(self, tool_name: str) -> Optional[PolicyViolation]:
        """Return PolicyViolation if tool_name is blocked, else None."""
        blocked = _BLOCKED_TOOL_NAMES | self.extra_blocked_tools
        if tool_name in blocked:
            return PolicyViolation(
                rule="BLOCKED_TOOL",
                detail=f"Tool '{tool_name}' is blocked by guardrail policy (requires HITL clearance).",
            )
        return None


# ---------------------------------------------------------------------------
# Module-level default instance and convenience function
# ---------------------------------------------------------------------------

_DEFAULT_POLICY = GuardrailPolicy()


def check_request(message: str, tool_name: Optional[str] = None) -> Optional[PolicyViolation]:
    """
    Convenience wrapper using the module-level default policy.

    Returns PolicyViolation on first match, or None if the request is clean.
    """
    violation = _DEFAULT_POLICY.check_message(message)
    if violation:
        return violation
    if tool_name is not None:
        return _DEFAULT_POLICY.check_tool_call(tool_name)
    return None
