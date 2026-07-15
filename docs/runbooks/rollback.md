# Rollback Runbook — Legion Engine (OpenManus)
# Version: 1.0 | Created: 2026-07-15 | PROP-2026-0714-001

This runbook covers how to roll back the Legion engine after a failed deployment,
config change, or quality-gate regression. Follow these steps in order.

---

## 1. Symptoms Requiring Rollback

| Symptom | Severity | Action |
|---------|----------|--------|
| `/health` returns `degraded` immediately after deploy | HIGH | Rollback deploy |
| Eval gate exits non-zero after deploy | HIGH | Rollback code |
| `p95 latency > 10s` on `/chat` or `/stream` | MEDIUM | Rollback config first |
| Error rate > 5% on `/chat` for >2 minutes | HIGH | Rollback deploy |
| LLM cost spike (>2× baseline) with no load increase | MEDIUM | Check rate limiter config |
| OOM / process crash loop | HIGH | Rollback + reduce memory config |

---

## 2. Pre-Rollback: Capture State

Before rolling back, capture the failure evidence so it can be analysed later.

```bash
# 1. Snapshot current health
curl -s http://127.0.0.1:8090/health | tee /tmp/rollback-health-$(date +%s).json

# 2. Capture last 500 lines of agent server log
journalctl -u legion-agent -n 500 --no-pager > /tmp/rollback-agent-$(date +%s).log 2>&1 || \
  tail -500 ~/.openmanus/legion.log > /tmp/rollback-agent-$(date +%s).log

# 3. Record current git state
cd ~/OpenManus && git log --oneline -10 > /tmp/rollback-git-$(date +%s).txt
```

---

## 3. Code Rollback (bad commit)

```bash
cd ~/OpenManus

# Find the last known-good commit SHA from git log
git log --oneline -20

# Roll back to that SHA (replace <SHA> with actual)
git checkout <SHA>

# Or: roll back by N commits without losing history
git revert HEAD~1..HEAD   # creates a revert commit — preferred over reset

# Restart the service
sudo systemctl restart legion-agent
sleep 3
curl -s http://127.0.0.1:8090/health
```

If you need to roll back the quality-gate worktree specifically:
```bash
cd ~/OpenManus-quality-gate-20260714
git log --oneline -10
git checkout <LAST_GOOD_SHA>
```

---

## 4. Config Rollback (environment variable or YAML change)

```bash
# Legion reads config from environment at startup.
# To revert a config change, restore the previous value and restart.

# Example: revert a rate limit change
export RATE_LIMIT_RPM=60        # restore default
export RATE_LIMIT_TOKEN_BUDGET=100000

# Or edit systemd override
sudo systemctl edit legion-agent
# Change the EnvironmentFile or Environment= lines, then:
sudo systemctl daemon-reload
sudo systemctl restart legion-agent

# Verify
curl -s http://127.0.0.1:8090/health
```

---

## 5. Qdrant Memory Rollback

If a Qdrant schema or collection change caused problems:

```bash
# Option A: disable Qdrant, use SQLite fallback
export QDRANT_HOST=invalid   # forces fallback
sudo systemctl restart legion-agent

# Option B: drop the collection and recreate (loses vector index, not SQLite data)
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(host='127.0.0.1', port=6333)
c.delete_collection('legion_memory')
print('Collection deleted — will be recreated on next ingest')
"
sudo systemctl restart legion-agent
```

---

## 6. Verify Recovery

```bash
# Health check
curl -s http://127.0.0.1:8090/health | python3 -m json.tool

# Smoke test: minimal chat roundtrip
curl -s -X POST http://127.0.0.1:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "ping", "session_id": "rollback-check"}' | python3 -m json.tool

# Eval gate (advisory)
python3 scripts/eval_quality_gate.py
echo "Gate exit code: $?"
```

Expected recovery state:
- `/health` returns `{"status": "ok", ...}`
- `/chat` returns a non-error response within 5 seconds
- Eval gate exits 0 (or known advisory state)

---

## 7. Post-Rollback: File Incident

After confirming recovery:

1. Record what changed, what broke, and what was reverted.
2. Create a follow-up ticket referencing this rollback event.
3. If a runbook step was missing or wrong, update this file.
4. If the eval gate caught the regression: update `scripts/eval_baseline.json` after the fix.

---

## 8. Contacts & Escalation

| Role | When to escalate |
|------|-----------------|
| Operator (this machine) | Any rollback |
| BANXE factory agent | If rollback requires code changes |
| Architect review | If rollback affects a public API contract |

> **Charter §8 reminder**: Do NOT route any traffic through Tor/onion paths during incident
> response or rollback. Use only direct `127.0.0.1` or Cloudflare Tunnel connections.
