# Qdrant Setup Runbook — Legion Engine (OpenManus)
# Version: 1.0 | Created: 2026-07-15 | PROP-2026-0714-001

This runbook covers installing, configuring, and verifying Qdrant for Legion's
vector memory layer. If Qdrant is unavailable, Legion falls back automatically
to SQLite keyword search (see `openmanus_rl/memory/qdrant_memory.py`).

---

## 1. Prerequisites

- Docker (recommended) or native binary
- Port 6333 (HTTP) and 6334 (gRPC) available on 127.0.0.1
- Python package: `qdrant-client >= 1.7` (add to requirements or venv)
- Embedding model: `nomic-embed-text` available via Ollama at 127.0.0.1:11434

---

## 2. Install Qdrant (Docker — recommended)

```bash
# Pull and run Qdrant locally (bind to 127.0.0.1 only — Charter §8)
docker run -d \
  --name qdrant-legion \
  --restart unless-stopped \
  -p 127.0.0.1:6333:6333 \
  -p 127.0.0.1:6334:6334 \
  -v ~/.openmanus/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Verify
curl -s http://127.0.0.1:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

> **IMPORTANT**: Never expose Qdrant on 0.0.0.0. The `-p 127.0.0.1:6333:6333` bind
> keeps it localhost-only. Qdrant has no built-in auth — public exposure = data leak.

---

## 3. Install Qdrant Client in Python venv

```bash
# In the Legion venv (hf-env or project venv)
pip install "qdrant-client>=1.7"
pip install httpx   # required for the embedding helper
```

---

## 4. Pull the Embedding Model (Ollama)

```bash
# nomic-embed-text is the default embedding model in qdrant_config.yaml
ollama pull nomic-embed-text

# Verify
curl -s -X POST http://127.0.0.1:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}' | python3 -m json.tool
# Expected: {"embedding": [0.12, -0.34, ...]}
```

---

## 5. Configure Legion to Use Qdrant

The config file is `openmanus_rl/config/qdrant_config.yaml`.
Override individual values via environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `QDRANT_HOST` | `127.0.0.1` | Qdrant host (never change to 0.0.0.0) |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION` | `legion_memory` | Collection name |
| `QDRANT_VECTOR_SIZE` | `768` | Must match your embedding model dim |
| `QDRANT_TOP_K` | `5` | Results per search |
| `QDRANT_EMBED_URL` | `http://127.0.0.1:11434/api/embeddings` | Ollama endpoint |
| `QDRANT_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |

---

## 6. Smoke Test

```bash
cd ~/OpenManus-quality-gate-20260714

python3 - <<'EOF'
from openmanus_rl.memory.qdrant_memory import QdrantMemory

mem = QdrantMemory()
print("Using vector store:", mem.using_vector_store)

# Ingest a test record
ok = mem.ingest("Legion is an OpenManus RL agent engine.", session_id="smoke")
print("Ingest OK:", ok)

# Search
results = mem.search("agent engine", session_id="smoke")
print(f"Search returned {len(results)} results")
for r in results:
    print(f"  score={r.score:.3f}  text={r.text[:60]}")
EOF
```

Expected output:
```
Using vector store: True
Ingest OK: True
Search returned 1 results
  score=0.923  text=Legion is an OpenManus RL agent engine.
```

If `Using vector store: False`, Qdrant is unreachable and fallback is active (check Docker).

---

## 7. Systemd Integration (optional)

If running Legion as a systemd service, add Qdrant dependency:

```ini
# /etc/systemd/system/legion-agent.service
[Unit]
After=docker.service qdrant-legion.service

[Service]
Environment=QDRANT_HOST=127.0.0.1
Environment=QDRANT_PORT=6333
```

---

## 8. Rollback: Disable Qdrant

To fall back to SQLite without restarting Docker:

```bash
export QDRANT_HOST=invalid   # any non-resolvable host forces SQLite fallback
sudo systemctl restart legion-agent
```

See also: `docs/runbooks/rollback.md` §5.
