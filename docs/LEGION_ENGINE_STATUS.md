# Legion Engine — Status Report (S9–S19)

**Repo:** CarmiBanxe/OpenManus-RL (fork) · **Branch:** main · **HEAD:** 64e62e7
**Контур:** приватный Legion (S-18) — localhost-bound, секреты из env, uncensored-движок изолирован.

---

## Архитектура (слои снизу вверх)

```
                    ┌─────────────────────────────────────────────┐
  Entry (S18/S19)   │ agent_server (FastAPI /chat /stream /reset)  │
                    │  + rate-limit/audit/SecurityAudit · CLI ·    │
                    │  Dockerfile.agent   (127.0.0.1, API-key opt) │
                    └───────────────────────┬─────────────────────┘
  Facade (S17)      │            LegionAgent · AgentConfig         │
                    └───┬───────────┬───────────┬─────────────┬────┘
  Capabilities        streaming   memory/RAG   tools        observability
                     (S12)       (S13/S14)    (S15)         (S11)
                        │           │           │             │
  Core (S10)          └───────  LiteLLMAdapter (:4000, auth, fallback) ──────┘
  Perf (S9)            RTX 4070/64GB detect · optimized_ollama · rollout
  Eval (S16)           EvalHarness (мерит S10–S15, det + live suites)
```

## Слои и способности

| # | Слой | Что даёт | Ключевые модули |
|---|---|---|---|
| S9 | perf | детект RTX 4070/64GB, GPU-слои, rollout-воркеры | `scripts/performance_*`, `engines/optimized_ollama_engine` |
| S10 | adapter | тонкий LiteLLM-адаптер (:4000): generate/chat, auth, fallback, метрики | `engines/enhanced_factory.py` (`LiteLLMAdapter`) |
| S11 | observability | Prometheus-метрики (изолир. registry, 127.0.0.1), structlog, health | `observability/{metrics,logging,health}` |
| S12 | streaming | async SSE generate/chat, reasoning_content, retry-до-1-токена; WS+SSE API | `engines/streaming_adapter.py`, `api/streaming.py` |
| S13 | memory | персистентные диалоговые turn'ы (SQLite), контекст, trim/summarize | `memory/{sqlite_memory,conversation_memory}` |
| S14 | RAG | семантич. поиск (Ollama nomic-embed 768d, cosine), инъекция контекста | `memory/{embeddings,semantic_memory}` |
| S15 | tools | OpenAI tool-calling, агентик-луп (non-stream), safe builtins, octotools-мост | `tool_calling/*`, `engines/tool_calling_adapter.py` |
| S16 | eval | харнесс (latency p50/p95, success-rate), det + live suites, CLI | `eval/*`, `scripts/run_eval.py` |
| S17 | agent-фасад | единый `LegionAgent` (chat/stream + auto memory/RAG/tools) | `agent/{config,legion_agent,cli}` |
| S18 | entry point | REST-сервер, CLI, Dockerfile — движок как сервис | `api/agent_server.py`, `agent/cli.py` |
| S19 | security | rate-limit + audit (без секретов) + SecurityAudit self-check | `api/security.py` |

## Тесты / гейты

- **502 теста** собрано; валидатор-группы (`scripts/validate_sprint.py`) все зелёные:
  `perf-optimization · external-api · observability · streaming · memory · rag · tools · eval · agent · entrypoint · security` (+ ранние sprint5–7/docs/metrics/network/security-scan).
- Каждый слой проверен **вживую** на шлюзе (не только моки): stream reasoning, RAG-recall (Ollama), tool-calling (smart→4183), agent-капстоун, REST /chat→144.

## Ограничения (осознанные)

- ✅ ~~streaming + tools не совмещены~~ **— СНЯТО в S20**: `stream()` теперь resolve-ит
  инструменты non-stream (надёжные tool_calls), затем стримит финал. `chat()` — тоже tools.
- ✅ ~~Память по умолчанию `:memory:`~~ **— решено в S21**: `agent_server` по умолчанию
  использует персистентный файловый db + `SessionManager` (TTL/лимит); история переживает
  рестарт. (Библиотечный `AgentConfig` по-прежнему `:memory:` для тестов/встраивания.)
- **Reasoning-модели (smart/fast) медленны** (adapter_chat ~56s): часть бюджета уходит в `reasoning_content` (флаг `include_reasoning`).
- Non-master ключи к шлюзу → `400 "No connected db"` (LiteLLM в no-SQL-DB режиме) — by design, не баг движка.

## Открытые пункты (не блокеры)

- 🟢 **Security закрыт:** старый gateway-ключ ротирован и мёртв (HTTP 400); 2 hardcoded-страгглера исправлены; авто-источника 401-шума нет (13 мин idle = 0). Мёртвый ключ остался как **текст** в ~17 доках/ledger'ах `~/banxe/*` — безвредно (косметика).
- 🟡 **docker `litellm-gateway`**: второй LiteLLM с ОТДЕЛЬНЫМ ключом, publish `0.0.0.0:4000` конфликтует с native (native владеет портом). Инфра-вопрос (Terminal-A), не движок.
- 🟡 `agent_server` не подключён к `docker-compose.yml` (есть `Dockerfile.agent`; существующий S6-compose не трогали).

## Рекомендации по развитию

1. ✅ **streaming+tools в одном потоке** — сделано (**S20**): resolve non-stream → стрим финала.
2. ✅ **Персистентная память** — сделано (**S21**): файловый `memory_db` + `SessionManager` TTL/лимит.
3. ✅ **Prompt/persona-слой** — сделано (**S22**): `PersonaConfig`/PERSONAS + операционные guardrails (S-18: без контент-цензуры).
4. ✅ **compose-сервис** — сделано (**S23**): `docker-compose.agent.yml` (127.0.0.1-only,
   host.docker.internal, секреты из env, persist-volume; `docker compose config` валиден).
5. ✅ **Расширить eval** — сделано (**S24**): live-наборы agent-recall/RAG (Ollama) +
   `RegressionThresholds` (min success-rate / max p95/mean) с exit-code гейтом для CI.

**Roadmap полностью закрыт (S20–S24).**

---
*S9–S19: perf → adapter → observability → streaming → memory → RAG → tools → eval → agent-фасад → REST/CLI/Docker → security. Всё зелёное, влито в `main`, запушено на форк.*
