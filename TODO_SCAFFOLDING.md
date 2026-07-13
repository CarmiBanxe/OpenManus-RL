# Sprint 6 — Scaffolding TODO (после P0)

P0 (сделано, в main @ 34a1db9): config.py + auth/ + api/server.py + 13 тестов.
Ниже — исправное scaffolding под РЕАЛЬНЫЙ API и security-дефолты Legion (S-18 §1.2).

## Core scaffolding (приоритет)
- [ ] config/production.py, config/development.py, config/testing.py — тонкие обёртки над
      openmanus_rl.config.load_config (единый источник, без class-vs-dict расхождений).
- [ ] config/logging.py — RotatingFileHandler + консоль; уровень из load_config["log_level"].
- [ ] Dockerfile — python:3.12-slim (не 3.9), COPY requirements-legion.txt (не requirements.txt),
      CMD uvicorn openmanus_rl.api.server:app --host 127.0.0.1. Не выставлять наружу.
- [ ] docker-compose.yml — bind 127.0.0.1:8000 (не 0.0.0.0), redis localhost;
      БЕЗ nginx-wildcard/публичных портов; secret из env-файла.

## UI (следующим приоритетом, ТОЛЬКО безопасно)
- [ ] ui/streamlit_app.py — под реальный agent.select_action; localhost-only.
- [ ] ui/gradio_app.py — launch(share=False, server_name="127.0.0.1"). НИКОГДА share=True (красная линия).

## Отложено / требует отдельного решения оператора (НЕ в этом заходе)
- [ ] CI (.github/workflows) — переписать под ubuntu-runner (убрать /home/mmber/hf-env путь), реальные тесты.
- [ ] Prometheus/Grafana, backup/restore, Sphinx docs — тянут deps; делать при явной необходимости.
- [ ] sprint6_final_check.py — заменить os.path.exists на реальный прогон pytest (иначе ложная зелень).

## Инварианты
- Реальный API: select_action / process_voice_input_advanced / analyze_multi_agent_scenario.
- Никаких process_input / load_config-as-class / config.get на классах.
- Security: host 127.0.0.1, CORS localhost, secret+creds из env, share=False, без /query/public.
