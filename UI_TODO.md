# Sprint 6 — UI TODO

Под РЕАЛЬНЫЙ API агента: EnhancedDecisionAgent.select_action(state, available_actions, priority).
НЕ process_input. Sandbox-конфиг через openmanus_rl.config.load_config.

## Задачи
- [ ] ui/streamlit_app.py — select_action, share/публичность НЕТ (Streamlit по умолчанию localhost).
- [ ] ui/gradio_app.py — demo.launch(share=False, server_name="127.0.0.1"). НИКОГДА share=True (S-18 §1.2).
- [ ] Smoke-проверка: импорт модулей UI + вызов build-функции без запуска сервера (headless).

## Инварианты (красная линия)
- share=False, bind 127.0.0.1 — uncensored Legion-движок не выставляется наружу.
- Никакого process_input / config.get на классах / EnhancedDecisionAgent(config) с config в 1-м арг.
- Результат select_action: action/confidence/explanation/osint_enhanced/episode_id/timestamp.
