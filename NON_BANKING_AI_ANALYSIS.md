# Анализ мирового опыта ИИ-помощников (не-банковских)

## Ключевые выводы для OpenManus RL

### 1. UX/UI паттерны (адаптированные)
- **Dual-Track UX**: классический + AI режим (Alipay "Project Treasure")
- **Hybrid Intent Interface**: чат/голос ИЛИ традиционные кнопки
- **AI Tab**: специальная вкладка в навигации (KakaoBank)
- **Rich Cards**: интерактивные карточки вместо текстовых ответов
- **Avatar Assistant**: персонализированные AI-помощники (OCBC Wendy & Wayne)

### 2. Архитектурные паттерны
- **Composite Tools**: детерминированная логика вне LLM (Nubank)
- **Специализированные агенты по доменам**: research, writing, analysis (Toss Bank)
- **Local LLM deployment**: DeepSeek локально (WeLab Bank)
- **Federated Learning**: для обучения без передачи данных (WeBank)
- **MCP-first**: для внешних сервисов и API (Alipay, MercadoPago)

### 3. Production-стек
- **LangGraph + LangSmith + LangChain**: основной AI-оркестрации (Nubank)
- **DSPy + Japa optimizer**: автоматическая оптимизация промптов
- **Temporal workflow**: для durability и audit trail
- **AWS Bedrock**: 4 сервиса для быстрого старта (Toss)
- **Self-hosted LLM**: для privacy и безопасности

### 4. Региональные особенности (адаптированные)
- **Китай**: Alipay "Абао" (300M AI транзакций), WeBank FATE
- **Япония**: SBI AI phone operator (естественная речь)
- **Корея**: KakaoBank AI Tab, Toss specialized agents
- **Латам**: Nubank (131M пользователей), MercadoPago AI Assistant
- **ЮВА**: DBS (430+ AI use cases), WeLab DeepSeek local

### 5. Open Source компоненты
- **FATE**: Federated learning (WeBank)
- **mpesa-mcp**: Первый African fintech MCP (адаптировать под другие API)
- **assistant-ui**: Conversational UI (MIT)
- **LangGraph**: Stateful agent orchestration
- **DSPy**: Prompt optimization
