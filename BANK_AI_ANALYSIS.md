# Анализ документа EMI BANXE AI BANK

## Ключевые выводы для OpenManus RL

### 1. Архитектурные паттерны
- **Слоистая архитектура**: 7 слоев от инфраструктуры до презентации
- **Специализированные агенты**: лучше работают, чем один генеральный
- **Stateful graph workflow**: через LangGraph вместо цепочек
- **Composite Tools**: детерминированные процессы в tool-функции

### 2. Математические основы
- **PRAGMA (Revolut)**: двухветвевой трансформер для транзакций
- **nuFormer (Nubank)**: GPT-style decoder + DCNv2 fusion
- **GNN для fraud**: гетерогенные графы user-merchant-transaction
- **Federated Learning**: GDPR-compliant обучение

### 3. Production-стек
- **Temporal**: workflow engine для банковских операций
- **LangGraph**: оркестрация реалтайм-транзакций
- **DeerFlow 2.0**: SuperAgent для long-horizon задач
- **Strands SDK**: production multi-agent framework

### 4. Безопасность и Compliance
- **NeMo Guardrails**: программные ограничения агента
- **EU AI Act**: decision lineage schema
- **OWASP Top 10 для LLM**: mitigation strategies
