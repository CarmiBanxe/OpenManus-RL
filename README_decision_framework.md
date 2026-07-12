# Decision Framework for OpenManus RL

## Обзор
Этот фреймворк предоставляет возможности принятия решений на основе математической теории принятия решений, поведенческой экономики и обучения с подкреплением.

## Ключевые компоненты

### 1. Модуль теории принятия решений (`openmanus_rl/decision/decision_theory.py`)
- `DecisionOption`: Представляет вариант решения с предсказанными исходами
- `DecisionFramework`: Основной фреймворк, объединяющий несколько подходов к принятию решений
- `UtilityCalculator`: Рассчитывает полезность для исходов
- `ParetoOptimizer`: Находит оптимальные по Парето решения
- `BehavioralModel`: Реализует модели поведенческого принятия решений
- `OptimalStopping`: Реализует модели оптимальной остановки

### 2. Умный агент принятия решений (`openmanus_rl/agents/smart_decision_agent.py`)
- Интегрирует теорию принятия решений с Ollama LLM
- Обучается на основе опыта
- Предоставляет подробные объяснения для решений

## Использование

### Базовое использование
```python
from openmanus_rl.agents.smart_decision_agent import SmartDecisionAgent

config = {
    "decision": {
        "default_approach": "expected_utility",
        "utility": {
            "function": "linear",
            "attribute_weights": {
                "reward": 1.0,
                "time": -0.2
            }
        }
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:7b-instruct"
    }
}


agent = SmartDecisionAgent(config)

state = {"urgency": 0.7, "available_resources": ["cpu", "memory"]}
available_actions = ["search_web", "analyze_data", "generate_report"]

action = agent.select_action(state, available_actions)
explanation = agent.get_detailed_explanation(state, action)

### Подходы к принятию решений
1. **Expected Utility (Ожидаемая полезность)**: Максимизирует ожидаемую полезность
2. **Pareto (Оптимизация Парето)**: Находит оптимальные по Парето решения
3. **Behavioral (Поведенческий)**: Использует поведенческие модели, такие как теория перспектив
4. **Optimal Stopping (Оптимальная остановка)**: Использует правила оптимальной остановки

## Тестирование
Запустите тестовый скрипт:
```bash
python test_smart_decision_agent.py
