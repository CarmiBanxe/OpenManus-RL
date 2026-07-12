#!/usr/bin/env python3
"""
Интеграция фреймворка принятия решений с основным проектом OpenManus
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openmanus_rl.agents.smart_decision_agent import SmartDecisionAgent

class OpenManusDecisionIntegration:
    """Класс для интеграции фреймворка принятия решений с OpenManus"""
    
    def __init__(self, ollama_base_url="http://localhost:11434", ollama_model="qwen2.5:7b-instruct"):
        """Инициализация интеграции"""
        self.config = {
            "decision": {
                "default_approach": "expected_utility",
                "utility": {
                    "function": "linear",
                    "attribute_weights": {
                        "reward": 1.0,
                        "cost": -0.5,
                        "time": -0.2
                    }
                }
            },
            "ollama": {
                "base_url": ollama_base_url,
                "model": ollama_model
            }
        }
        
        self.agent = SmartDecisionAgent(self.config)
    
    def select_action(self, task_description, urgency, available_resources, time_constraint, available_actions):
        """Выбор действия на основе контекста задачи"""
        state = {
            "urgency": urgency,
            "available_resources": available_resources,
            "time_constraint": time_constraint,
            "current_task": task_description
        }
        
        action = self.agent.select_action(state, available_actions)
        explanation = self.agent.get_detailed_explanation(state, action)
        
        return {
            "action": action,
            "explanation": explanation,
            "approach": self.config["decision"]["default_approach"]
        }
    
    def update_policy(self, current_state, action, reward, next_state):
        """Обновление политики на основе обратной связи"""
        self.agent.update_policy(current_state, action, reward, next_state)
    
    def get_performance_stats(self):
        """Получение статистики производительности"""
        return self.agent.get_performance_stats()
    
    def set_decision_approach(self, approach):
        """Установка подхода к принятию решений"""
        if approach in ["expected_utility", "pareto", "behavioral", "optimal_stopping"]:
            self.config["decision"]["default_approach"] = approach
            self.agent.config["decision"]["default_approach"] = approach
            return True
        return False

# Пример использования в контексте OpenManus
def example_openmanus_usage():
    """Пример использования интеграции в контексте OpenManus"""
    
    # Создание интеграции
    decision_integration = OpenManusDecisionIntegration()
    
    # Сценарий: OpenManus получил запрос на анализ финансовых данных
    task_description = "analyze_financial_data"
    urgency = 0.7  # Высокая срочность
    available_resources = ["cpu", "memory", "network"]
    time_constraint = 45  # 45 минут
    available_actions = [
        "search_web_for_financial_news",
        "analyze_data_with_python",
        "generate_financial_report",
        "optimize_analysis_code"
    ]
    
    # Выбор действия
    result = decision_integration.select_action(
        task_description, urgency, available_resources, 
        time_constraint, available_actions
    )
    
    print(f"Выбранное действие: {result['action']}")
    print(f"Объяснение: {result['explanation']}")
    print(f"Подход: {result['approach']}")
    
    # Обновление политики на основе обратной связи
    next_state = {
        "urgency": 0.5,
        "available_resources": ["cpu", "memory", "network"],
        "time_constraint": 40,
        "current_task": "analyze_financial_data"
    }
    
    decision_integration.update_policy(
        {
            "urgency": urgency,
            "available_resources": available_resources,
            "time_constraint": time_constraint,
            "current_task": task_description
        },
        result['action'],
        0.8,  # Награда за хорошее выполнение
        next_state
    )
    
    # Получение статистики
    stats = decision_integration.get_performance_stats()
    print("\nСтатистика производительности:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_openmanus_usage()
