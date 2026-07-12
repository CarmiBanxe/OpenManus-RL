#!/usr/bin/env python3
"""
Демонстрация использования фреймворка принятия решений в контексте OpenManus
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openmanus_rl.agents.smart_decision_agent import SmartDecisionAgent

def demo_decision_framework():
    """Демонстрация работы фреймворка принятия решений"""
    
    # Конфигурация агента
    config = {
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
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:7b-instruct"
        }
    }
    
    # Создание агента
    agent = SmartDecisionAgent(config)
    
    # Сценарий 1: Анализ данных с ограниченным временем
    print("=" * 60)
    print("Сценарий 1: Анализ данных с ограниченным временем")
    print("=" * 60)
    
    state = {
        "urgency": 0.8,
        "available_resources": ["cpu", "memory"],
        "time_constraint": 30,  # 30 минут
        "current_task": "analyze_data"
    }
    
    available_actions = [
        "search_web_for_information",
        "run_complex_analysis",
        "generate_summary",
        "optimize_code"
    ]
    
    action = agent.select_action(state, available_actions)
    explanation = agent.get_detailed_explanation(state, action)
    
    print(f"Выбранное действие: {action}")
    print(f"Объяснение: {explanation}")
    
    # Обновление политики с наградой
    agent.update_policy(state, action, 0.9, state)
    
    # Сценарий 2: Поиск информации с низкой срочностью
    print("\n" + "=" * 60)
    print("Сценарий 2: Поиск информации с низкой срочностью")
    print("=" * 60)
    
    state = {
        "urgency": 0.2,
        "available_resources": ["cpu", "memory", "network"],
        "time_constraint": 120,  # 2 часа
        "current_task": "research_topic"
    }
    
    available_actions = [
        "search_web_for_information",
        "analyze_data",
        "generate_report",
        "optimize_code"
    ]
    
    action = agent.select_action(state, available_actions)
    explanation = agent.get_detailed_explanation(state, action)
    
    print(f"Выбранное действие: {action}")
    print(f"Объяснение: {explanation}")
    
    # Обновление политики с наградой
    agent.update_policy(state, action, 0.7, state)
    
    # Сценарий 3: Тестирование различных подходов
    print("\n" + "=" * 60)
    print("Сценарий 3: Тестирование различных подходов")
    print("=" * 60)
    
    state = {
        "urgency": 0.5,
        "available_resources": ["cpu", "memory"],
        "time_constraint": 60,
        "current_task": "mixed_task"
    }
    
    available_actions = [
        "search_web_for_information",
        "analyze_data",
        "generate_report",
        "optimize_code"
    ]
    
    approaches = ["expected_utility", "pareto", "behavioral", "optimal_stopping"]
    
    for approach in approaches:
        agent.config["decision"]["default_approach"] = approach
        action = agent.select_action(state, available_actions)
        print(f"Подход: {approach}, Выбранное действие: {action}")
    
    # Вывод статистики производительности
    print("\n" + "=" * 60)
    print("Статистика производительности")
    print("=" * 60)
    
    stats = agent.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo_decision_framework()
