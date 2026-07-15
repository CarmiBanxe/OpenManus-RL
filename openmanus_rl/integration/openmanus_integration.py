"""
Интеграция фреймворка принятия решений с OpenManus
"""

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
