import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmanus_rl.integration.openmanus_integration import OpenManusDecisionIntegration

class TestDecisionIntegration(unittest.TestCase):
    def setUp(self):
        self.integration = OpenManusDecisionIntegration()
    
    def test_select_action(self):
        """Тест выбора действия"""
        result = self.integration.select_action(
            "analyze_data", 0.7, ["cpu", "memory"], 30, 
            ["search_web", "analyze_data", "generate_report"]
        )
        
        self.assertIn("action", result)
        self.assertIn("explanation", result)
        self.assertIn("approach", result)
    
    def test_update_policy(self):
        """Тест обновления политики"""
        state = {"urgency": 0.7}
        next_state = {"urgency": 0.5}
        action = "search_web"
        
        self.integration.update_policy(state, action, 0.8, next_state)
        stats = self.integration.get_performance_stats()
        self.assertEqual(stats["total_experiences"], 1)

if __name__ == "__main__":
    unittest.main()
