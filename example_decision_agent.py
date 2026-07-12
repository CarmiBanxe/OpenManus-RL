#!/usr/bin/env python3
"""
Example of using DecisionAgent with Ollama
"""

import sys
import logging
from pathlib import Path

# Add OpenManus to path
sys.path.append(str(Path(__file__).parent))

from openmanus_rl.agents.decision_agent import DecisionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate DecisionAgent"""
    logger.info("Running DecisionAgent example with Ollama integration...")
    
    # Create agent with Ollama configuration
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
            },
            "behavioral": {
                "use_satisficing": False,
                "reference_point": 0.0,
                "loss_aversion": 2.0
            }
        },
        "learning_rate": 0.01
    }
    
    agent = DecisionAgent(config)
    
    # Example scenario: choosing between different tasks
    state = {
        "urgency": 0.7,
        "available_resources": ["cpu", "memory"],
        "time_constraint": 30  # minutes
    }
    
    available_actions = [
        "search_web_for_information",
        "analyze_local_document",
        "generate_report",
        "optimize_code"
    ]
    
    logger.info(f"Available actions: {available_actions}")
    logger.info(f"Current state: {state}")
    
    # Select action using decision theory
    selected_action = agent.select_action(state, available_actions)
    logger.info(f"Selected action: {selected_action}")
    
    # Get explanation
    explanation = agent.get_decision_explanation(state, selected_action)
    if explanation:
        logger.info(f"Explanation: {explanation}")
    
    # Simulate reward and update policy
    reward = 0.8  # Positive reward for good action
    next_state = {
        "urgency": 0.3,
        "available_resources": ["cpu", "memory"],
        "time_constraint": 25
    }
    
    agent.update_policy(state, selected_action, reward, next_state)
    logger.info(f"Policy updated with reward: {reward}")
    
    # Test different decision approaches
    logger.info("\nTesting different decision approaches:")
    for approach in ["expected_utility", "pareto", "behavioral", "optimal_stopping"]:
        agent.default_approach = approach
        action = agent.select_action(state, available_actions)
        logger.info(f"Approach: {approach}, Selected action: {action}")

if __name__ == "__main__":
    main()
