#!/usr/bin/env python3
"""
Test script for Smart Decision Agent
"""

import sys
import logging
from pathlib import Path

# Add OpenManus to path
sys.path.append(str(Path(__file__).parent))

from openmanus_rl.agents.smart_decision_agent import SmartDecisionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_decision_framework():
    """Test the decision framework directly"""
    logger.info("Testing Decision Framework...")
    
    from openmanus_rl.decision.decision_theory import DecisionFramework, DecisionOption
    
    config = {
        "utility": {
            "function": "linear",
            "attribute_weights": {
                "reward": 1.0,
                "cost": -0.5,
                "time": -0.2
            }
        }
    }
    
    framework = DecisionFramework(config)
    
    options = [
        DecisionOption(
            action="action_1",
            outcomes=[
                {"reward": 0.8, "cost": 0.3, "time": 0.5},
                {"reward": 0.4, "cost": 0.2, "time": 0.3}
            ],
            probabilities=[0.7, 0.3]
        ),
        DecisionOption(
            action="action_2",
            outcomes=[
                {"reward": 0.6, "cost": 0.4, "time": 0.6},
                {"reward": 0.3, "cost": 0.2, "time": 0.4}
            ],
            probabilities=[0.6, 0.4]
        )
    ]
    
    state = {"urgency": 0.5}
    
    for approach in ["expected_utility", "pareto", "behavioral", "optimal_stopping"]:
        result = framework.make_decision(state, options, approach=approach)
        logger.info(f"Approach: {approach}, Decision: {result['decision']}, Reasoning: {result['reasoning']}")

def test_smart_agent():
    """Test the Smart Decision Agent"""
    logger.info("Testing Smart Decision Agent...")
    
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
    
    agent = SmartDecisionAgent(config)
    
    state = {
        "urgency": 0.7,
        "available_resources": ["cpu", "memory"],
        "time_constraint": 30,
        "current_task": "analyze data"
    }
    
    available_actions = [
        "search_web_for_information",
        "analyze_local_document",
        "generate_report",
        "optimize_code"
    ]
    
    # Test action selection
    action = agent.select_action(state, available_actions)
    logger.info(f"Selected action: {action}")
    
    # Test explanation
    explanation = agent.get_decision_explanation(state, action)
    logger.info(f"Basic explanation: {explanation}")
    
    detailed_explanation = agent.get_detailed_explanation(state, action)
    logger.info(f"Detailed explanation: {detailed_explanation}")
    
    # Test learning
    agent.update_policy(state, action, 0.8, {"urgency": 0.3})
    logger.info("Policy updated with reward: 0.8")
    
    # Test different approaches
    logger.info("\nTesting different decision approaches:")
    for approach in ["expected_utility", "pareto", "behavioral", "optimal_stopping"]:
        agent.default_approach = approach
        action = agent.select_action(state, available_actions)
        logger.info(f"Approach: {approach}, Selected action: {action}")
    
    # Get performance stats
    stats = agent.get_performance_stats()
    logger.info(f"Performance stats: {stats}")

def main():
    """Main function"""
    logger.info("Running Smart Decision Agent tests...")
    
    try:
        test_decision_framework()
        test_smart_agent()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
