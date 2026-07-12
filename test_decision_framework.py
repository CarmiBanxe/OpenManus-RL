#!/usr/bin/env python3
"""
Test script for Decision Framework
"""

import sys
import logging
from pathlib import Path

# Add OpenManus to path
sys.path.append(str(Path(__file__).parent))

from openmanus_rl.decision.core import DecisionFramework, DecisionOption
from openmanus_rl.agents.decision_agent import DecisionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_decision_framework():
    """Test the Decision Framework"""
    logger.info("Testing Decision Framework...")
    
    # Create decision framework
    config = {
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
            "satisficing_threshold": 0.5
        }
    }
    
    framework = DecisionFramework(config)
    
    # Create test options
    options = [
        DecisionOption(
            action="action_1",
            outcomes=[
                {"reward": 1.0, "cost": 0.5, "time": 1.0},
                {"reward": 0.5, "cost": 0.3, "time": 0.8}
            ],
            probabilities=[0.7, 0.3]
        ),
        DecisionOption(
            action="action_2",
            outcomes=[
                {"reward": 1.5, "cost": 0.8, "time": 1.2},
                {"reward": 0.2, "cost": 0.1, "time": 0.5}
            ],
            probabilities=[0.6, 0.4]
        )
    ]
    
    # Test different approaches
    state = {"urgency": 0.5}
    
    for approach in ["expected_utility", "pareto", "behavioral", "optimal_stopping"]:
        try:
            result = framework.make_decision(state, options, approach=approach)
            logger.info(f"Approach: {approach}, Decision: {result['decision']}, Reasoning: {result['reasoning']}")
        except Exception as e:
            logger.error(f"Error with approach {approach}: {str(e)}")

def test_decision_agent():
    """Test the Decision Agent"""
    logger.info("Testing Decision Agent...")
    
    # Create agent
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
                "satisficing_threshold": 0.5
            }
        },
        "learning_rate": 0.01
    }
    
    agent = DecisionAgent(config)
    
    # Test action selection
    state = {"urgency": 0.5}
    available_actions = ["action_1", "action_2", "action_3"]
    
    selected_action = agent.select_action(state, available_actions)
    logger.info(f"Selected action: {selected_action}")
    
    # Get explanation
    explanation = agent.get_decision_explanation(state, selected_action)
    if explanation:
        logger.info(f"Explanation: {explanation}")
    else:
        logger.info("No explanation available")
    
    # Test policy update
    agent.update_policy(state, selected_action, 0.8, {"urgency": 0.3})
    logger.info("Policy updated with reward: 0.8")

if __name__ == "__main__":
    logger.info("Running Decision Framework tests...")
    
    try:
        test_decision_framework()
        test_decision_agent()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)
