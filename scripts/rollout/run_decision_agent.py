#!/usr/bin/env python3
"""
Run script for decision-enhanced agent in OpenManus RL
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add OpenManus to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openmanus_rl.agents.decision_agent import DecisionAgent
from openmanus_rl.environments import get_environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run decision-enhanced agent")
    parser.add_argument("--env_name", type=str, default="alfworld", help="Environment name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--total_envs", type=int, default=10, help="Total environments")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--decision_approach", type=str, default="expected_utility", 
                        choices=["expected_utility", "pareto", "behavioral", "optimal_stopping", "bellman"],
                        help="Decision approach to use")
    
    args = parser.parse_args()
    
    # Create agent
    agent_config = {
        "decision": {
            "default_approach": args.decision_approach,
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
    
    agent = DecisionAgent(agent_config)
    
    # Create environment
    env = get_environment(args.env_name)
    
    logger.info(f"Running decision agent with approach: {args.decision_approach}")
    logger.info(f"Environment: {args.env_name}")
    
    # Run episodes
    total_reward = 0
    for episode in range(args.total_envs):
        state = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
            # Get available actions
            available_actions = env.get_available_actions(state)
            
            # Select action using decision theory
            action = agent.select_action(state, available_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update_policy(state, action, reward, next_state)
            
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
        
        total_reward += episode_reward
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    logger.info(f"Average reward: {total_reward / args.total_envs}")

if __name__ == "__main__":
    main()
