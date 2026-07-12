"""
Decision-enhanced agent for OpenManus RL
"""

import logging
from typing import Dict, Any, List, Optional
from ..decision.core import DecisionFramework
from ..decision.types import DecisionOption

logger = logging.getLogger(__name__)

class DecisionAgent:
    """
    Agent enhanced with decision theory capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.decision_framework = DecisionFramework(self.config.get('decision', {}))
        self.memory = []
        self.learning_rate = self.config.get('learning_rate', 0.01)
        
        # Default decision approach
        self.default_approach = self.config.get('default_approach', 'expected_utility')
        
        logger.info(f"DecisionAgent initialized with approach: {self.default_approach}")
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Select an action using decision theory
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        logger.info(f"Selecting action from {len(available_actions)} options")
        
        # Convert actions to decision options
        options = []
        for action in available_actions:
            # Predict outcomes for each action
            outcomes = self._predict_outcomes(state, action)
            probabilities = [1.0] * len(outcomes)  # Equal probabilities for now
            
            option = DecisionOption(
                action=action,
                outcomes=outcomes,
                probabilities=probabilities
            )
            options.append(option)
        
        # Make decision
        decision_result = self.decision_framework.make_decision(
            state=state,
            options=options,
            approach=self.default_approach
        )
        
        selected_action = decision_result['decision']
        
        # Store in memory
        self.memory.append({
            'state': state,
            'action': selected_action,
            'decision_result': decision_result
        })
        
        logger.info(f"Selected action: {selected_action}")
        return selected_action
    
    def _predict_outcomes(self, state: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
        """
        Predict outcomes for an action
        
        Args:
            state: Current state
            action: Action to predict outcomes for
            
        Returns:
            List of predicted outcomes
        """
        # This is a simplified implementation
        # In practice, you would use a model to predict outcomes
        
        # Default outcomes with different reward values
        outcomes = [
            {'reward': 1.0, 'cost': 0.5, 'time': 1.0},
            {'reward': 0.5, 'cost': 0.3, 'time': 0.8},
            {'reward': 0.0, 'cost': 0.1, 'time': 0.5}
        ]
        
        # Adjust outcomes based on action and state
        for outcome in outcomes:
            # Simple heuristic: more complex actions have higher potential rewards
            if 'search' in action.lower() or 'analyze' in action.lower():
                outcome['reward'] *= 1.5
                outcome['time'] *= 1.2
            
            # Adjust based on state
            if state.get('urgency', 0) > 0.5:
                outcome['time'] *= 0.8  # Faster outcomes for urgent states
        
        return outcomes
    
    def update_policy(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any]):
        """
        Update policy based on experience
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
        """
        # Find the decision in memory
        for i, memory_item in enumerate(self.memory):
            if (memory_item['state'] == state and 
                memory_item['action'] == action):
                
                # Update with actual reward
                self.memory[i]['actual_reward'] = reward
                
                # Simple learning: adjust decision approach if reward is low
                if reward < 0:
                    # Try a different approach next time
                    approaches = ['expected_utility', 'pareto', 'behavioral']
                    current_idx = approaches.index(self.default_approach)
                    self.default_approach = approaches[(current_idx + 1) % len(approaches)]
                    logger.info(f"Changed decision approach to {self.default_approach} due to low reward")
                
                break
    
    def get_decision_explanation(self, state: Dict[str, Any], action: str) -> Optional[str]:
        """
        Get explanation for a decision
        
        Args:
            state: State where decision was made
            action: Action that was selected
            
        Returns:
            Explanation string or None if not found
        """
        for memory_item in self.memory:
            if (memory_item['state'] == state and 
                memory_item['action'] == action and
                'decision_result' in memory_item):
                
                decision_result = memory_item['decision_result']
                return f"Approach: {decision_result.get('approach', 'unknown')}, " \
                       f"Reasoning: {decision_result.get('reasoning', 'no reasoning provided')}"
        
        return None
