"""
Behavioral decision making models
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .types import DecisionOption

logger = logging.getLogger(__name__)

class BehavioralModel:
    """
    Implements behavioral decision making models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.reference_point = self.config.get('reference_point', 0.0)
        self.loss_aversion = self.config.get('loss_aversion', 2.0)
        
        logger.info(f"BehavioralModel initialized with reference_point: {self.reference_point}")
    
    def adjust_option(self, option: DecisionOption, state: Dict[str, Any]) -> DecisionOption:
        """
        Adjust option based on behavioral model
        
        Args:
            option: Original decision option
            state: Current state
            
        Returns:
            Adjusted decision option
        """
        # Create adjusted outcomes
        adjusted_outcomes = []
        for outcome in option.outcomes:
            adjusted_outcome = outcome.copy()
            
            # Apply prospect theory utility transformation
            if 'reward' in adjusted_outcome:
                adjusted_outcome['reward'] = self.prospect_utility(adjusted_outcome['reward'])
            
            adjusted_outcomes.append(adjusted_outcome)
        
        return DecisionOption(
            action=option.action,
            outcomes=adjusted_outcomes,
            probabilities=option.probabilities.copy()
        )
    
    def prospect_utility(self, outcome: float) -> float:
        """
        Calculate prospect theory utility
        
        Args:
            outcome: Raw outcome value
            
        Returns:
            Prospect theory utility
        """
        gain_loss = outcome - self.reference_point
        
        if gain_loss >= 0:
            # Gains: concave utility function
            return gain_loss ** 0.88
        else:
            # Losses: convex utility function with loss aversion
            return -self.loss_aversion * (-gain_loss) ** 0.88
    
    def satisficing(
        self, 
        options: List[DecisionOption], 
        threshold: float
    ) -> Optional[DecisionOption]:
        """
        Select first satisficing option
        
        Args:
            options: List of decision options
            threshold: Satisficing threshold
            
        Returns:
            First satisficing option or None
        """
        for option in options:
            # Calculate expected utility
            total_utility = 0.0
            for outcome, prob in zip(option.outcomes, option.probabilities):
                if 'reward' in outcome:
                    utility = self.prospect_utility(outcome['reward'])
                    total_utility += prob * utility
            
            # Check if option meets threshold
            if total_utility >= threshold:
                logger.debug(f"Found satisficing option with utility {total_utility}")
                return option
        
        # No satisficing option found
        return None
    
    def minimax_regret(self, options: List[DecisionOption]) -> DecisionOption:
        """
        Select option using minimax regret criterion
        
        Args:
            options: List of decision options
            
        Returns:
            Option with minimum maximum regret
        """
        # Extract all possible outcomes
        all_outcomes = set()
        for option in options:
            for outcome in option.outcomes:
                if 'reward' in outcome:
                    all_outcomes.add(outcome['reward'])
        
        # Calculate regret matrix
        regret_matrix = []
        for option in options:
            regrets = []
            for outcome_val in all_outcomes:
                # Find best possible outcome for this scenario
                best_outcome = max(
                    o['reward'] for o in option.outcomes if 'reward' in o
                )
                
                # Calculate regret
                regret = best_outcome - outcome_val
                regrets.append(regret)
            
            regret_matrix.append(regrets)
        
        # Find maximum regret for each option
        max_regrets = [max(regrets) for regrets in regret_matrix]
        
        # Select option with minimum maximum regret
        min_max_regret_idx = max_regrets.index(min(max_regrets))
        
        return options[min_max_regret_idx]
