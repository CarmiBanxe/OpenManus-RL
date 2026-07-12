from typing import Optional
"""
Utility calculation module for decision theory
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

class UtilityCalculator:
    """
    Calculates utility for outcomes based on various utility functions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.utility_function = self._get_utility_function()
        logger.info(f"UtilityCalculator initialized with function: {self.utility_function.__name__}")
    
    def _get_utility_function(self) -> Callable:
        """Get the utility function based on configuration"""
        function_name = self.config.get('function', 'linear')
        
        if function_name == 'linear':
            return self._linear_utility
        elif function_name == 'logarithmic':
            return self._logarithmic_utility
        elif function_name == 'exponential':
            return self._exponential_utility
        elif function_name == 'cubic':
            return self._cubic_utility
        else:
            logger.warning(f"Unknown utility function: {function_name}, using linear")
            return self._linear_utility
    
    def _linear_utility(self, value: float) -> float:
        """Linear utility function: U(x) = x"""
        return value
    
    def _logarithmic_utility(self, value: float) -> float:
        """Logarithmic utility function: U(x) = log(x + 1)"""
        return np.log(max(0, value) + 1)
    
    def _exponential_utility(self, value: float) -> float:
        """Exponential utility function: U(x) = exp(x) - 1"""
        return np.exp(value) - 1
    
    def _cubic_utility(self, value: float) -> float:
        """Cubic utility function: U(x) = x^3"""
        return value ** 3
    
    def utility(self, outcome: Dict[str, Any]) -> float:
        """
        Calculate utility for a single outcome
        
        Args:
            outcome: Dictionary with outcome attributes
            
        Returns:
            Utility value
        """
        if 'utility' in outcome:
            # Pre-calculated utility
            return outcome['utility']
        
        if 'reward' in outcome:
            # Use reward as utility
            return self.utility_function(outcome['reward'])
        
        # Calculate utility from multiple attributes
        utility = 0.0
        weights = self.config.get('attribute_weights', {})
        
        for attribute, value in outcome.items():
            if attribute in ['action', 'probabilities']:
                continue  # Skip non-utility attributes
            
            weight = weights.get(attribute, 1.0)
            utility += weight * self.utility_function(value)
        
        return utility
    
    def expected_utility(
        self, 
        action: str, 
        outcomes: List[Dict[str, Any]], 
        probabilities: List[float]
    ) -> float:
        """
        Calculate expected utility for an action
        
        Args:
            action: Action description
            outcomes: List of possible outcomes
            probabilities: Probability of each outcome
            
        Returns:
            Expected utility value
        """
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have the same length")
        
        if not outcomes:
            return 0.0
        
        # Calculate utility for each outcome
        utilities = [self.utility(outcome) for outcome in outcomes]
        
        # Calculate expected utility
        expected_utility = sum(u * p for u, p in zip(utilities, probabilities))
        
        logger.debug(f"Expected utility for {action}: {expected_utility}")
        return expected_utility
    
    def marginal_utility(self, value: float, delta: float = 0.01) -> float:
        """
        Calculate marginal utility at a given point
        
        Args:
            value: Current value
            delta: Small change for numerical differentiation
            
        Returns:
            Marginal utility (derivative of utility function)
        """
        return (self.utility_function(value + delta) - self.utility_function(value - delta)) / (2 * delta)
