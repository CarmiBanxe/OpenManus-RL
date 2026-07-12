"""
Core Decision Framework for OpenManus RL
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .types import DecisionOption
from .utility import UtilityCalculator
from .bellman import BellmanSolver
from .pareto import ParetoOptimizer
from .behavioral import BehavioralModel
from .stopping import OptimalStopping

logger = logging.getLogger(__name__)

class DecisionFramework:
    """
    Core decision-making framework that combines multiple decision theory approaches
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.utility_calculator = UtilityCalculator(self.config.get('utility', {}))
        self.bellman_solver = BellmanSolver(self.config.get('bellman', {}))
        self.pareto_optimizer = ParetoOptimizer(self.config.get('pareto', {}))
        self.behavioral_model = BehavioralModel(self.config.get('behavioral', {}))
        self.optimal_stopping = OptimalStopping(self.config.get('stopping', {}))
        
        logger.info("DecisionFramework initialized with config: %s", self.config)
    
    def make_decision(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]] = None,
        approach: str = "expected_utility"
    ) -> Dict[str, Any]:
        """
        Make a decision based on the current state and available options
        
        Args:
            state: Current state information
            options: List of decision options
            criteria: Criteria for multi-criteria decision making
            approach: Decision approach to use (expected_utility, pareto, behavioral, etc.)
            
        Returns:
            Dictionary with the optimal decision and supporting information
        """
        logger.info(f"Making decision with approach: {approach}")
        
        if approach == "expected_utility":
            return self._expected_utility_approach(state, options, criteria)
        elif approach == "pareto":
            return self._pareto_approach(state, options, criteria)
        elif approach == "behavioral":
            return self._behavioral_approach(state, options, criteria)
        elif approach == "optimal_stopping":
            return self._optimal_stopping_approach(state, options, criteria)
        elif approach == "bellman":
            return self._bellman_approach(state, options, criteria)
        else:
            raise ValueError(f"Unknown decision approach: {approach}")
    
    def _expected_utility_approach(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Expected utility maximization approach"""
        utilities = []
        
        for option in options:
            utility = self.utility_calculator.expected_utility(
                option.action, option.outcomes, option.probabilities
            )
            utilities.append(utility)
        
        # Find option with maximum utility
        max_idx = utilities.index(max(utilities))
        optimal_option = options[max_idx]
        
        return {
            "decision": optimal_option.action,
            "approach": "expected_utility",
            "utility": utilities[max_idx],
            "all_utilities": utilities,
            "reasoning": "Selected option with maximum expected utility"
        }
    
    def _pareto_approach(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Pareto optimization approach"""
        if not criteria:
            # Default criteria for outcomes
            criteria = ["reward", "cost", "time"]
        
        # Extract criteria values for each option
        options_criteria = []
        for option in options:
            criteria_values = []
            for criterion in criteria:
                # Average criterion value across all outcomes
                values = [outcome.get(criterion, 0) for outcome in option.outcomes]
                avg_value = sum(values) / len(values) if values else 0
                criteria_values.append(avg_value)
            options_criteria.append(criteria_values)
        
        # Find Pareto optimal options
        pareto_indices = self.pareto_optimizer.find_pareto_front(options_criteria)
        pareto_options = [options[i] for i in pareto_indices]
        
        if len(pareto_options) == 1:
            selected_option = pareto_options[0]
            reasoning = "Single Pareto optimal option found"
        else:
            # Select from Pareto front using weighted sum
            weights = self.config.get('pareto', {}).get('criteria_weights', [1.0] * len(criteria))
            selected_option = self._select_from_pareto(pareto_options, criteria, weights)
            reasoning = f"Selected from {len(pareto_options)} Pareto optimal options using weighted criteria"
        
        return {
            "decision": selected_option.action,
            "approach": "pareto",
            "pareto_front_size": len(pareto_options),
            "reasoning": reasoning
        }
    
    def _behavioral_approach(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Behavioral decision-making approach"""
        # Apply behavioral model to adjust utilities
        adjusted_options = []
        for option in options:
            adjusted_option = self.behavioral_model.adjust_option(option, state)
            adjusted_options.append(adjusted_option)
        
        # Use satisficing if configured
        if self.config.get('behavioral', {}).get('use_satisficing', False):
            threshold = self.config.get('behavioral', {}).get('satisficing_threshold', 0.5)
            selected_option = self.behavioral_model.satisficing(adjusted_options, threshold)
            reasoning = f"Selected first satisficing option with threshold {threshold}"
        else:
            # Use behavioral adjusted utilities
            utilities = []
            for option in adjusted_options:
                utility = self.utility_calculator.expected_utility(
                    option.action, option.outcomes, option.probabilities
                )
                utilities.append(utility)
            
            max_idx = utilities.index(max(utilities))
            selected_option = adjusted_options[max_idx]
            reasoning = "Selected option with maximum behaviorally-adjusted utility"
        
        return {
            "decision": selected_option.action,
            "approach": "behavioral",
            "reasoning": reasoning
        }
    
    def _optimal_stopping_approach(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Optimal stopping approach"""
        # This is applicable when options are presented sequentially
        # For now, we'll use the 37% rule
        
        if len(options) < 3:
            # Not enough options for optimal stopping, use expected utility
            return self._expected_utility_approach(state, options, criteria)
        
        # Calculate expected utilities for all options
        utilities = []
        for option in options:
            utility = self.utility_calculator.expected_utility(
                option.action, option.outcomes, option.probabilities
            )
            utilities.append(utility)
        
        # Apply optimal stopping rule
        selected_index = self.optimal_stopping.secretary_problem(utilities)
        selected_option = options[selected_index]
        
        return {
            "decision": selected_option.action,
            "approach": "optimal_stopping",
            "reasoning": f"Applied optimal stopping rule, selected option at index {selected_index}"
        }
    
    def _bellman_approach(
        self, 
        state: Dict[str, Any], 
        options: List[DecisionOption],
        criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Bellman equation approach for sequential decisions"""
        # This is a simplified implementation
        # In practice, you'd need to define the full MDP structure
        
        # Extract rewards for each option
        rewards = []
        for option in options:
            # Use first outcome's reward as immediate reward
            immediate_reward = option.outcomes[0].get('reward', 0) if option.outcomes else 0
            rewards.append(immediate_reward)
        
        # Apply Bellman equation (simplified)
        values = self.bellman_solver.bellman_equation(state, options, rewards)
        
        # Select option with maximum value
        max_idx = values.index(max(values))
        selected_option = options[max_idx]
        
        return {
            "decision": selected_option.action,
            "approach": "bellman",
            "value": values[max_idx],
            "reasoning": "Selected option with maximum Bellman value"
        }
    
    def _select_from_pareto(
        self, 
        pareto_options: List[DecisionOption], 
        criteria: List[str],
        weights: List[float]
    ) -> DecisionOption:
        """Select from Pareto front using weighted criteria"""
        best_option = None
        best_score = float('-inf')
        
        for option in pareto_options:
            score = 0.0
            for i, criterion in enumerate(criteria):
                # Average criterion value across outcomes
                values = [outcome.get(criterion, 0) for outcome in option.outcomes]
                avg_value = sum(values) / len(values) if values else 0
                
                # Apply weight
                if i < len(weights):
                    score += weights[i] * avg_value
                else:
                    score += avg_value
            
            if score > best_score:
                best_score = score
                best_option = option
        
        return best_option
